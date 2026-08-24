#!/usr/bin/env python3
"""Self-contained Kaggle runner for ConflictNet MELD training.

Downloads MELD, warms up pretrained models, trains, and evaluates --
all in one script. Designed to be called from a single Kaggle cell.

Usage (from repo root, after cloning):
    python scripts/run_meld_kaggle.py \\
        --meld_root /kaggle/working/meld \\
        --output_dir /kaggle/working/output_meld \\
        [--audio_encoder wavlm_weighted] \\
        [--no_eval]
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

# Fix env before any torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Point HF / SpeechBrain caches to /tmp so the working dir stays clean
HF_CACHE = "/tmp/hf_cache"
Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = str(Path(HF_CACHE) / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_CACHE) / "hub")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Emotion2Vec is downloaded by FunASR/ModelScope, not huggingface_hub.
os.environ["MODELSCOPE_CACHE"] = str(Path(HF_CACHE) / "modelscope")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def run(cmd, silent=False, **kwargs):
    """Run a subprocess cleanly, streaming output live unless silent."""
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **kwargs,
    )
    for line in iter(proc.stdout.readline, ""):
        if not silent:
            print(line, end="", flush=True)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def download_fast(url: str, output_path: Path) -> None:
    """Fast multi-connection download using aria2c (16 parallel streams) or curl quietly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_aria2 = subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
    if not has_aria2:
        try:
            subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            has_aria2 = subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
        except Exception:
            has_aria2 = False

    print("⚡ Fast downloading archive (16 parallel connections)...", flush=True)
    if has_aria2:
        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "1M",
            "--file-allocation=none",
            "--console-log-level=warn",
            "--summary-interval=0",
            "--quiet=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "-d", str(output_path.parent),
            "-o", output_path.name,
            url,
        ]
        run(cmd, silent=True)
    else:
        cmd = ["curl", "-sL", "-o", str(output_path), url]
        run(cmd, silent=True)


MELD_KAGGLE_INPUTS = [
    Path("/kaggle/input/datasets/nith27/meld-datasets/MELD.Raw"),
    Path("/kaggle/input/datasets/nith27/meld-datasets"),
    Path("/kaggle/input/datasets/nith27/meld-dataset/MELD.Raw"),
    Path("/kaggle/input/datasets/nith27/meld-dataset"),
    Path("/kaggle/input/notebooks/nith27/meld-dataset/MELD.Raw"),
    Path("/kaggle/input/notebooks/nith27/meld-dataset"),
    Path("/kaggle/input/meld-dataset/MELD.Raw"),
    Path("/kaggle/input/meld-dataset"),
    Path("/kaggle/input/meld-datasets/MELD.Raw"),
    Path("/kaggle/input/meld-datasets"),
]


def find_meld_root(requested_root: Path, explicit_mount: Optional[str] = None) -> Path:
    """Find MELD dataset from explicit mount, Kaggle mount paths, or requested_root."""
    # Never write into /kaggle/input (read-only filesystem)
    if str(requested_root).startswith("/kaggle/input"):
        target_root = Path("/kaggle/working/meld")
    else:
        target_root = requested_root

    # Check if target already has prepared dataset
    if (target_root / "train_sent_emo.csv").exists() or (target_root / "train" / "train_sent_emo.csv").exists():
        return target_root

    search_mounts: list[Path] = []
    if explicit_mount:
        search_mounts.append(Path(explicit_mount))
    if str(requested_root).startswith("/kaggle/input"):
        search_mounts.append(requested_root)
    search_mounts.extend(MELD_KAGGLE_INPUTS)

    # Also check any dataset attached in /kaggle/input (top 2 levels)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        try:
            for p in kaggle_input.glob("*/*"):
                if p.is_dir() and p not in search_mounts:
                    search_mounts.append(p)
            for p in kaggle_input.glob("*"):
                if p.is_dir() and p not in search_mounts:
                    search_mounts.append(p)
        except Exception:
            pass

    # Check mounted Kaggle inputs
    mounted_path = None
    for candidate in search_mounts:
        if candidate.exists():
            if (candidate / "MELD.Raw").exists() and candidate.name != "MELD.Raw":
                candidate = candidate / "MELD.Raw"
            # Verify candidate has MELD data
            if (candidate / "train.tar.gz").exists() or (candidate / "train_sent_emo.csv").exists() or (candidate / "dev_sent_emo.csv").exists():
                mounted_path = candidate
                break

    if mounted_path is not None:
        import shutil
        target_root.mkdir(parents=True, exist_ok=True)
        print(f"[MELD] 🚀 Found mounted Kaggle dataset at {mounted_path}")

        # Copy CSVs if directly in mounted dir
        for csv_name in ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]:
            src = mounted_path / csv_name
            dst = target_root / csv_name
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))

        # Unpack inner tar archives (train.tar.gz, dev.tar.gz, test.tar.gz) into working dir
        for tar_name in ["train.tar.gz", "dev.tar.gz", "test.tar.gz"]:
            tar_file = mounted_path / tar_name
            if tar_file.exists():
                print(f"[MELD] 📦 Unpacking {tar_name} into {target_root} ...", flush=True)
                run(["tar", "-xzf", str(tar_file), "-C", str(target_root)])

        # Copy pre-extracted audio split directories (train/, dev/, test/) if they exist
        # as directories rather than tar archives (some Kaggle datasets ship pre-extracted).
        for split in ["train", "dev", "test"]:
            src_split = mounted_path / split
            dst_split = target_root / split
            if src_split.exists() and src_split.is_dir() and not dst_split.exists():
                print(f"[MELD] 📂 Copying pre-extracted {split}/ ({len(list(src_split.rglob('*')))} files) ...", flush=True)
                shutil.copytree(str(src_split), str(dst_split))

        # Ensure CSVs are at target_root (search subdirs/parent or fetch 1MB CSV directly)
        for csv_name in ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]:
            dst = target_root / csv_name
            if not dst.exists():
                found = False
                for search_dir in [mounted_path.parent, target_root / "train", target_root / "dev", target_root / "test"]:
                    if (search_dir / csv_name).exists():
                        shutil.copy2(str(search_dir / csv_name), str(dst))
                        found = True
                        break
                if not found:
                    csv_url = f"https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{csv_name}"
                    print(f"[MELD] 📥 Fetching missing metadata {csv_name} (1 MB) from official repository ...", flush=True)
                    try:
                        import urllib.request
                        urllib.request.urlretrieve(csv_url, str(dst))
                    except Exception as e:
                        print(f"[MELD] Warning: could not fetch {csv_name}: {e}")

        return target_root

    return target_root


def download_meld(meld_root: Path, explicit_mount: Optional[str] = None) -> Path:
    """Prepare MELD from mounted dataset or download if missing."""
    target_root = find_meld_root(meld_root, explicit_mount=explicit_mount)
    has_csv = (target_root / "train_sent_emo.csv").exists() or (target_root / "train" / "train_sent_emo.csv").exists()
    has_videos = (target_root / "train_splits").exists() or (target_root / "train").exists() or (target_root / "dev_splits_complete").exists()

    if has_csv:
        print(f"[MELD] Dataset ready at {target_root}.")
        return target_root

    # If videos exist but CSV was missing, fetch CSV
    if has_videos and not has_csv:
        for csv_name in ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]:
            dst = target_root / csv_name
            if not dst.exists():
                csv_url = f"https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{csv_name}"
                try:
                    import urllib.request
                    urllib.request.urlretrieve(csv_url, str(dst))
                except Exception:
                    pass
        print(f"[MELD] Dataset ready at {target_root}.")
        return target_root

    target_root.mkdir(parents=True, exist_ok=True)
    tar_path = Path("/tmp/MELD.Raw.tar.gz")

    url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    download_fast(url, tar_path)

    print("[MELD] Extracting archive...")
    run(["tar", "-xzf", str(tar_path), "-C", str(target_root), "--strip-components=1"])
    if tar_path.exists():
        tar_path.unlink()

    print("[MELD] ✅ Dataset ready.")
    return target_root


def warmup_models(audio_encoder, no_word_divergence):
    """Pre-download and initialise all pretrained models in a single process
    before launching torchrun to prevent parallel download races."""
    print("\n[Warmup] Initialising DeBERTa+LoRA, WavLM/Emotion2Vec, ECAPA-TDNN ...")
    from models.conflictnet import ConflictNet
    model = ConflictNet(
        audio_encoder_name=audio_encoder,
        embed_dim=256,
        use_word_divergence=not no_word_divergence,
        lora_r=16,
        lora_alpha=32,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Warmup] Warm-up complete -- trainable params: {trainable:,}")
    del model
    gc.collect()


def main():
    p = argparse.ArgumentParser(description="Kaggle MELD training runner")
    p.add_argument("--meld_root", type=str, default="/kaggle/working/meld",
                   help="Where to download/extract MELD")
    p.add_argument("--meld_mount", type=str, default=None,
                   help="Explicit path to mounted MELD dataset (e.g. /kaggle/input/notebooks/nith27/meld-dataset/MELD.Raw)")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/output_meld")
    p.add_argument("--audio_encoder", type=str, default="wavlm_weighted",
                   choices=["emotion2vec", "wavlm", "wavlm_weighted", "wav2vec2"])
    p.add_argument("--no_eval", action="store_true",
                   help="Skip evaluation after training")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip MELD download (dataset already at --meld_root)")
    args = p.parse_args()

    meld_root = Path(args.meld_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download / locate MELD
    if not args.skip_download:
        meld_root = download_meld(meld_root, explicit_mount=args.meld_mount)
    else:
        meld_root = find_meld_root(meld_root, explicit_mount=args.meld_mount)

    # Step 2: Warm up pretrained models (single-process, before torchrun)
    import torch
    os.chdir(REPO)  # SpeechBrain writes pretrained_models/ relative to cwd
    warmup_models(args.audio_encoder, no_word_divergence=True)

    # Step 3: Train
    n_gpus = torch.cuda.device_count() or 1
    print(f"\n[Train] Launching on {n_gpus} GPU(s) ...\n")

    # Hyperparameters deliberately match MUStARD baseline for fair comparison.
    # Changes vs MUStARD:
    #   warmup_steps: 500 -> 200
    #       MELD train split is capped to 560 samples -> ~140 steps/epoch.
    #       200 steps ~= 1.4 epochs of warmup, proportionally similar to
    #       MUStARD's 500 steps over a ~560-sample training set.
    #   meld_max_samples: 700 total (stratified 80/20 -> ~560 train / ~140 val)
    #       Matches MUStARD's dataset scale exactly.
    # Everything else (lr, batch_size, epochs, grad_accum, amp, label_smoothing,
    # early_stop_patience) is kept identical to the MUStARD baseline run.
    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(REPO / "scripts" / "train.py"),
        "--meld_root",                   str(meld_root),
        "--meld_max_samples",            "1200",
        "--epochs",                      "40",
        "--pretrain_epochs",             "3",
        "--batch_size",                  "4",
        "--lr",                          "1e-4",
        "--warmup_steps",                "100",
        "--audio_encoder",               args.audio_encoder,
        "--gradient_accumulation_steps", "2",
        "--amp",
        "--label_smoothing",             "0.05",
        "--early_stop_patience",         "15",
        "--no_word_divergence",
        "--output_dir",                  str(output_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    run(cmd, env=env)
    print("\n[Train] Training complete.")

    # Step 4: Evaluate on MELD test set
    if not args.no_eval:
        ckpt = output_dir / "best_model.safetensors"
        if not ckpt.exists():
            candidates = list(output_dir.glob("*.safetensors"))
            if candidates:
                ckpt = candidates[0]
            else:
                print("[Eval] No checkpoint found -- skipping evaluation.")
                return

        print(f"\n[Eval] Evaluating checkpoint: {ckpt}\n")
        eval_cmd = [
            sys.executable,
            str(REPO / "scripts" / "evaluate.py"),
            "--checkpoint",    str(ckpt),
            "--meld_root",     str(meld_root),
            "--audio_encoder", args.audio_encoder,
            "--batch_size",    "16",
            "--output_dir",    str(output_dir / "eval_results"),
        ]
        run(eval_cmd, env=env)
        print("\n[Eval] Evaluation complete. Results in:", output_dir / "eval_results")

    # Step 5: Copy outputs to /kaggle/working/ for download
    import shutil
    print("\n[Save] Moving output files to /kaggle/working/ ...")
    for pattern in ["*.safetensors", "*.pt", "*.pth", "*_meta.json", "*.log"]:
        for f in output_dir.glob(pattern):
            dest = Path("/kaggle/working") / f.name
            shutil.copy2(str(f), str(dest))
            print(f"  Copied: {dest}")
    print("\nDone -- download from the Output tab.")


if __name__ == "__main__":
    main()
