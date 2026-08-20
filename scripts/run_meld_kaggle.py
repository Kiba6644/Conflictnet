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

# Point HF / SpeechBrain caches to /tmp so the working dir stays clean
HF_CACHE = "/tmp/hf_cache"
Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = str(Path(HF_CACHE) / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_CACHE) / "hub")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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


def find_meld_root(requested_root: Path) -> Path:
    """Find MELD dataset if already extracted or mounted in /kaggle/input."""
    if (requested_root / "train" / "train_sent_emo.csv").exists():
        return requested_root

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train_sent_emo.csv"):
            candidate = p.parent.parent
            if (candidate / "dev" / "dev_sent_emo.csv").exists() or (candidate / "train" / "train_sent_emo.csv").exists():
                print(f"[MELD] 🚀 Found attached Kaggle dataset at {candidate} (0s load time)")
                return candidate

    return requested_root


def download_meld(meld_root: Path) -> Path:
    """Download and extract MELD using high-speed multi-connection transfer."""
    meld_root = find_meld_root(meld_root)
    if (meld_root / "train" / "train_sent_emo.csv").exists():
        print(f"[MELD] Dataset ready at {meld_root}, skipping download.")
        return meld_root

    meld_root.mkdir(parents=True, exist_ok=True)
    tar_path = meld_root.parent / "MELD.Raw.tar.gz"

    url = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    download_fast(url, tar_path)

    print("[MELD] Extracting archive...")
    run(["tar", "-xzf", str(tar_path), "-C", str(meld_root), "--strip-components=1"])
    if tar_path.exists():
        tar_path.unlink()

    # Verify expected structure
    for split, csv_name in [
        ("train", "train_sent_emo.csv"),
        ("dev", "dev_sent_emo.csv"),
        ("test", "test_sent_emo.csv"),
    ]:
        p = meld_root / split / csv_name
        if not p.exists():
            raise RuntimeError(f"[MELD] Expected file not found after extraction: {p}")
    print("[MELD] ✅ Dataset ready.")
    return meld_root


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
        meld_root = download_meld(meld_root)
    else:
        meld_root = find_meld_root(meld_root)

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
        "--meld_max_samples",            "700",
        "--epochs",                      "30",
        "--pretrain_epochs",             "5",
        "--batch_size",                  "4",
        "--lr",                          "5e-5",
        "--warmup_steps",                "200",
        "--audio_encoder",               args.audio_encoder,
        "--gradient_accumulation_steps", "2",
        "--amp",
        "--label_smoothing",             "0.05",
        "--early_stop_patience",         "5",
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
