#!/usr/bin/env python3
"""Self-contained Kaggle runner for ConflictNet MUStARD training.

Downloads the MUStARD raw video/audio clips, warms up pretrained models,
trains, and evaluates -- all in one script.

Usage (from repo root, after cloning):
    python scripts/run_mustard_kaggle.py \\
        --mustard_clips_dir /kaggle/working/mustard_clips \\
        --output_dir /kaggle/working/output_mustard \\
        [--audio_encoder wavlm_weighted] \\
        [--no_eval]

The mustard_root (JSON metadata) is taken from data/mustard/ inside the repo
and does NOT need to be downloaded -- it is already committed to the repo.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# MUStARD metadata JSON is already in the repo at data/mustard/
MUSTARD_ROOT = REPO / "data" / "mustard"


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


MUSTARD_KAGGLE_INPUTS = [
    Path("/kaggle/input/mustard-sarcasm-detection/MUStARD/data/clips"),
    Path("/kaggle/input/mustard-sarcasm-detection/MUStARD/data"),
    Path("/kaggle/input/mustard-sarcasm-detection"),
    Path("/kaggle/input/datasets/nith27/mustard-sarcasm-detection"),
    Path("/kaggle/input/datasets/nith27/mustard-dataset"),
    Path("/kaggle/input/notebooks/nith27/mustard-sarcasm-detection"),
]


def find_mustard_clips_dir(requested_dir: Path, explicit_mount: Optional[str] = None) -> Path:
    """Find MUStARD clips from explicit mount, hardcoded Kaggle paths, or requested_dir."""
    if explicit_mount:
        exp_path = Path(explicit_mount)
        if exp_path.exists():
            print(f"[MUStARD] 🚀 Found explicit mounted clips at {exp_path}")
            return exp_path

    if requested_dir.exists() and any(requested_dir.iterdir()):
        return requested_dir

    for candidate in MUSTARD_KAGGLE_INPUTS:
        if candidate.exists():
            print(f"[MUStARD] 🚀 Found attached Kaggle dataset at {candidate} (0s load time)")
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        try:
            for p in kaggle_input.glob("*/*"):
                if p.is_dir() and (p.name == "clips" or (p / "clips").exists()):
                    return p / "clips" if (p / "clips").exists() else p
        except Exception:
            pass

    return requested_dir


def download_mustard_clips(clips_dir: Path, explicit_mount: Optional[str] = None) -> Path:
    """Download and extract MUStARD raw video clips with multi-stream high speed."""
    clips_dir = find_mustard_clips_dir(clips_dir, explicit_mount=explicit_mount)
    if clips_dir.exists():
        media_files = list(clips_dir.rglob("*.mp4")) + list(clips_dir.rglob("*.wav"))
        if media_files:
            print(f"[MUStARD] Already available ({len(media_files)} clips) at {clips_dir}, skipping download.")
            return clips_dir

    if str(clips_dir).startswith("/kaggle/input"):
        target_clips_dir = Path("/kaggle/working/mustard_clips")
    else:
        target_clips_dir = clips_dir

    target_clips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path("/tmp/mmsd.zip")

    url = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    download_fast(url, zip_path)

    print("[MUStARD] Extracting clips archive...")
    run(["unzip", "-q", "-o", str(zip_path), "-d", str(target_clips_dir)])
    if zip_path.exists():
        zip_path.unlink()

    media_files = list(target_clips_dir.rglob("*.mp4")) + list(target_clips_dir.rglob("*.wav"))
    if not media_files:
        raise RuntimeError(f"[MUStARD] No media files found after extraction in {target_clips_dir}")
    print(f"[MUStARD] ✅ {len(media_files)} clips ready.")
    return target_clips_dir


def warmup_models(audio_encoder):
    """Pre-download and initialise all pretrained models in a single process
    before launching torchrun to prevent parallel download races."""
    print("\n[Warmup] Initialising DeBERTa+LoRA, WavLM/Emotion2Vec, ECAPA-TDNN ...")
    from models.conflictnet import ConflictNet
    model = ConflictNet(
        audio_encoder_name=audio_encoder,
        embed_dim=256,
        use_word_divergence=False,
        lora_r=16,
        lora_alpha=32,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Warmup] Warm-up complete -- trainable params: {trainable:,}")
    del model
    gc.collect()


def main():
    p = argparse.ArgumentParser(description="Kaggle MUStARD training runner")
    p.add_argument("--mustard_clips_dir", type=str, default="/kaggle/working/mustard_clips",
                   help="Where to download/extract MUStARD video clips")
    p.add_argument("--mustard_mount", type=str, default=None,
                   help="Explicit path to mounted MUStARD clips (e.g. /kaggle/input/mustard-sarcasm-detection/MUStARD/data/clips)")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/output_mustard")
    p.add_argument("--audio_encoder", type=str, default="wavlm_weighted",
                   choices=["emotion2vec", "wavlm", "wavlm_weighted", "wav2vec2"])
    p.add_argument("--no_eval", action="store_true",
                   help="Skip evaluation after training")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip clip download (already present at --mustard_clips_dir)")
    args = p.parse_args()

    clips_dir = Path(args.mustard_clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download / locate MUStARD clips
    if not args.skip_download:
        clips_dir = download_mustard_clips(clips_dir, explicit_mount=args.mustard_mount)
    else:
        clips_dir = find_mustard_clips_dir(clips_dir, explicit_mount=args.mustard_mount)

    # Step 2: Warm up pretrained models (single-process, before torchrun)
    import torch
    os.chdir(REPO)  # SpeechBrain writes pretrained_models/ relative to cwd
    warmup_models(args.audio_encoder)

    # Step 3: Train
    n_gpus = torch.cuda.device_count() or 1
    print(f"\n[Train] Launching on {n_gpus} GPU(s) ...\n")

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(REPO / "scripts" / "train.py"),
        "--mustard_root",                str(MUSTARD_ROOT),
        "--mustard_wav_dir",             str(clips_dir),
        "--epochs",                      "30",
        "--pretrain_epochs",             "5",
        "--batch_size",                  "4",
        "--lr",                          "5e-5",
        "--warmup_steps",                "500",
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

    # Step 4: Evaluate on MUStARD val split
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
            "--checkpoint",      str(ckpt),
            "--mustard_root",    str(MUSTARD_ROOT),
            "--mustard_wav_dir", str(clips_dir),
            "--audio_encoder",   args.audio_encoder,
            "--batch_size",      "16",
            "--output_dir",      str(output_dir / "eval_results"),
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
