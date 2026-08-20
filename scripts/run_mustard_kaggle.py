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


def run(cmd, **kwargs):
    """Run a subprocess, streaming output live and raising on error."""
    print("\n$ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **kwargs,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def download_mustard_clips(clips_dir):
    """Download and extract MUStARD raw video/audio clips from HuggingFace.

    The zip contains the raw mp4 video clips. The dataset loader (MUStARDDataset)
    can read audio directly from mp4 via torchaudio/soundfile, so no separate
    audio extraction step is required.
    """
    # Check if already extracted -- look for any media file as a proxy
    if clips_dir.exists():
        media_files = list(clips_dir.rglob("*.mp4")) + list(clips_dir.rglob("*.wav"))
        if media_files:
            print(f"[MUStARD] Already extracted ({len(media_files)} clips) at {clips_dir}, skipping download.")
            return

    clips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = clips_dir.parent / "mmsd.zip"

    url = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    print(f"[MUStARD] Downloading clips from {url} ...")
    run(["wget", "-q", "--show-progress", "-O", str(zip_path), url])

    print("[MUStARD] Extracting ...")
    run(["unzip", "-q", "-o", str(zip_path), "-d", str(clips_dir)])
    zip_path.unlink()

    media_files = list(clips_dir.rglob("*.mp4")) + list(clips_dir.rglob("*.wav"))
    if not media_files:
        raise RuntimeError(f"[MUStARD] No media files found after extraction in {clips_dir}")
    print(f"[MUStARD] {len(media_files)} clips ready.")


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

    # Step 1: Download MUStARD clips
    if not args.skip_download:
        download_mustard_clips(clips_dir)

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
