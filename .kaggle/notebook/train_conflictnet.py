#!/usr/bin/env python3
"""ConflictNet Training Script for Kaggle GPU (T4 / AMP optimized).
Clones the ConflictNet repo from GitHub at runtime — no code dataset needed.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "ConflictNet"
CREMAD_DIR = WORK_DIR / "cremad"
OUTPUT_DIR = WORK_DIR / "output"
HF_CACHE = WORK_DIR / "hf_cache"

GITHUB_REPO = "https://github.com/DevodG/ConflictNet.git"


def install_deps():
    logger.info("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "sentencepiece", "tiktoken", "safetensors",
         "speechbrain", "audiomentations", "praat-parselmouth",
         "captum", "fairlearn", "hydra-core", "evaluate",
         "peft>=0.7", "funasr"],
        check=False,
    )
    logger.info("Dependencies installed.")


def clone_repo():
    logger.info(f"Cloning {GITHUB_REPO} ...")
    if REPO_DIR.exists():
        logger.info("Repo already exists, pulling latest...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth=1", GITHUB_REPO, str(REPO_DIR)], check=True)
    logger.info(f"Repo ready at {REPO_DIR}")


def install_repo_deps():
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        logger.info("Installing repo requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=False,
        )


def find_cremad():
    """Find CREMA-D AudioWAV directory in Kaggle input."""
    CREMAD_DIR.mkdir(parents=True, exist_ok=True)
    link_target = CREMAD_DIR / "AudioWAV"
    if link_target.exists():
        return  # already set up

    for p in KAGGLE_INPUT.rglob("AudioWAV"):
        if p.is_dir():
            link_target.symlink_to(p.resolve())
            logger.info(f"CREMA-D found at {p}")
            return

    logger.error("CREMA-D AudioWAV not found in /kaggle/input!")
    sys.exit(1)


def setup_env():
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
    os.environ["PYTHONPATH"] = f"{REPO_DIR}:{os.environ.get('PYTHONPATH', '')}"
    # Performance Optimization: PyTorch CUDA / cuDNN benchmarking & memory allocator settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_training():
    logger.info("Starting training...")
    train_script = REPO_DIR / "scripts" / "train.py"
    if not train_script.exists():
        logger.error(f"train.py not found at {train_script}")
        sys.exit(1)

    prosody_stats = REPO_DIR / "prosody_stats.json"

    # Optimizations added:
    # 1. num_workers 4 for fast parallel PyTorch data loading on Kaggle CPUs
    # 2. pin_memory enabled via PyTorch DataLoader inside trainer if applicable
    # 3. AMP fp16 enabled for T4 Tensor Cores speedup
    args = [
        sys.executable, str(train_script),
        "--cremad_root",               str(CREMAD_DIR),
        "--epochs",                    "30",
        "--batch_size",                "16",
        "--lr",                        "5e-5",
        "--audio_encoder",             "emotion2vec",
        "--no_word_divergence",
        "--gradient_accumulation_steps", "1",
        "--pretrain_epochs",           "5",
        "--amp",
        "--num_workers",               "4",
        "--output_dir",                str(OUTPUT_DIR),
    ]

    if prosody_stats.exists():
        args += ["--prosody_stats", str(prosody_stats)]

    logger.info(f"Command: {' '.join(args)}")
    env = os.environ.copy()
    result = subprocess.run(args, cwd=str(REPO_DIR), env=env)
    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    logger.info("Training completed!")


def save_outputs():
    logger.info("Saving outputs to /kaggle/working/ for download...")
    for pattern in ["*.safetensors", "*.pt", "*.pth", "*_meta.json", "*.log"]:
        for p in OUTPUT_DIR.glob(pattern):
            dest = WORK_DIR / p.name
            p.rename(dest)
            logger.info(f"  Saved: {dest}")
    logger.info("All outputs saved. Download them from the Kaggle notebook output tab.")


def main():
    install_deps()
    clone_repo()
    install_repo_deps()
    setup_env()
    find_cremad()
    run_training()
    save_outputs()
    logger.info("=== ALL DONE ===")


if __name__ == "__main__":
    main()
