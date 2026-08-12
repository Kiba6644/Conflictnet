#!/usr/bin/env python3
"""ConflictNet Training Script for Kaggle GPU.

Usage: python train_conflictnet.py
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
CREMAD_DIR = WORK_DIR / "cremad"
MUSTARD_DIR = WORK_DIR / "mustard"
OUTPUT_DIR = WORK_DIR / "output"


def install_deps():
    logger.info("Installing dependencies...")
    deps = ["sentencepiece", "tiktoken", "kagglehub"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + deps,
        check=False, capture_output=False,
    )
    logger.info("Dependencies installed.")


def find_code_dir():
    for p in KAGGLE_INPUT.rglob("scripts/train.py"):
        return p.parent.parent
    for p in KAGGLE_INPUT.rglob("prosody_stats.json"):
        return p.parent
    logger.error("Cannot find code directory in Kaggle inputs!")
    sys.exit(1)


def find_inputs():
    """Log the Kaggle input directory structure for debugging."""
    logger.info("=== Kaggle input structure ===")
    for p in sorted(KAGGLE_INPUT.rglob("*")):
        if p.is_file():
            logger.info(f"  FILE: {p.relative_to(KAGGLE_INPUT)} ({p.stat().st_size} bytes)")
        elif p.is_dir():
            logger.info(f"  DIR:  {p.relative_to(KAGGLE_INPUT)}")
    logger.info("=== End input structure ===")


def setup_models():
    """Find pretrained model directories and set up env vars."""
    hf_cache = WORK_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)

    # Find model directories by looking for config.json with known names
    import tarfile
    models_dir = WORK_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("wavlm", "wavlm-large/model.safetensors", "CONFLICTNET_WAVLM_PATH"),
        ("deberta", "deberta-v3-large/model.safetensors", "CONFLICTNET_DEBERTA_PATH"),
    ]

    for model_key, marker, env_var in configs:
        found_path = None

        # Strategy 1: Direct path to extracted model dir
        for p in KAGGLE_INPUT.rglob(marker):
            found_path = str(p.parent)
            logger.info(f"Found {model_key} at {found_path}")
            break

        # Strategy 2: Look for tar file and extract it
        if found_path is None:
            for ext in [".tar.gz", ".tar"]:
                for p in KAGGLE_INPUT.rglob(f"{model_key}*{ext}"):
                    logger.info(f"Found {model_key} tarball at {p}, extracting...")
                    extract_dir = models_dir / model_key
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        mode = "r:gz" if ext == ".tar.gz" else "r:"
                        with tarfile.open(str(p), mode) as tar:
                            tar.extractall(path=str(extract_dir))
                        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                        if subdirs:
                            found_path = str(subdirs[0])
                        else:
                            found_path = str(extract_dir)
                    except Exception as e:
                        logger.warning(f"Failed to extract {p}: {e}")
                    break
                if found_path:
                    break

        if found_path is None:
            logger.warning(f"Model {model_key} not found in Kaggle inputs!")
        else:
            os.environ[env_var] = found_path
            logger.info(f"Set {env_var}={found_path}")

    # Tokenizer comes from the DeBERTa model dir
    tok_dir = os.environ.get("CONFLICTNET_DEBERTA_PATH")
    if tok_dir:
        import transformers
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(tok_dir)
            logger.info(f"Tokenizer: {type(tokenizer).__name__} (vocab_size={tokenizer.vocab_size})")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tok_dir}: {e}")
            tok_dir = None
    return tok_dir


def setup_data():
    logger.info("Setting up data directories...")
    find_inputs()
    CREMAD_DIR.mkdir(parents=True, exist_ok=True)
    MUSTARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    code_dir = find_code_dir()
    logger.info(f"Code directory: {code_dir}")

    # CREMA-D from Kaggle input
    cremad_input = list(KAGGLE_INPUT.rglob("ejlok1/cremad/AudioWAV"))
    if cremad_input:
        logger.info(f"Found CREMA-D at {cremad_input[0]}")
        (CREMAD_DIR / "AudioWAV").symlink_to(cremad_input[0])
    else:
        for p in KAGGLE_INPUT.rglob("AudioWAV"):
            if p.is_dir():
                (CREMAD_DIR / "AudioWAV").symlink_to(p)
                logger.info(f"Found CREMA-D at {p}")
                break
        else:
            logger.error("CREMA-D AudioWAV not found in Kaggle inputs!")
            sys.exit(1)

    # MUStARD++ data from code dataset
    mustard_json_src = code_dir / "data" / "mustard" / "mustard++_raw_data.json"
    if mustard_json_src.exists():
        (MUSTARD_DIR / "mustard++_raw_data.json").symlink_to(mustard_json_src.resolve())

    # MUStARD++ audio unavailable — CREMA-D only
    logger.info("MUStARD++ audio unavailable — training on CREMA-D only")

    # Copy prosody stats
    for fname in ["prosody_stats.json", "prosody_stats.zscores.json"]:
        src = code_dir / fname
        if src.exists():
            (WORK_DIR / fname).symlink_to(src.resolve())

    tok_dir = setup_models()
    return code_dir, tok_dir


def run_training(code_dir, tok_dir):
    logger.info("Starting training...")
    train_script = code_dir / "scripts" / "train.py"
    if not train_script.exists():
        logger.error(f"Training script not found at {train_script}")
        sys.exit(1)

    import torch
    n_gpus = torch.cuda.device_count()
    logger.info(f"Detected {n_gpus} GPU(s)")

    # Base training args
    train_args = [
        str(train_script),
        "--cremad_root", str(CREMAD_DIR),
        "--epochs", "30",
        "--pretrain_epochs", "5",
        "--batch_size", "4",
        "--lr", "5e-5",
        # wavlm_weighted: learnable softmax over all 25 WavLM-large hidden states
        # (~2-4% F1 improvement over last-hidden-state only)
        "--audio_encoder", "wavlm_weighted",
        "--no_word_divergence",
        "--gradient_accumulation_steps", "2",
        "--amp",
        "--label_smoothing", "0.05",
        "--early_stop_patience", "5",
        "--prosody_stats", str(WORK_DIR / "prosody_stats.json"),
        "--output_dir", str(OUTPUT_DIR),
    ]
    if tok_dir:
        train_args.extend(["--tokenizer_path", str(tok_dir)])

    mustard_json = MUSTARD_DIR / "mustard++_raw_data.json"
    if mustard_json.exists():
        mustard_audio = MUSTARD_DIR / "utterances_final"
        if mustard_audio.exists() and list(mustard_audio.glob("*")):
            train_args.extend(["--mustard_root", str(MUSTARD_DIR)])
            logger.info("MUStARD++ root found, including in training")

    # Use torchrun for multi-GPU DDP (2x T4 on Kaggle)
    if n_gpus >= 2:
        cmd = ["torchrun", f"--nproc_per_node={n_gpus}"] + train_args
        logger.info(f"Running with torchrun ({n_gpus} GPUs): {' '.join(cmd)}")
    else:
        cmd = [sys.executable] + train_args
        logger.info(f"Running single-GPU: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{code_dir}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Stream output line by line for real-time Kaggle logs
    import subprocess
    process = subprocess.Popen(
        cmd, cwd=str(code_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        logger.error(f"Training failed with code {return_code}")
        sys.exit(return_code)
    logger.info("Training completed!")


def save_outputs():
    logger.info("Saving outputs...")
    patterns = ["*.safetensors", "*.pt", "*.pth", "*_meta.json"]
    for pattern in patterns:
        for p in OUTPUT_DIR.glob(pattern):
            final_path = WORK_DIR / p.name
            p.rename(final_path)
            logger.info(f"Saved: {final_path}")

    logs = list(OUTPUT_DIR.glob("*.log"))
    for log in logs:
        final_path = WORK_DIR / log.name
        log.rename(final_path)
        logger.info(f"Log saved: {final_path}")


def main():
    install_deps()
    code_dir, tok_dir = setup_data()
    run_training(code_dir, tok_dir)
    save_outputs()
    logger.info("All done!")


if __name__ == "__main__":
    main()
