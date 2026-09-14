#!/usr/bin/env python3
"""Run Montreal Forced Aligner (MFA) on MELD dataset to generate .TextGrid files.

Prerequisites on Kaggle:
    conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst kaldi
    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa

Usage:
    python scripts/run_mfa_meld.py \
        --corpus_dir /kaggle/working/wav_dataset \
        --output_dir /kaggle/working/meld_textgrids \
        --jobs 4
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_mfa() -> bool:
    """Return True if mfa command is available on PATH."""
    return shutil.which("mfa") is not None


def ensure_models(dictionary: str, acoustic_model: str):
    """Download pretrained models if not already downloaded."""
    for model_type, name in [("dictionary", dictionary), ("acoustic", acoustic_model)]:
        logger.info(f"[MFA] Ensuring {model_type} model '{name}' is downloaded...")
        cmd = ["mfa", "model", "download", model_type, name]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 and "already exists" not in res.stderr.lower():
            logger.warning(f"[MFA] Download message for {name}: {res.stderr.strip() or res.stdout.strip()}")


def run_alignment(
    corpus_dir: Path,
    output_dir: Path,
    dictionary: str,
    acoustic_model: str,
    jobs: int,
    clean: bool,
) -> int:
    """Run Montreal Forced Aligner."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total wav files in corpus
    total_wavs = len(list(corpus_dir.rglob("*.wav")))
    total_txts = len(list(corpus_dir.rglob("*.txt")))
    logger.info(f"[MFA] Found {total_wavs} .wav files and {total_txts} .txt transcripts in {corpus_dir}")

    if total_wavs == 0:
        logger.error(f"[MFA] No .wav files found in {corpus_dir}!")
        return 1

    if total_txts == 0:
        logger.error(f"[MFA] No .txt transcript files found in {corpus_dir}! Run convert_to_wav.py with --with_transcripts first.")
        return 1

    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--jobs", str(jobs),
        "--single_speaker",
        "--output_format", "long_textgrid",
    ]
    if clean:
        cmd.append("--clean")

    logger.info(f"[MFA] Launching alignment command:\n{' '.join(cmd)}")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

    process.stdout.close()
    return_code = process.wait()

    # Verify generated TextGrids
    generated_tgs = len(list(output_dir.rglob("*.TextGrid")))
    logger.info(f"[MFA] Alignment finished (exit code {return_code}).")
    logger.info(f"[MFA] Generated {generated_tgs}/{total_wavs} .TextGrid files in {output_dir}")

    return return_code


def main():
    p = argparse.ArgumentParser(description="Run MFA alignment on MELD dataset")
    p.add_argument("--corpus_dir", type=str, default="/kaggle/working/wav_dataset",
                   help="Path to WAV dataset containing .wav and .txt files")
    p.add_argument("--output_dir", type=str, default="/kaggle/working/meld_textgrids",
                   help="Destination directory for output .TextGrid files")
    p.add_argument("--dictionary", type=str, default="english_us_arpa",
                   help="MFA dictionary (default: english_us_arpa)")
    p.add_argument("--acoustic_model", type=str, default="english_us_arpa",
                   help="MFA acoustic model (default: english_us_arpa)")
    p.add_argument("--jobs", type=int, default=None,
                   help="Number of parallel alignment jobs (default: CPU count)")
    p.add_argument("--no_clean", action="store_false", dest="clean",
                   help="Do not clean MFA cache before running")

    args = p.parse_args()

    if not check_mfa():
        logger.error(
            "Montreal Forced Aligner (mfa) is not installed or not on PATH!\n"
            "On Kaggle, please install it using:\n"
            "  conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst kaldi\n"
        )
        sys.exit(1)

    jobs = args.jobs or max(1, os.cpu_count() or 4)
    logger.info(f"[MFA] Running with {jobs} parallel worker jobs")

    ensure_models(args.dictionary, args.acoustic_model)

    corpus_path = Path(args.corpus_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    rc = run_alignment(
        corpus_dir=corpus_path,
        output_dir=output_path,
        dictionary=args.dictionary,
        acoustic_model=args.acoustic_model,
        jobs=jobs,
        clean=args.clean,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
