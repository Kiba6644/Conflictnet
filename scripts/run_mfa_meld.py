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

# Suppress pydevd frozen modules warnings and force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[FlushHandler(sys.stdout)],
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
    fast_mode: bool = True,
    clean_transcripts_after: bool = False,
) -> int:
    """Run Montreal Forced Aligner with speed optimizations and live progress monitoring."""
    import threading
    import time

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total wav and txt files in corpus
    all_wavs = list(corpus_dir.rglob("*.wav"))
    total_wavs = len(all_wavs)
    total_txts = len(list(corpus_dir.rglob("*.txt")))
    logger.info(f"[MFA] Found {total_wavs:,} .wav files and {total_txts:,} .txt transcripts in {corpus_dir}")

    if total_wavs == 0:
        logger.error(f"[MFA] No .wav files found in {corpus_dir}!")
        return 1

    if total_txts == 0:
        logger.error(f"[MFA] No .txt transcript files found in {corpus_dir}! Run convert_to_wav.py with --with_transcripts first.")
        return 1

    temp_dir = Path("/tmp/mfa_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--jobs", str(jobs),
        "--single_speaker",
        "--output_format", "long_textgrid",
        "--temp_directory", str(temp_dir),
    ]

    if fast_mode:
        # Tighter beam width cuts Kaldi lattice search time by ~2.5x
        cmd.extend(["--beam", "10", "--retry_beam", "40"])
        # Check if --no_speaker_adaptation is supported in this MFA version
        try:
            help_res = subprocess.run(["mfa", "align", "--help"], capture_output=True, text=True, check=False)
            if "--no_speaker_adaptation" in help_res.stdout or "--no_speaker_adaptation" in help_res.stderr:
                cmd.append("--no_speaker_adaptation")
            elif "--fast" in help_res.stdout or "--fast" in help_res.stderr:
                cmd.append("--fast")
        except Exception:
            pass

    if clean:
        cmd.append("--clean")

    logger.info(f"[MFA] Launching alignment command with speed optimizations:\n{' '.join(cmd)}")
    sys.stdout.flush()

    start_time = time.time()
    stop_monitor = threading.Event()

    # Background progress thread to monitor generated TextGrids
    def progress_monitor():
        while not stop_monitor.wait(15.0):
            current_tgs = len(list(output_dir.rglob("*.TextGrid")))
            if current_tgs > 0:
                elapsed = time.time() - start_time
                rate = current_tgs / max(1.0, elapsed)
                pct = 100.0 * current_tgs / max(1, total_wavs)
                remaining = (total_wavs - current_tgs) / max(0.1, rate)
                logger.info(
                    f"[MFA Live Progress] {current_tgs:,}/{total_wavs:,} TextGrids ({pct:.1f}%) | "
                    f"Speed: {rate:.1f} clips/s | Elapsed: {int(elapsed//60)}m {int(elapsed%60):02d}s | "
                    f"ETA: {int(remaining//60)}m {int(remaining%60):02d}s"
                )
                sys.stdout.flush()

    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        # Stream Kaldi log lines live
        print(line, end="", flush=True)

    process.stdout.close()
    return_code = process.wait()
    stop_monitor.set()
    monitor_thread.join(timeout=2.0)

    total_time = time.time() - start_time
    generated_tgs = len(list(output_dir.rglob("*.TextGrid")))

    logger.info(
        f"✅ MFA Alignment finished in {int(total_time//60)}m {int(total_time%60):02d}s (exit code {return_code})."
    )
    logger.info(
        f"📊 Successfully generated {generated_tgs:,}/{total_wavs:,} .TextGrid files ({100*generated_tgs/max(1, total_wavs):.1f}%) in {output_dir}"
    )

    # Optional cleanup: remove .txt files from corpus directory so wav_dataset stays clean
    if clean_transcripts_after:
        logger.info(f"🧹 Cleaning up intermediate .txt transcripts from {corpus_dir}...")
        removed = 0
        for txt in corpus_dir.rglob("*.txt"):
            try:
                txt.unlink()
                removed += 1
            except Exception:
                pass
        logger.info(f"Removed {removed:,} temporary .txt files. {corpus_dir} now contains only clean WAVs & CSVs.")

    return return_code


def main():
    p = argparse.ArgumentParser(description="Run fast MFA alignment on MELD dataset")
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
    p.add_argument("--no_fast", action="store_false", dest="fast_mode",
                   help="Disable speed optimizations (tight beam, skip speaker adaptation)")
    p.add_argument("--clean_transcripts", action="store_true", default=True,
                   help="Remove .txt files from corpus_dir after alignment finishes (default: True)")
    p.add_argument("--keep_transcripts", action="store_false", dest="clean_transcripts",
                   help="Keep .txt files in corpus_dir")

    args = p.parse_args()

    if not check_mfa():
        logger.error(
            "Montreal Forced Aligner (mfa) is not installed or not on PATH!\n"
            "On Kaggle, please install it using:\n"
            "  conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst kaldi\n"
        )
        sys.exit(1)

    jobs = args.jobs or max(1, os.cpu_count() or 4)
    logger.info(f"[MFA] Running with {jobs} parallel worker jobs (Fast Mode: {args.fast_mode})")

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
        fast_mode=args.fast_mode,
        clean_transcripts_after=args.clean_transcripts,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

