#!/usr/bin/env python3
"""High-speed multi-threaded MELD MP4 to 16kHz WAV converter.

Extracts 16 kHz mono WAV audio from MELD MP4 video files using multi-threaded ffmpeg.
Optionally generates matching .txt transcript files required by Montreal Forced Aligner (MFA).

Supports converting:
  - Full dataset (all ~13,700 utterances across train, dev, and test)
  - Specific subset / sample caps
  - Resumable (skips already converted, non-empty files)

Usage:
    # Full dataset conversion with transcripts for MFA:
    python scripts/convert_to_wav.py --src auto --dst /kaggle/working/wav_dataset --all --with_transcripts

    # Subset conversion (e.g. for quick local testing):
    python scripts/convert_to_wav.py --src /data/meld --dst /data/meld_wav --train_samples 1500 --val_samples 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# Add project root to sys.path so 'data' can be imported if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KAGGLE_CANDIDATE_ROOTS = [
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


def find_source_root(requested_src: str) -> Path:
    """Find MELD root directory from explicit path or common Kaggle inputs."""
    if requested_src and requested_src.lower() != "auto":
        p = Path(requested_src)
        if p.exists():
            return p
        logger.warning(f"Specified source path '{requested_src}' not found. Searching Kaggle inputs...")

    for cand in KAGGLE_CANDIDATE_ROOTS:
        if cand.exists() and any(cand.glob("*sent_emo.csv")):
            return cand

    # Deep search under /kaggle/input if mounted under custom dataset title
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for csv_file in kaggle_input.rglob("*train_sent_emo.csv"):
            return csv_file.parent

    raise FileNotFoundError(
        "Could not find MELD dataset! Please specify --src with the valid path to MELD.Raw."
    )


def collect_split_tasks(
    src_root: Path,
    dst_root: Path,
    split_name: str,
    max_samples: Optional[int],
    with_transcripts: bool,
) -> Tuple[List[Tuple[Path, Path, Optional[Path], str]], List[Path]]:
    """Scan CSV and filesystem to build list of files to convert for a split."""
    csv_candidates = [
        src_root / split_name / f"{split_name}_sent_emo.csv",
        src_root / f"{split_name}_sent_emo.csv",
    ]
    csv_path = next((c for c in csv_candidates if c.exists()), None)
    if not csv_path:
        logger.warning(f"No CSV found for split '{split_name}' in {src_root}")
        return [], []

    # Find split folder with MP4/WAV files
    folder_candidates = [
        src_root / split_name / f"{split_name}_splits",
        src_root / split_name / "dev_splits_complete",
        src_root / split_name / "output_repeated_splits_test",
        src_root / split_name / "output_repeated_splits",
        src_root / split_name,
        src_root,
    ]
    split_folder = next((f for f in folder_candidates if f.exists() and f.is_dir()), None)
    if not split_folder:
        logger.warning(f"No video/audio folder found for split '{split_name}' in {src_root}")
        return [], [csv_path]

    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dia_id = str(row.get("Dialogue_ID", row.get("dialogue_id", ""))).strip()
            utt_id = str(row.get("Utterance_ID", row.get("utterance_id", ""))).strip()
            text = str(row.get("Utterance", row.get("utterance", ""))).strip()

            src_mp4 = split_folder / f"dia{dia_id}_utt{utt_id}.mp4"
            src_wav = split_folder / f"dia{dia_id}_utt{utt_id}.wav"

            # Prefer existing wav if present, else mp4
            src_file = src_wav if src_wav.exists() else (src_mp4 if src_mp4.exists() else None)
            if not src_file:
                continue

            # Output folder mirrors split: dst_root/train/diaX_uttY.wav
            rel_folder = Path(split_name)
            target_wav = dst_root / rel_folder / f"dia{dia_id}_utt{utt_id}.wav"
            target_txt = dst_root / rel_folder / f"dia{dia_id}_utt{utt_id}.txt" if with_transcripts else None

            tasks.append((src_file, target_wav, target_txt, text))
            if max_samples and len(tasks) >= max_samples:
                break

    return tasks, [csv_path]


def convert_single_file(item: Tuple[Path, Path, Optional[Path], str]) -> bool:
    """Worker task: convert single file to 16kHz mono WAV and optionally write transcript."""
    src_file, target_wav, target_txt, text = item

    target_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write transcript for MFA if enabled
    if target_txt is not None:
        try:
            if not target_txt.exists() or target_txt.stat().st_size == 0:
                with open(target_txt, "w", encoding="utf-8") as tf:
                    tf.write(text.strip())
        except Exception:
            pass

    # 2. Check if already converted and valid (>100 bytes)
    if target_wav.exists() and target_wav.stat().st_size > 100:
        return True

    # 3. If source is already a WAV file, copy or resample
    if src_file.suffix.lower() == ".wav":
        try:
            cmd = [
                "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                "-i", str(src_file),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(target_wav),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return res.returncode == 0
        except Exception:
            try:
                shutil.copy2(str(src_file), str(target_wav))
                return True
            except Exception:
                return False

    # 4. Convert MP4 -> 16kHz Mono WAV via ffmpeg
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(src_file),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(target_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return res.returncode == 0


def main():
    p = argparse.ArgumentParser(description="Multi-threaded MELD MP4 to 16kHz WAV converter")
    p.add_argument("--src", type=str, default="auto", help="Path to raw MELD dataset root (or 'auto')")
    p.add_argument("--dst", type=str, required=True, help="Destination directory for WAV dataset")
    p.add_argument("--all", action="store_true", help="Convert all splits (train, dev, test) without caps")
    p.add_argument("--train_samples", type=int, default=None, help="Cap train samples (None = all)")
    p.add_argument("--val_samples", type=int, default=None, help="Cap val samples (None = all)")
    p.add_argument("--test_samples", type=int, default=None, help="Cap test samples (None = all)")
    p.add_argument("--with_transcripts", action="store_true", default=True,
                   help="Generate .txt transcript files next to .wav files for MFA (default: True)")
    p.add_argument("--no_transcripts", action="store_false", dest="with_transcripts",
                   help="Disable transcript generation")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel ffmpeg worker threads (default: CPU cores * 2)")

    args = p.parse_args()

    src_root = find_source_root(args.src)
    dst_root = Path(args.dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source root : {src_root}")
    logger.info(f"Target root : {dst_root}")

    train_cap = None if args.all else args.train_samples
    val_cap = None if args.all else args.val_samples
    test_cap = None if args.all else args.test_samples

    # Determine splits to process
    splits = [
        ("train", train_cap),
        ("dev", val_cap),
        ("test", test_cap),
    ]

    all_tasks = []
    csv_to_copy = []

    for split_name, cap in splits:
        tasks, csvs = collect_split_tasks(
            src_root=src_root,
            dst_root=dst_root,
            split_name=split_name,
            max_samples=cap,
            with_transcripts=args.with_transcripts,
        )
        all_tasks.extend(tasks)
        csv_to_copy.extend(csvs)
        logger.info(f"Split '{split_name}': {len(tasks)} utterances queued (cap: {cap or 'ALL'})")

    if not all_tasks:
        logger.error("No valid audio/video files found to convert!")
        sys.exit(1)

    # Copy CSV metadata files into root and split subdirectories for maximum compatibility
    for csv_file in set(csv_to_copy):
        dst_csv = dst_root / csv_file.name
        shutil.copy2(str(csv_file), str(dst_csv))
        for split_dir in ["train", "dev", "test"]:
            sub_csv = dst_root / split_dir / csv_file.name
            sub_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(csv_file), str(sub_csv))

    logger.info(f"Copied metadata CSV files to {dst_root}")

    num_workers = args.workers or min(16, max(4, (os.cpu_count() or 4) * 2))
    logger.info(f"Starting conversion of {len(all_tasks):,} files using {num_workers} parallel workers...")

    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_single_file, task) for task in all_tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting to 16kHz WAV"):
            if fut.result():
                successful += 1

    logger.info(
        f"Conversion complete! Successfully processed {successful}/{len(all_tasks)} files into {dst_root}"
    )


if __name__ == "__main__":
    main()
