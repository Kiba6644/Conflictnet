import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from tqdm.auto import tqdm

# Add project root to python path so we can import datasets.py
sys.path.append(str(Path(__file__).parent.parent))
from data.datasets import MELDDataset, load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_dataset(src_dir: str, dst_dir: str, max_train: int, max_val: int):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        logger.error(f"Source directory {src_dir} does not exist!")
        return

    logger.info(f"Initializing MELDDataset to fetch exact deterministic subset (Train: {max_train}, Val: {max_val})...")
    
    # We load the dataset using the exact same deterministic subset logic used in training!
    train_ds = MELDDataset(str(src_path), split="train", max_samples=max_train)
    val_ds = MELDDataset(str(src_path), split="val", max_samples=max_val)
    
    all_items = train_ds.items + val_ds.items
    
    # Extract unique file paths
    files_to_convert = []
    for item in all_items:
        if "wav_path" in item and item["wav_path"] is not None:
            files_to_convert.append(Path(item["wav_path"]))
            
    # Deduplicate in case of oversampling
    files_to_convert = list(set(files_to_convert))
    logger.info(f"Subset selected! Found {len(files_to_convert)} unique files to convert.")
    
    converted = 0
    os.environ["CONFLICTNET_PT_DIR"] = ""  # Ensure load_audio returns raw waveforms, not cached dicts
    
    import subprocess
    
    # Progress bar!
    for mp4 in tqdm(files_to_convert, desc="Converting MP4 to WAV"):
        rel_path = mp4.relative_to(src_path)
        target_wav = dst_path / rel_path.with_suffix(".wav")
        
        if not target_wav.exists():
            target_wav.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Kaggle's torchaudio native MP4 decoder segfaults or silently returns empty tensors!
                # We use ffmpeg directly in a subprocess to flawlessly rip the audio.
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp4), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(target_wav)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                converted += 1
            except Exception as e:
                logger.warning(f"\nFailed to convert {mp4}: {e}")
            
    logger.info(f"Successfully converted {converted} new audio files to {dst_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to raw MELD directory")
    parser.add_argument("--dst", type=str, required=True, help="Path to output WAV directory")
    parser.add_argument("--train_samples", type=int, default=1500, help="Number of train samples")
    parser.add_argument("--val_samples", type=int, default=200, help="Number of val samples")
    args = parser.parse_args()
    convert_dataset(args.src, args.dst, args.train_samples, args.val_samples)
