import argparse
import os
import logging
from pathlib import Path

import torchaudio
import torchaudio.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_dataset(src_dir: str, dst_dir: str):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    if not src_path.exists():
        logger.error(f"Source directory {src_dir} does not exist!")
        return

    splits = ["train", "val", "test"]
    converted = 0
    
    for split in splits:
        s_src = src_path / split
        s_dst = dst_path / split
        
        if not s_src.exists():
            continue
            
        s_dst.mkdir(parents=True, exist_ok=True)
        logger.info(f"Scanning {s_src} for .mp4 files...")
        
        for mp4 in s_src.glob("*.mp4"):
            target_wav = s_dst / mp4.with_suffix(".wav").name
            if not target_wav.exists():
                try:
                    wf, sr = torchaudio.load(str(mp4))
                    if sr != 16000:
                        wf = F.resample(wf, sr, 16000)
                    # Convert stereo to mono
                    if wf.shape[0] > 1:
                        wf = wf.mean(dim=0, keepdim=True)
                    torchaudio.save(str(target_wav), wf, 16000)
                    converted += 1
                except Exception as e:
                    logger.warning(f"Failed to convert {mp4}: {e}")
            
    logger.info(f"Successfully converted {converted} new audio files to {dst_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to raw MELD directory containing train/val folders")
    parser.add_argument("--dst", type=str, required=True, help="Path to output WAV directory")
    args = parser.parse_args()
    convert_dataset(args.src, args.dst)
