"""Pre-extract audio and speaker embeddings for ConflictNet.

This script processes all .mp4 and .wav files in a dataset directory,
runs them through Emotion2Vec and ECAPA-TDNN, and saves the resulting
embeddings as .pt files right next to the original audio files.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path so we can import models/data
sys.path.append(str(Path(__file__).parent.parent))

import torch
from tqdm.auto import tqdm

from data.datasets import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_for_files(audio_files: list[Path], output_dir: str, batch_size: int = 16, audio_encoder_name: str = "emotion2vec"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Filter out macOS hidden files and those that already have a .pt file
    pending_files = []
    for f in audio_files:
        if f.name.startswith("._"):
            continue  # Skip macOS hidden metadata files
            
        pt_path = out_path / f.with_suffix('.pt').name
        if not pt_path.exists():
            pending_files.append(f)
            
    if not pending_files:
        logger.info(f"All {len(audio_files)} files already extracted in {output_dir}. Skipping extraction.")
        return
        
    # 1. Initialize models (frozen)
    logger.info(f"Initializing models to extract {len(pending_files)} missing features...")
    from models.encoders.audio import build_audio_encoder
    from models.speaker_norm.speaker_norm import SpeakerNormalizer
    
    # We only instantiate what we need to avoid downloading text models
    audio_encoder = build_audio_encoder(audio_encoder_name)
    audio_encoder.eval()
    audio_encoder.to(device)
    
    # ECAPA-TDNN outputs 192, we can just use defaults
    speaker_norm = SpeakerNormalizer(embed_dim=256, use_baseline_subtract=True)
    speaker_norm.eval()
    speaker_norm.to(device)
    
    logger.info(f"Extracting features for {len(pending_files)} files (skipped {len(audio_files) - len(pending_files)} already processed or hidden)...")
    
    # 3. Process in batches
    for i in tqdm(range(0, len(pending_files), batch_size), desc="Extracting"):
        batch_files = pending_files[i:i+batch_size]
        
        # Load audio waveforms
        waveforms = []
        for path in batch_files:
            wave = load_audio(str(path))
            if isinstance(wave, dict):
                # somehow got a .pt, should have been filtered
                wave = torch.zeros(16000)
            waveforms.append(wave)
            
        # Pad waveforms
        max_len = max(w.shape[-1] for w in waveforms)
        audio_padded = torch.zeros(len(waveforms), max_len, device=device)
        audio_attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.bool, device=device)
        for j, w in enumerate(waveforms):
            audio_padded[j, :w.shape[-1]] = w.to(device)
            audio_attention_mask[j, :w.shape[-1]] = True
            
        # Extract features
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
            # Audio embed (Emotion2Vec)
            audio_embeds, audio_frames = audio_encoder(audio_padded, attention_mask=audio_attention_mask, return_frames=True)
            # Speaker embed (ECAPA-TDNN)
            speaker_embeds = speaker_norm.encode_speaker(audio_padded)
            
        # Save individually
        for j, path in enumerate(batch_files):
            pt_path = out_path / path.with_suffix('.pt').name
            data = {
                "audio": audio_embeds[j].cpu().clone(),
                "speaker": speaker_embeds[j].cpu().clone(),
                "audio_frames": audio_frames[j].cpu().clone() if audio_frames is not None else None,
            }
            torch.save(data, pt_path)
            
    logger.info("Extraction complete!")

def extract_features(data_root: str, output_dir: str, batch_size: int = 16, audio_encoder_name: str = "emotion2vec"):
    root = Path(data_root)
    audio_files = []
    for ext in ["*.mp4", "*.wav"]:
        audio_files.extend(list(root.rglob(ext)))
        
    logger.info(f"Found {len(audio_files)} audio files in {data_root}")
    extract_features_for_files(audio_files, output_dir, batch_size, audio_encoder_name=audio_encoder_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root (e.g. /kaggle/input/.../MELD.Raw)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/features", help="Output directory for pt files")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--audio_encoder", type=str, default="emotion2vec", help="Audio encoder to use")
    parser.add_argument("--audio_encoder_path", type=str, default=None, help="Local path for Emotion2Vec")
    args = parser.parse_args()
    
    if args.audio_encoder_path:
        os.environ["CONFLICTNET_EMOTION2VEC_PATH"] = args.audio_encoder_path
    
    extract_features(args.data_root, args.output_dir, args.batch_size, args.audio_encoder)
