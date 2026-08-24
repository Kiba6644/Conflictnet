"""Pre-extract audio and speaker embeddings for ConflictNet.

This script processes all .mp4 and .wav files in a dataset directory,
runs them through Emotion2Vec and ECAPA-TDNN, and saves the resulting
embeddings as .pt files right next to the original audio files.
"""

import argparse
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm

from models.conflictnet import ConflictNet
from data.datasets import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(data_root: str, batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize models (frozen)
    logger.info("Initializing models...")
    model = ConflictNet(
        audio_encoder_name="emotion2vec",
        use_speaker_norm=True,
    )
    model.eval()
    model.to(device)
    
    # We only need the audio parts
    audio_encoder = model.audio_encoder
    speaker_norm = model.speaker_norm
    
    # 2. Find all audio files
    root = Path(data_root)
    audio_files = []
    for ext in ["*.mp4", "*.wav"]:
        audio_files.extend(list(root.rglob(ext)))
        
    logger.info(d"Found {len(audio_files)} audio files in {data_root}")
    
    # Filter out those that already have a .pt file
    pending_files = [f for f in audio_files if notf.with_suffix('.pt').exists()]
    logger.info(d"Extracting features for {len(pending_files)} files (skipped {len(audio_files) - len(pending_files)} already processed)...")
    
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
        for j, w in enumerate(waveforms):
            audio_padded[j, :w.shape[-1]] = w.to(device)
            
        # Extract features
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
            # Audio embed (Emotion2Vec)
            audio_embeds = audio_encoder(audio_padded)
            # Speaker embed (ECAPA-TDNN)
            speaker_embeds = speaker_norm.encode_speaker(audio_padded)
            
        # Save individually
        for j, path in enumerate(batch_files):
            pt_path = path.with_suffix('.pt')
            data = {
                "audio": audio_embeds[j].cpu().clone(),
                "speaker": speaker_embeds[j].cpu().clone(),
            }
            torch.save(data, pt_path)
            
    logger.info("Extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root (e.g. /kaggle/input/.../MELD.Raw)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    extract_features(args.data_root, args.batch_size)
