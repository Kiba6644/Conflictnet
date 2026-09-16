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
from typing import List, Optional

# Add project root to Python path so we can import models/data
sys.path.append(str(Path(__file__).parent.parent))

import torch
from tqdm.auto import tqdm

from data.datasets import load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_for_files(audio_files: list[Path], output_dir: str, batch_size: int = 16, audio_encoder_name: str = "wavlm_weighted", data_root: Optional[Path] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Ensure load_audio does not load precomputed .pt dicts during feature extraction
    old_pt_dir = os.environ.pop("CONFLICTNET_PT_DIR", None)

    try:
        # Filter out macOS hidden files and those that already have a .pt file
        pending_files = []
        for f in audio_files:
            if f.name.startswith("._"):
                continue  # Skip macOS hidden metadata files
            
            if data_root is not None and data_root in f.parents:
                rel = f.relative_to(data_root)
                pt_path = out_path / rel.with_suffix('.pt')
            else:
                pt_path = out_path / f.parent.name / f.with_suffix('.pt').name
                
            if not pt_path.exists():
                pending_files.append(f)
                
        if not pending_files:
            logger.info(f"All {len(audio_files)} files already extracted in {output_dir}. Skipping extraction.")
            return
            
        # 1. Initialize models (frozen)
        logger.info(f"Initializing models to extract {len(pending_files)} missing features...")
        from models.encoders.audio import build_audio_encoder
        from models.speaker_norm.speaker_norm import SpeakerNormalizer
        
        audio_encoder = build_audio_encoder(audio_encoder_name)
        audio_encoder.eval()
        audio_encoder.to(device)
        
        # ECAPA-TDNN outputs 192, we can just use defaults
        speaker_norm = SpeakerNormalizer(embed_dim=256, use_baseline_subtract=True)
        speaker_norm.eval()
        speaker_norm.to(device)
        
        logger.info(f"Extracting features for {len(pending_files)} files (skipped {len(audio_files) - len(pending_files)} already processed or hidden)...")
        
        from concurrent.futures import ThreadPoolExecutor
        import sys

        # 3. Process in batches with multi-threaded audio decoding
        total_batches = (len(pending_files) + batch_size - 1) // batch_size
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
            for b_idx, i in enumerate(range(0, len(pending_files), batch_size)):
                batch_files = pending_files[i:i+batch_size]
                
                # Parallel audio loading across CPU threads
                def _safe_load(p):
                    wave = load_audio(str(p))
                    if isinstance(wave, dict):
                        wave = torch.zeros(16000)
                    return wave

                waveforms = list(pool.map(_safe_load, batch_files))
                    
                # Pad waveforms
                max_len = max(w.shape[-1] for w in waveforms)
                audio_padded = torch.zeros(len(waveforms), max_len, device=device)
                audio_attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.bool, device=device)
                for j, w in enumerate(waveforms):
                    audio_padded[j, :w.shape[-1]] = w.to(device)
                    audio_attention_mask[j, :w.shape[-1]] = True
                    
                # Extract features
                with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
                    # Audio embed (WavLM / Emotion2Vec)
                    audio_embeds, audio_frames = audio_encoder(audio_padded, attention_mask=audio_attention_mask, return_frames=True)
                    # Speaker embed (ECAPA-TDNN)
                    speaker_embeds = speaker_norm.encode_speaker(audio_padded)
                    
                # Save individually with directory hierarchy preserved
                for j, path in enumerate(batch_files):
                    if data_root is not None and data_root in path.parents:
                        rel = path.relative_to(data_root)
                        pt_path = out_path / rel.with_suffix('.pt')
                    else:
                        pt_path = out_path / path.parent.name / path.with_suffix('.pt').name
                    
                    pt_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Trim audio_frames to the valid length for this specific waveform (50 frames/sec for 16kHz audio)
                    saved_frames = None
                    if audio_frames is not None:
                        n_valid_frames = max(1, int(round(waveforms[j].shape[-1] / 320.0)))
                        saved_frames = audio_frames[j, :n_valid_frames].cpu().clone()

                    data = {
                        "audio": audio_embeds[j].cpu().clone(),
                        "speaker": speaker_embeds[j].cpu().clone(),
                        "audio_frames": saved_frames,
                    }
                    torch.save(data, pt_path)

                if (b_idx + 1) % 25 == 0 or (b_idx + 1) == total_batches:
                    processed_count = min(i + batch_size, len(pending_files))
                    logger.info(f"Feature extraction progress: {processed_count}/{len(pending_files)} files ({100*processed_count/len(pending_files):.1f}%)")
                    sys.stdout.flush()
                    
        logger.info("Extraction complete!")
        sys.stdout.flush()
    finally:
        if old_pt_dir is not None:
            os.environ["CONFLICTNET_PT_DIR"] = old_pt_dir

def extract_features(data_root: str, output_dir: str, batch_size: int = 16, audio_encoder_name: str = "wavlm_weighted"):
    root = Path(data_root)
    audio_files = []
    for ext in ["*.mp4", "*.wav"]:
        audio_files.extend(list(root.rglob(ext)))
        
    logger.info(f"Found {len(audio_files)} audio files in {data_root}")
    extract_features_for_files(audio_files, output_dir, batch_size, audio_encoder_name=audio_encoder_name, data_root=root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--meld_root", type=str, default=None, help="Alias for --data_root")
    parser.add_argument("--output_dir", type=str, default="/workspace/features", help="Output directory for pt files")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--audio_encoder", type=str, default="wavlm_weighted", help="Audio encoder to use (wavlm_weighted, emotion2vec, wavlm, etc.)")
    parser.add_argument("--audio_encoder_path", type=str, default=None, help="Local path for audio encoder")
    args = parser.parse_args()
    
    root_path = args.data_root or args.meld_root
    if not root_path:
        parser.error("Must provide either --data_root or --meld_root")

    # Clear CONFLICTNET_PT_DIR so load_audio extracts from raw audio
    os.environ["CONFLICTNET_PT_DIR"] = ""
    if args.audio_encoder_path:
        os.environ["CONFLICTNET_EMOTION2VEC_PATH"] = args.audio_encoder_path
        os.environ["CONFLICTNET_WAVLM_PATH"] = args.audio_encoder_path
    
    extract_features(root_path, args.output_dir, args.batch_size, args.audio_encoder)
