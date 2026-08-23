#!/usr/bin/env python3
"""Run CTC Soft Alignment using Wav2Vec2 CTC forced alignment.

Usage:
    python scripts/run_ctc_alignment.py \
        --dataset_root /data/iemocap \
        --output_dir ctc_output
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_alignment(waveform, transcript, model, dictionary, device):
    """Compute word-level alignment using Wav2Vec2 and CTC forced alignment."""
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
        log_probs = torch.log_softmax(emissions, dim=-1)
    
    transcript = transcript.replace('|', ' ').upper()
    tokens = []
    word_boundaries = []
    
    # Simple tokenization
    current_word = []
    for i, char in enumerate(transcript):
        if char in dictionary:
            tokens.append(dictionary[char])
            current_word.append(char)
        elif char == ' ':
            tokens.append(dictionary['|'])
            if current_word:
                word_boundaries.append("".join(current_word))
                current_word = []
    if current_word:
        word_boundaries.append("".join(current_word))
        
    if not tokens:
        return []

    targets = torch.tensor([tokens], dtype=torch.int32).to(device)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.int32).to(device)
    target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32).to(device)
    
    try:
        alignments, scores = torchaudio.functional.forced_align(
            log_probs, targets, input_lengths, target_lengths, blank=0
        )
    except Exception as e:
        logger.warning(f"Forced alignment failed: {e}")
        return []

    alignments = alignments[0].cpu().tolist()
    
    # Convert token alignments back to words
    words = []
    # frame_duration is usually 20ms for Wav2Vec2
    # The stride for base model is 320 on 16000Hz (20ms)
    frame_duration = 0.02
    
    current_word_idx = 0
    current_word_text = word_boundaries[0] if word_boundaries else ""
    start_frame = -1
    
    # We will just collect contiguous non-blank non-space tokens
    space_token = dictionary.get('|')
    
    in_word = False
    for frame_idx, token_id in enumerate(alignments):
        if token_id != 0 and token_id != space_token:
            if not in_word:
                start_frame = frame_idx
                in_word = True
        elif token_id == space_token or token_id == 0:
            if in_word:
                end_frame = frame_idx
                if current_word_idx < len(word_boundaries):
                    words.append({
                        "word": word_boundaries[current_word_idx],
                        "start": round(start_frame * frame_duration, 3),
                        "end": round(end_frame * frame_duration, 3)
                    })
                current_word_idx += 1
                in_word = False

    if in_word and current_word_idx < len(word_boundaries):
        words.append({
            "word": word_boundaries[current_word_idx],
            "start": round(start_frame * frame_duration, 3),
            "end": round(len(alignments) * frame_duration, 3)
        })

    return words

def main():
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 CTC alignment and produce word-level manifest")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="iemocap")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    dictionary = bundle.get_dict()
    
    root = Path(args.dataset_root)
    manifest = {}
    
    logger.info(f"Processing {args.dataset_name} in {root}")

    count = 0
    for sess_dir in sorted(root.glob("Session*")):
        wav_root = sess_dir / "sentences" / "wav"
        trans_dir = sess_dir / "dialog" / "transcriptions"
        if not wav_root.exists() or not trans_dir.exists():
            continue

        transcripts = {}
        for tf in trans_dir.glob("*.txt"):
            with open(tf) as f:
                for line in f:
                    line = line.strip()
                    if " [" in line:
                        utt_id, rest = line.split(" [", 1)
                        text = rest.split("]:")[1].strip() if "]:" in rest else ""
                        transcripts[utt_id.strip()] = text

        for wav_path in wav_root.rglob("*.wav"):
            utt_id = wav_path.stem
            text = transcripts.get(utt_id, "")
            if not text:
                continue
            
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != bundle.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
                
                words = compute_alignment(waveform, text, model, dictionary, device)
                if words:
                    manifest[utt_id] = {
                        "words": words,
                        "audio_path": str(wav_path)
                    }
                    count += 1
            except Exception as e:
                logger.warning(f"Error processing {wav_path}: {e}")

    manifest_path = out_dir / "alignment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated alignments for {count} utterances → {manifest_path}")

if __name__ == "__main__":
    main()
