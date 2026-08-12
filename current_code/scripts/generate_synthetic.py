"""CLI: generate synthetic conflict data via StarGANv2-VC.

Usage:
    python scripts/generate_synthetic.py \
        --neutral_dir /data/iemocap/neutral_wavs \
        --transcript_json /data/iemocap/neutral_transcripts.json \
        --output_dir /data/synthetic_conflict \
        --target_emotion angry \
        --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic conflict data")
    p.add_argument("--neutral_dir", type=str, required=True,
                   help="Directory of neutral .wav files")
    p.add_argument("--transcript_json", type=str, required=True,
                   help="JSON mapping filename → transcript text")
    p.add_argument("--output_dir", type=str, default="synthetic_conflict")
    p.add_argument("--target_emotion", type=str, default="angry",
                   choices=["angry", "sad", "happy", "fearful"])
    p.add_argument("--limit", type=int, default=-1,
                   help="Max number of pairs to generate (-1 = all)")
    return p.parse_args()


def main():
    args = parse_args()

    neutral_dir = Path(args.neutral_dir)
    wav_paths = sorted(neutral_dir.glob("*.wav"))
    if args.limit > 0:
        wav_paths = wav_paths[:args.limit]

    with open(args.transcript_json) as f:
        transcripts = json.load(f)

    audio_paths = []
    texts = []
    for wav in wav_paths:
        text = transcripts.get(wav.name, transcripts.get(wav.stem, ""))
        if text:
            audio_paths.append(str(wav))
            texts.append(text)

    logger.info(f"Generating {len(audio_paths)} synthetic conflict pairs → emotion={args.target_emotion}")

    from data.synthetic import generate_conflict_pairs, StarGANv2VoiceConverter

    converter = StarGANv2VoiceConverter()
    samples = generate_conflict_pairs(
        neutral_audio_paths=audio_paths,
        neutral_texts=texts,
        output_dir=args.output_dir,
        target_emotion=args.target_emotion,
        converter=converter,
    )

    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Manifest saved to {manifest_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
