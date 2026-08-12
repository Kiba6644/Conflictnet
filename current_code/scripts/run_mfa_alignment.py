#!/usr/bin/env python3
"""Run Montreal Forced Aligner (MFA) on a dataset and produce a word-level manifest.

Organises audio + transcripts in MFA-compatible directory structure, calls
``mfa align``, then parses all output TextGrids into a JSON manifest.

Usage:
    python scripts/run_mfa_alignment.py \
        --dataset_root /data/iemocap \
        --output_dir mfa_output
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_mfa() -> bool:
    """Return True if mfa is available on PATH."""
    return shutil.which("mfa") is not None


def prepare_mfa_corpus(
    dataset_root: str,
    output_dir: str,
    dataset_name: str = "iemocap",
) -> str:
    """Organise audio + lab files in MFA-compatible structure.

    Creates:
        output_dir/mfa_input/
          ├── audio/
          └── lab/

    Returns path to corpus directory.
    """
    mfa_dir = Path(output_dir) / "mfa_input"
    audio_dir = mfa_dir / "audio"
    lab_dir = mfa_dir / "lab"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    root = Path(dataset_root)
    count = 0

    # IEMOCAP
    for sess_dir in sorted(root.glob("Session*")):
        wav_root = sess_dir / "sentences" / "wav"
        trans_dir = sess_dir / "dialog" / "transcriptions"
        if not wav_root.exists() or not trans_dir.exists():
            continue

        transcripts: Dict[str, str] = {}
        for tf in trans_dir.glob("*.txt"):
            with open(tf) as f:
                for line in f:
                    line = line.strip()
                    if " [" in line:
                        utt_id, rest = line.split(" [", 1)
                        text = rest.split("]:")[1].strip() if "]:" in rest else ""
                        transcripts[utt_id.strip()] = text

        for wav in wav_root.rglob("*.wav"):
            utt_id = wav.stem
            text = transcripts.get(utt_id, "")
            if not text:
                continue
            # Copy audio
            dest_wav = audio_dir / f"{dataset_name}_{utt_id}.wav"
            shutil.copy2(str(wav), str(dest_wav))
            # Write lab file
            lab_path = lab_dir / f"{dataset_name}_{utt_id}.lab"
            with open(lab_path, "w") as lf:
                lf.write(text)
            count += 1

    logger.info(f"[MFA] Prepared {count} utterances in {mfa_dir}")
    return str(mfa_dir)


def run_mfa(
    corpus_dir: str,
    output_dir: str,
    acoustic_model: str = "english_us_mfa",
    dictionary: str = "english_us_arpa",
    n_jobs: int = 4,
) -> str:
    """Run ``mfa align`` and return path to output directory."""
    align_out = Path(output_dir) / "mfa_output"
    align_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        "--clean",
        "--overwrite",
        "-j", str(n_jobs),
        corpus_dir,
        dictionary,
        acoustic_model,
        str(align_out),
    ]

    logger.info(f"[MFA] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"[MFA] Alignment failed:\n{result.stderr}")
        raise RuntimeError(f"MFA alignment failed with return code {result.returncode}")

    logger.info(f"[MFA] Alignment complete → {align_out}")
    return str(align_out)


def parse_textgrid(textgrid_path: str) -> List[Dict[str, Any]]:
    """Parse a Praat TextGrid and return word-level intervals."""
    words = []
    try:
        import textgrid
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        for tier in tg.tiers:
            if tier.name.lower() in ("words", "word"):
                for interval in tier:
                    if interval.mark.strip():
                        words.append({
                            "word": interval.mark.strip(),
                            "start": round(interval.minTime, 3),
                            "end": round(interval.maxTime, 3),
                        })
                break
    except ImportError:
        # Manual TextGrid parser (simple format)
        logger.warning("[MFA] textgrid library not installed; using fallback parser")
        words = _parse_textgrid_fallback(textgrid_path)
    except Exception as e:
        logger.warning(f"[MFA] Failed to parse {textgrid_path}: {e}")

    return words


def _parse_textgrid_fallback(textgrid_path: str) -> List[Dict[str, Any]]:
    """Basic TextGrid parser — handles the standard word-tier format."""
    words = []
    with open(textgrid_path) as f:
        lines = f.readlines()

    in_word_tier = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("name"):
            if "word" in stripped.lower():
                in_word_tier = True
            else:
                in_word_tier = False
        if in_word_tier and stripped.startswith("intervals ["):
            in_word_tier = True
        if in_word_tier and "text =" in stripped:
            text = stripped.split("=", 1)[1].strip().strip('"')
            if text:
                words.append({"word": text, "start": 0.0, "end": 0.0})
        if in_word_tier and "xmin =" in stripped and words:
            val = float(stripped.split("=", 1)[1].strip())
            if words[-1]["start"] == 0.0:
                words[-1]["start"] = val
        if in_word_tier and "xmax =" in stripped and words:
            val = float(stripped.split("=", 1)[1].strip())
            if words[-1]["end"] == 0.0:
                words[-1]["end"] = val

    return words


def build_manifest(align_out: str, corpus_dir: str, output_dir: str) -> str:
    """Parse all TextGrids into a JSON manifest."""
    align_dir = Path(align_out)
    corpus_name = Path(corpus_dir).name

    manifest: Dict[str, Any] = {}
    for tg_file in sorted(align_dir.rglob("*.TextGrid")):
        utterance_id = tg_file.stem.replace(f"{corpus_name}_", "", 1) if tg_file.stem.startswith(f"{corpus_name}_") else tg_file.stem
        words = parse_textgrid(str(tg_file))
        manifest[utterance_id] = {
            "words": words,
            "textgrid_path": str(tg_file),
        }

    manifest_path = Path(output_dir) / "alignment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[MFA] Manifest: {len(manifest)} utterances → {manifest_path}")
    return str(manifest_path)


def parse_args():
    p = argparse.ArgumentParser(description="Run MFA alignment and produce word-level manifest")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--acoustic_model", type=str, default="english_us_mfa")
    p.add_argument("--dictionary", type=str, default="english_us_arpa")
    p.add_argument("--dataset_name", type=str, default="iemocap")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--skip_align", action="store_true", help="Skip MFA; just prepare corpus if TextGrids already exist")
    return p.parse_args()


def main():
    args = parse_args()

    if not check_mfa() and not args.skip_align:
        logger.error("MFA not found on PATH. Install via: conda install -c conda-forge montreal-forced-aligner")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare corpus
    corpus_dir = prepare_mfa_corpus(args.dataset_root, str(out_dir), args.dataset_name)

    # 2. Run alignment
    if args.skip_align:
        align_out = str(out_dir / "mfa_output")
        logger.info(f"[MFA] Skipping alignment, looking for TextGrids in {align_out}")
    else:
        align_out = run_mfa(corpus_dir, str(out_dir), args.acoustic_model, args.dictionary, args.n_jobs)

    # 3. Build manifest
    manifest_path = build_manifest(align_out, corpus_dir, str(out_dir))
    logger.info(f"[MFA] Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
