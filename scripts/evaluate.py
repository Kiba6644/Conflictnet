"""CLI: evaluate a trained ConflictNet checkpoint.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --iemocap_root /data/iemocap \
        --fairness \
        --attribution \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Fix for macOS OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Fix for DeBERTa v2 "fabs" TorchScript compilation bug
os.environ["PYTORCH_JIT"] = "0"


# Add project root to sys.path so 'data', 'models' etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ConflictNet v2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--mustard_wav_dir", type=str, default="utterances_final", help="Path to MUStARD wav files")
    p.add_argument("--case_root", type=str, default=None, help="CASE 2026 benchmark root")
    p.add_argument("--audio_encoder", type=str, default="emotion2vec",
                   choices=["emotion2vec", "wavlm", "wavlm_weighted", "wav2vec2"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fairness", action="store_true", help="Run fairness audit")
    p.add_argument("--attribution", action="store_true", help="Compute IG attribution")
    p.add_argument("--llm_baseline", action="store_true", help="Run GPT-4o text baseline")
    p.add_argument("--ood_probe", action="store_true", help="Run speaker-OOD probe")
    p.add_argument("--held_out_speakers", type=str, default=None,
                   help="Comma-separated held-out speaker IDs for OOD probe")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--prosody_stats", type=str, default=None,
                   help="Path to .pt file from compute_prosody_stats.py with per-utterance z-scores")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    from models.conflictnet import ConflictNet

    checkpoint_path = args.checkpoint
    from models.checkpoint_utils import load_checkpoint_state, extract_model_state

    ckpt = load_checkpoint_state(checkpoint_path, device=args.device)
    model_state = extract_model_state(ckpt)
    model = ConflictNet(audio_encoder_name=args.audio_encoder)
    model.load_state_dict(model_state, strict=False)
    model.to(args.device)
    model.eval()
    logger.info(f"[Eval] Loaded checkpoint from {args.checkpoint}")

    # Load pre-computed prosody z-scores if available
    prosody_lookup = None
    if args.prosody_stats:
        prosody_path = Path(args.prosody_stats)
        zscores_p = prosody_path.parent / f"{prosody_path.stem}.zscores.json"
        if not zscores_p.exists():
            zscores_p = prosody_path.with_suffix(".zscores.json")
        if zscores_p.exists():
            try:
                with open(zscores_p) as f:
                    raw = json.load(f)
                prosody_lookup = {k: torch.tensor(v) for k, v in raw.items()}
                if prosody_lookup:
                    logger.info(f"[Prosody] loaded {len(prosody_lookup)} z-score entries from {zscores_p}")
            except Exception as e:
                logger.warning(f"[Prosody] failed to load {zscores_p}: {e}")

    # --- Build eval dataset ---
    from data.datasets import IEMOCAPDataset, MUStARDDataset, CASEDataset, make_collate_fn

    eval_datasets = []
    if args.iemocap_root:
        eval_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[5]))
    if args.mustard_root:
        eval_datasets.append(MUStARDDataset(
            root=args.mustard_root, 
            wav_dir=args.mustard_wav_dir, 
            split="val"
        ))
    if args.case_root:
        eval_datasets.append(CASEDataset(args.case_root, split="val"))

    if not eval_datasets:
        raise ValueError("Provide at least one of --iemocap_root, --mustard_root, or --case_root")

    eval_set = ConcatDataset(eval_datasets)
    pin = (args.device != "cpu")  # pin_memory only valid for CUDA
    eval_collate = make_collate_fn(prosody_lookup=prosody_lookup)
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,          # let PyTorch handle pinning automatically
        collate_fn=eval_collate,
    )

    # --- Run inference ---
    all_probs, all_labels, all_genders = [], [], []
    all_severity_pred, all_severity_true = [], []
    sample_audio = []

    with torch.no_grad():
        for batch in eval_loader:
            # non_blocking=True pairs with pin_memory=True for async H→D transfers
            batch_gpu = {
                k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                audio=batch_gpu["audio"],
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                audio_attention_mask=batch_gpu.get("audio_attention_mask"),
                prosody_z=batch_gpu.get("prosody_z"),
            )
            all_probs.append(out.probs_type.cpu().numpy())
            all_labels.append(batch["conflict_type_labels"].numpy())
            if out.severity is not None:
                all_severity_pred.append(out.severity.squeeze(-1).cpu().numpy())
                all_severity_true.append(batch["severity"].squeeze(-1).numpy())
            all_genders.extend(batch.get("genders", [None] * batch["audio"].size(0)))
            # Store a few audio samples for attribution
            if len(sample_audio) < 4:
                sample_audio.append(batch_gpu["audio"][:1])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- Metrics ---
    from evaluation.metrics import compute_all_metrics, print_metrics

    sev_pred = np.concatenate(all_severity_pred) if all_severity_pred else None
    sev_true = np.concatenate(all_severity_true) if all_severity_true else None
    metrics = compute_all_metrics(all_probs, all_labels, sev_pred, sev_true)
    print_metrics(metrics)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Fairness audit ---
    if args.fairness:
        from evaluation.fairness import fairness_audit
        valid_genders = [g if g in ("M", "F") else "unknown" for g in all_genders]
        binary_pred = (all_probs >= 0.5).any(axis=1).astype(int)
        binary_true = all_labels.any(axis=1).astype(int)
        fairness_report = fairness_audit(binary_pred, binary_true, valid_genders)
        logger.info(f"[Fairness] Disparity={fairness_report['disparity']:.4f}")
        with open(out_dir / "fairness.json", "w") as f:
            json.dump(fairness_report, f, indent=2)

    # --- Attribution (first sample) ---
    if args.attribution and sample_audio:
        from evaluation.attribution import ConflictNetAttribution
        attr = ConflictNetAttribution(model, n_steps=50)
        sample = sample_audio[0]
        try:
            tokenizer = model.text_encoder.tokenizer
            sample_batch = eval_loader.dataset[0]
            input_ids = sample_batch["input_ids"].unsqueeze(0).to(args.device)
            attention_mask = sample_batch["attention_mask"].unsqueeze(0).to(args.device)
            text_saliency = attr.text_attribution(input_ids, attention_mask, sample)
            audio_saliency = attr.audio_attribution(sample, input_ids, attention_mask)
            if text_saliency is not None:
                token_scores = text_saliency[0].cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                saliency_data = {"tokens": tokens, "scores": token_scores}
                with open(out_dir / "attribution.json", "w") as f:
                    json.dump(saliency_data, f, indent=2)
                logger.info(f"[Attribution] Token saliency → {out_dir / 'attribution.json'}")
            if audio_saliency is not None:
                from safetensors.torch import save_file as st_save
                st_save({"audio_saliency": audio_saliency.cpu()}, str(out_dir / "audio_saliency.safetensors"))
                logger.info(f"[Attribution] Audio saliency → {out_dir / 'audio_saliency.safetensors'}")
        except Exception as e:
            logger.warning(f"[Attribution] Attribution failed: {e} — falling back gracefully")

    # --- LLM baseline ---
    if args.llm_baseline:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("[LLM Baseline] OPENAI_API_KEY not set, skipping")
        else:
            # Requires test items with 'text' field — skipping here without data
            logger.info("[LLM Baseline] Set up test_items list with 'text' field and call run_llm_baseline()")

    # --- Speaker-OOD probe ---
    if args.ood_probe:
        if not args.held_out_speakers:
            logger.warning("[OOD-Probe] --held_out_speakers not set, skipping")
        else:
            logger.info("[OOD-Probe] Running speaker-OOD evaluation...")
            import subprocess
            ood_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "evaluation" / "ood_probe.py"),
                "--checkpoint", args.checkpoint,
                "--held_out_speakers", args.held_out_speakers,
                "--batch_size", str(args.batch_size),
                "--device", args.device,
                "--output_dir", str(out_dir / "ood"),
            ]
            if args.iemocap_root:
                ood_cmd.extend(["--iemocap_root", args.iemocap_root])
            if args.mustard_root:
                ood_cmd.extend(["--mustard_root", args.mustard_root])
            if args.case_root:
                ood_cmd.extend(["--case_root", args.case_root])
            try:
                subprocess.run(ood_cmd, check=True, capture_output=True, text=True)
                logger.info(f"[OOD-Probe] Report saved to {out_dir / 'ood' / 'ood_probe.json'}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"[OOD-Probe] Failed: {e.stderr[:200]}")

    logger.info(f"[Eval] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
