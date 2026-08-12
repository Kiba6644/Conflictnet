#!/usr/bin/env python3
"""
Local evaluation of ConflictNet on CREMA-D.
Usage:
    python run_eval_local.py --cremad_root <path_to_cremad> [--device cuda]

GTX 1650 note: runs in fp16, batch_size=8 to stay within 4GB VRAM.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Fix Windows terminal encoding (emoji support)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cremad_root", type=str, default=None,
                   help="Path to CREMA-D root (folder containing AudioWAV/)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .safetensors checkpoint (auto-detected if not set)")
    p.add_argument("--meta", type=str, default=None,
                   help="Path to _meta.json (auto-detected if not set)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size (lower if OOM, default 8 works on GTX 1650)")
    p.add_argument("--no_fp16", action="store_true", default=False,
                   help="Disable fp16 (use fp32 — slower but more precise)")
    return p.parse_args()


def find_file(name: str, search_roots: list):
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def find_cremad(hint):
    candidates = []
    if hint:
        candidates.append(hint)
    candidates += [
        r"C:\Users\Nithi\Downloads\cremad",
        r"C:\datasets\cremad",
        r"C:\data\cremad",
        str(REPO_ROOT / "data" / "cremad"),
        str(REPO_ROOT / "cremad"),
    ]
    for c in candidates:
        p = Path(c)
        if (p / "AudioWAV").exists():
            return str(p)
        if p.exists() and p.name == "AudioWAV":
            return str(p.parent)  # passed AudioWAV directly
    return None


def main():
    args = parse_args()

    # ── Find checkpoint ──────────────────────────────────────────────────────
    search_roots = [REPO_ROOT, REPO_ROOT / "models", Path.home() / "Downloads"]
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        find_file("best_model.safetensors", search_roots)
    if ckpt_path is None:
        ckpt_path = find_file("latest_model.safetensors", search_roots)
    if ckpt_path is None:
        print("❌ Checkpoint not found. Pass --checkpoint <path>")
        sys.exit(1)
    print(f"✅ Checkpoint : {ckpt_path}  ({ckpt_path.stat().st_size / 1e9:.2f} GB)")

    # ── Find meta ────────────────────────────────────────────────────────────
    meta_path = Path(args.meta) if args.meta else \
        find_file("best_model_meta.json", search_roots)
    if meta_path is None:
        meta_path = find_file("latest_model_meta.json", search_roots)
    meta = {}
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✅ Meta        : {meta_path}")
    else:
        print("⚠️  Meta JSON not found — using defaults (wavlm, embed_dim=256)")

    # ── Find CREMA-D ─────────────────────────────────────────────────────────
    cremad_root = find_cremad(args.cremad_root)
    if cremad_root is None:
        print("\n❌ CREMA-D not found.")
        print("   Download from: https://www.kaggle.com/datasets/ejlok1/cremad")
        print("   Then run:  python run_eval_local.py --cremad_root C:\\path\\to\\cremad")
        sys.exit(1)
    print(f"✅ CREMA-D     : {cremad_root}")

    # ── Device / precision ───────────────────────────────────────────────────
    device = args.device
    use_fp16 = (not args.no_fp16) and device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU         : {gpu}  ({vram:.1f} GB VRAM)")
    print(f"✅ Precision   : {'fp16' if use_fp16 else 'fp32'}\n")

    # ── Patch audio encoder to WavLM-large (1024-dim) ───────────────────────
    audio_file = REPO_ROOT / "models" / "encoders" / "audio.py"
    if audio_file.exists():
        content = audio_file.read_text()
        if '"microsoft/wavlm-base-plus"' in content:
            audio_file.write_text(content.replace(
                '"microsoft/wavlm-base-plus"', '"microsoft/wavlm-large"'))
            print("✅ Patched audio encoder → WavLM-large")

    # ── Peek at checkpoint to infer actual architecture ─────────────────────
    # meta.json can be stale — read true shapes from the saved tensors instead.
    from safetensors import safe_open
    ckpt_shapes = {}
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            ckpt_shapes[k] = tuple(f.get_slice(k).get_shape())

    # Infer temporal_max_turns from pos_encoding embedding shape
    if "temporal.pos_encoding.embedding.weight" in ckpt_shapes:
        inferred_max_turns = ckpt_shapes["temporal.pos_encoding.embedding.weight"][0]
        print(f"✅ Inferred temporal_max_turns={inferred_max_turns} from checkpoint")
    else:
        inferred_max_turns = 16  # ConflictNet default

    # Infer audio encoder dim from audio_proj shape
    if "audio_proj.net.0.weight" in ckpt_shapes:
        inferred_audio_dim = ckpt_shapes["audio_proj.net.0.weight"][1]
        print(f"✅ Inferred audio encoder dim={inferred_audio_dim} from checkpoint")
    else:
        inferred_audio_dim = 1024

    ec = meta.get("experiment_config", {})
    audio_enc_name = ec.get("audio_encoder", "wavlm")
    embed_dim      = ec.get("embed_dim", 256)

    if audio_enc_name == "wavlm" and inferred_audio_dim == 1024:
        print("📥 Ensuring WavLM-large is in HuggingFace cache...")
        # Clear offline flags that would silently block download
        for _v in ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]:
            os.environ.pop(_v, None)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                "microsoft/wavlm-large",
                ignore_patterns=["*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
            )
            print("✅ WavLM-large ready")
        except Exception as e:
            print(f"❌ WavLM-large download FAILED: {e}")
            print("   Checkpoint needs 1024-dim audio encoder but fallback is 768-dim.")
            print("   Run this once to download manually:")
            print("     python -c \"from huggingface_hub import snapshot_download; snapshot_download('microsoft/wavlm-large')\"")
            sys.exit(1)

    print("📥 Ensuring DeBERTa-v3-large is in HuggingFace cache...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "microsoft/deberta-v3-large",
            ignore_patterns=["*.h5", "flax_model*", "tf_model*", "*.ot"],
        )
        print("✅ DeBERTa-v3-large ready")
    except Exception as e:
        print(f"⚠️  Could not download DeBERTa-v3-large: {e}")
        print("   Text encoder will use fallback — evaluation will still run but text features degrade.")

    # ── Build model using EXACT flags from training meta.json ────────────────
    # NOTE: ConflictNet defaults differ from training config.
    # e.g. use_word_divergence=True (default) vs False (training) adds 8 dims
    # to classifier input (256→264), causing load_state_dict shape mismatch.
    # CVE-2025-32434: NOT applicable — .safetensors uses safetensors library
    # (no pickle), and .pt files use weights_only=True. Already mitigated.
    from models.conflictnet import ConflictNet
    from models.checkpoint_utils import load_checkpoint_state, extract_model_state

    print(f"\nBuilding ConflictNet  audio={audio_enc_name}  embed_dim={embed_dim}")
    print(f"  Flags from meta.json:")
    for flag in ["use_speaker_norm", "use_temporal", "use_cross_attn_injection",
                 "use_speaker_adaptive_threshold", "use_baseline_subtract", "use_word_divergence"]:
        print(f"    {flag} = {ec.get(flag, '(default)')}")

    model = ConflictNet(
        audio_encoder_name=audio_enc_name,
        embed_dim=embed_dim,
        # Replicate training config exactly to match checkpoint architecture:
        use_speaker_norm            = ec.get("use_speaker_norm",             True),
        use_temporal                = ec.get("use_temporal",                 True),
        use_cross_attn_injection    = ec.get("use_cross_attn_injection",     True),
        use_speaker_adaptive_threshold = ec.get("use_speaker_adaptive_threshold", True),
        use_baseline_subtract       = ec.get("use_baseline_subtract",        True),
        use_word_divergence         = ec.get("use_word_divergence",          False),  # was False in training!
        # Use inferred value from checkpoint shape — meta.json can be wrong
        # (e.g. meta says 8 but checkpoint tensor is [16, 256] → use 16)
        temporal_max_turns          = int(inferred_max_turns),
        lora_r                      = ec.get("lora_r",                       16),
    )
    print(f"  temporal_max_turns = {int(inferred_max_turns)} (from checkpoint, meta said {ec.get('temporal_max_turns', '?')})"
          if int(inferred_max_turns) != ec.get("temporal_max_turns") else "")

    print("Loading weights...")
    ckpt = load_checkpoint_state(str(ckpt_path), device="cpu")
    missing, unexpected = model.load_state_dict(extract_model_state(ckpt), strict=False)
    if missing:
        print(f"  ⚠️  Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

    # fp16: halves weight memory (3 GB → 1.5 GB), essential for 4 GB GPUs.
    # model.half() converts stored weights; then we cast inputs to match.
    if use_fp16:
        model.half()
    model = model.to(device=device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready   params={total_params:,}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    from data.datasets import CREMADDataset, make_collate_fn
    ds = CREMADDataset(cremad_root, split="val")
    # num_workers=0 on Windows: collate_fn is a local closure and can't be
    # pickled by the multiprocessing spawn method used on Windows.
    n_workers = 0 if sys.platform == "win32" else 2
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=n_workers, pin_memory=(device == "cuda"),
                        collate_fn=make_collate_fn())
    print(f"Evaluating {len(ds)} validation samples  (batch={args.batch_size})...")

    # ── Inference loop ───────────────────────────────────────────────────────
    all_probs, all_labels, all_sev_pred, all_sev_true = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i, b in enumerate(loader):
            bg = {}
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point() and use_fp16:
                        bg[k] = v.to(device=device, dtype=torch.float16, non_blocking=True)
                    else:
                        bg[k] = v.to(device=device, non_blocking=True)
                else:
                    bg[k] = v

            out = model(
                audio=bg["audio"],
                input_ids=bg["input_ids"],
                attention_mask=bg["attention_mask"],
            )
            all_probs.append(out.probs_type.float().cpu().numpy())
            all_labels.append(b["conflict_type_labels"].numpy())
            if out.severity is not None:
                all_sev_pred.append(out.severity.squeeze(-1).float().cpu().numpy())
            if "severity" in b:
                all_sev_true.append(b["severity"].squeeze(-1).numpy())

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                used_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"  [{i+1}/{len(loader)}]  {elapsed:.1f}s  VRAM: {used_gb:.1f} GB")

    # ── Metrics ──────────────────────────────────────────────────────────────
    from evaluation.metrics import compute_all_metrics

    elapsed = time.time() - t0
    probs   = np.concatenate(all_probs)
    labels  = np.concatenate(all_labels)
    sev_pred = np.concatenate(all_sev_pred) if all_sev_pred else None
    sev_true = np.concatenate(all_sev_true) if all_sev_true else None
    metrics  = compute_all_metrics(probs, labels, severity_pred=sev_pred, severity_true=sev_true)

    print(f"\n{'='*55}")
    print(f"  CREMA-D Evaluation  —  {len(ds)} samples  in {elapsed:.1f}s")
    print(f"{'='*55}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
    print(f"  {'best_val_f1 (during training)':35s}: {meta.get('best_val_f1', 'N/A')}")
    print(f"  {'epoch':35s}: {meta.get('epoch', 'N/A')}")
    print(f"{'='*55}")

    # ── Save report ──────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    report = {
        "dataset": "cremad",
        "n_samples": len(ds),
        "eval_time_s": round(elapsed, 2),
        "checkpoint": str(ckpt_path),
        "best_val_f1_training": meta.get("best_val_f1"),
        "epoch": meta.get("epoch"),
    }
    report.update({k: v for k, v in metrics.items() if isinstance(v, float)})
    out_file = out_dir / "cremad_metrics.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved → {out_file}")


if __name__ == "__main__":
    main()
