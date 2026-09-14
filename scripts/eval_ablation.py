"""Ablation evaluation script for ConflictNet.

Evaluates 5 fusion modes on clean MELD data.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import f1_score, roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.experiment_config import ExperimentConfig
from models.conflictnet import ConflictNet
from data.datasets import MELDDataset, collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_mode(model, val_loader, mode, device):
    """Evaluate model with a specific fusion mode."""
    # Monkey patch the fuse method for this mode
    original_fuse = model.fuse
    
    def custom_fuse(audio_embed, text_embed, speaker_feat, word_div_feats=None):
        if mode == "text_only":
            audio_embed = torch.zeros_like(audio_embed)
            alpha = None
        elif mode == "audio_only":
            text_embed = torch.zeros_like(text_embed)
            alpha = None
        elif mode == "adaptive_router":
            if getattr(model, 'modality_router', None) is not None:
                alpha = model.modality_router(text_embed, audio_embed)
                audio_embed = (1 - alpha) * audio_embed
                text_embed = alpha * text_embed
            else:
                alpha = None
        elif mode == "fixed_moe":
            alpha = None
        else: # concat_linear
            alpha = None
            
        if model.use_speaker_norm:
            combined = torch.cat([audio_embed, text_embed, speaker_feat], dim=-1)
        else:
            combined = torch.cat([audio_embed, text_embed], dim=-1)
            
        if mode == "concat_linear":
            if not hasattr(model, '_ablation_concat_linear'):
                # Initialize deterministically for consistent eval
                torch.manual_seed(42)
                model._ablation_concat_linear = nn.Linear(combined.shape[-1], model.embed_dim).to(combined.device)
            fused_embed = model._ablation_concat_linear(combined)
            return fused_embed, alpha

        # Call the original fusion gate logic
        if hasattr(model, 'fusion_gate') and 'MoEFusion' in str(type(model.fusion_gate)):
            gate_feats = []
            if model.use_speaker_norm:
                gate_feats.append(speaker_feat)
            if model.use_word_divergence:
                if word_div_feats is None:
                    word_div_feats = torch.zeros(audio_embed.size(0), 11, device=audio_embed.device)
                gate_feats.append(word_div_feats)
            gate_feat_tensor = torch.cat(gate_feats, dim=-1)
            fused_embed = model.fusion_gate(combined, gate_feat_tensor)
        else:
            fused_embed = model.fusion_gate(combined)
            
        return fused_embed, alpha
        
    # Apply monkey patch
    import types
    model.fuse = types.MethodType(custom_fuse, model)
    model.eval()
    
    all_probs = []
    all_labels = []
    all_binary = []
    all_alphas = []
    all_class_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {mode}"):
            audio = batch["audio"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_attention_mask = batch.get("audio_attention_mask")
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(device)
            
            outputs = model(
                audio=audio,
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
                prosody_z=batch.get("prosody_z").to(device) if batch.get("prosody_z") is not None else None,
            )
            
            all_probs.append(outputs.probs_type.cpu().numpy())
            all_labels.append(batch["conflict_type_labels"].numpy())
            all_binary.append(batch["conflict_binary"].numpy())
            
            if getattr(outputs, 'router_alpha', None) is not None:
                all_alphas.append(outputs.router_alpha.cpu().numpy())
                all_class_labels.append(batch["conflict_type_labels"].numpy())
                
    # Restore original fuse
    model.fuse = original_fuse
    
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    binary = np.concatenate(all_binary, axis=0)
    
    mean_alpha = None
    alpha_per_class = None
    raw_alphas = None
    raw_classes = None
    if mode == "adaptive_router" and len(all_alphas) > 0:
        alphas = np.concatenate(all_alphas, axis=0).flatten()
        mean_alpha = float(np.mean(alphas))
        
        class_labels = np.concatenate(all_class_labels, axis=0)
        alpha_per_class = {}
        class_names = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        
        raw_alphas = alphas.tolist()
        raw_classes = []
        for i in range(len(alphas)):
            # find which class is 1
            idx = np.where(class_labels[i] == 1)[0]
            if len(idx) > 0:
                raw_classes.append(class_names[idx[0]])
            else:
                raw_classes.append("unknown")
        
        for i in range(min(labels.shape[1], 6)):
            idx = class_labels[:, i] == 1
            if np.any(idx):
                alpha_per_class[class_names[i]] = float(np.mean(alphas[idx]))
                
    conflict_prob = probs[:, :3].max(axis=1)
    binary_int = binary.astype(int)

    best_binary_f1 = 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        p = (conflict_prob > t).astype(int)
        f1 = f1_score(binary_int, p, zero_division=0)
        if f1 > best_binary_f1:
            best_binary_f1 = f1

    try:
        auc_binary = roc_auc_score(binary_int, conflict_prob)
    except ValueError:
        auc_binary = 0.5

    best_f1_weighted = 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        p = (probs >= t).astype(int)
        f1w = f1_score(labels, p, average="weighted", zero_division=0)
        if f1w > best_f1_weighted:
            best_f1_weighted = f1w

    res = {
        "f1_binary": float(best_binary_f1),
        "f1_weighted": float(best_f1_weighted),
        "auc_binary": float(auc_binary),
        "mean_alpha": mean_alpha,
        "alpha_per_class": alpha_per_class
    }
    if raw_alphas is not None:
        res["raw_alphas"] = raw_alphas
        res["raw_classes"] = raw_classes
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--meld_root", type=str, required=True)
    parser.add_argument("--meld_csv_dir", type=str, default=None)
    parser.add_argument("--textgrid_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer_path", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--text_encoder_path", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Checkpoint logic
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / f"{ckpt_path.stem}_meta.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = ExperimentConfig.from_dict(json.load(f)["config"])
    else:
        logger.warning(f"Config not found at {config_path}, using default config with adaptive_router=True")
        cfg = ExperimentConfig(use_adaptive_router=True)
        
    model = ConflictNet(
        audio_encoder_name=cfg.audio_encoder,
        embed_dim=cfg.embed_dim,
        use_speaker_norm=cfg.use_speaker_norm,
        use_temporal=cfg.use_temporal,
        use_word_divergence=cfg.use_word_divergence,
        use_cross_attn_injection=cfg.use_cross_attn_injection,
        use_speaker_adaptive_threshold=cfg.use_speaker_adaptive_threshold,
        use_baseline_subtract=cfg.use_baseline_subtract,
        lora_r=cfg.lora_r,
        use_adaptive_router=True, # force enable to load weights if present
    )
    
    logger.info(f"Loading checkpoint {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
    # filter out module. prefix from DDP
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
        if any("modality_router" in k for k in missing):
            logger.warning("Modality router weights missing! The adaptive_router mode will run with random weights.")
    
    model.to(args.device)
    model.eval()
    
    logger.info("Loading MELD val dataset")
    # For evaluate, MELD dev split
    val_dataset = MELDDataset(
        root=args.meld_root,
        tokenizer_name=args.tokenizer_path,
        split="dev",
        textgrid_root=args.textgrid_root,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    modes = ["text_only", "audio_only", "concat_linear", "fixed_moe", "adaptive_router"]
    results = {}
    
    for mode in modes:
        res = evaluate_mode(model, val_loader, mode, args.device)
        results[mode] = res
        logger.info(f"Mode: {mode} - f1_weighted: {res['f1_weighted']:.4f}, f1_binary: {res['f1_binary']:.4f}")
        
    out_json = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_json}")

if __name__ == "__main__":
    main()
