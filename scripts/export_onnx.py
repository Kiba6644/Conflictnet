"""Export ConflictNet to ONNX for optimized inference.

Usage:
    python scripts/export_onnx.py \\
        --checkpoint checkpoints/best_model.safetensors \\
        --output model.onnx

Note: Exports the full model at a fixed maximum audio length (10s = 160000 samples)
and fixed text length (512 tokens). Variable-length support requires dynamic axes
which most serving runtimes handle, but verification is more involved.
"""

from __future__ import annotations

import argparse
import logging

import torch

from models.checkpoint_utils import load_checkpoint_state, extract_model_state
from models.conflictnet import ConflictNet

logger = logging.getLogger(__name__)


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    max_audio_samples: int = 160000,
    max_text_len: int = 512,
    embed_dim: int = 256,
    opset_version: int = 17,
):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state = load_checkpoint_state(checkpoint_path, device=device)
    model_state = extract_model_state(state)

    model = ConflictNet(
        use_word_divergence=False,
    )
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    B = 1
    dummy_audio = torch.randn(B, max_audio_samples, device=device)
    dummy_input_ids = torch.randint(0, 100, (B, max_text_len), device=device)
    dummy_attn_mask = torch.ones(B, max_text_len, dtype=torch.long, device=device)

    dummy_audio_attn = torch.ones(B, max_audio_samples, dtype=torch.bool, device=device)
    dummy_prosody = torch.zeros(B, 3, device=device)

    logger.info("Tracing model with dummy inputs...")
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                args=(
                    dummy_audio,
                    dummy_input_ids,
                    dummy_attn_mask,
                    dummy_audio_attn,
                    None,  # context_embeds
                    None,  # context_padding
                    None,  # speaker_roles
                    dummy_prosody,
                    None,  # word_timestamps
                    None,  # token_word_boundaries
                    None,  # conflict_type_labels
                    None,  # severity_labels
                    None,  # conflict_binary_labels
                    False,  # pretraining
                ),
                f=output_path,
                input_names=[
                    "audio",
                    "input_ids",
                    "attention_mask",
                    "audio_attention_mask",
                    "prosody_z",
                ],
                output_names=[
                    "logits_type",
                    "probs_type",
                    "severity",
                    "conflict_flag",
                    "audio_embed",
                    "text_embed",
                    "speaker_feat",
                    "fused_embed",
                    "context_pooled",
                ],
                dynamic_axes={
                    "audio": {0: "batch", 1: "audio_len"},
                    "input_ids": {0: "batch", 1: "text_len"},
                    "attention_mask": {0: "batch", 1: "text_len"},
                    "audio_attention_mask": {0: "batch", 1: "audio_len"},
                    "logits_type": {0: "batch"},
                    "probs_type": {0: "batch"},
                    "severity": {0: "batch"},
                    "conflict_flag": {0: "batch"},
                    "audio_embed": {0: "batch"},
                    "text_embed": {0: "batch"},
                    "speaker_feat": {0: "batch"},
                    "fused_embed": {0: "batch"},
                    "context_pooled": {0: "batch"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            logger.info(f"ONNX model saved to {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.error(
                "This may be due to dynamic control flow in HuggingFace models. "
                "Try with a smaller max_audio_samples or use the sub-component export."
            )
            raise


def verify_onnx(onnx_path: str, device: str = "cpu"):
    import onnx
    import onnxruntime as ort

    logger.info(f"Verifying ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure is valid")

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"],
    )

    dummy_audio = torch.randn(1, 160000).numpy()
    dummy_input_ids = torch.randint(0, 100, (1, 512)).numpy()
    dummy_attn_mask = torch.ones((1, 512), dtype=torch.int64)
    dummy_audio_attn = torch.ones((1, 160000), dtype=torch.bool).numpy()
    dummy_prosody = torch.zeros((1, 3)).numpy()

    outputs = session.run(
        ["logits_type", "probs_type", "severity", "conflict_flag"],
        {
            "audio": dummy_audio,
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attn_mask,
            "audio_attention_mask": dummy_audio_attn,
            "prosody_z": dummy_prosody,
        },
    )
    logits, probs, severity, flag = outputs
    logger.info(f"  logits: {logits.shape}")
    logger.info(f"  probs:  {probs.shape}")
    logger.info(f"  severity: {severity.shape}")
    logger.info(f"  flag:   {flag.shape}")
    logger.info("ONNX model verified successfully")


def main():
    parser = argparse.ArgumentParser(description="Export ConflictNet to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.safetensors")
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verify", action="store_true", help="Run onnxruntime verification after export")
    parser.add_argument("--max_audio_samples", type=int, default=160000)
    parser.add_argument("--max_text_len", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        max_audio_samples=args.max_audio_samples,
        max_text_len=args.max_text_len,
        embed_dim=args.embed_dim,
        opset_version=args.opset,
    )

    if args.verify:
        verify_onnx(args.output, device=args.device)


if __name__ == "__main__":
    main()
