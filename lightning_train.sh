#!/usr/bin/env bash
set -euo pipefail

echo "=== ConflictNet Lightning H200 Training ==="

# ── Install dependencies ──────────────────────────────────────────────
pip install -q -r requirements.txt
pip install -q sentencepiece tiktoken kagglehub speechbrain audiomentations

# ── Download CREMA-D ───────────────────────────────────────────────────
echo "Downloading CREMA-D..."
CREMAD_PATH=$(python -c "
import kagglehub
import os
path = kagglehub.dataset_download('ejlok1/cremad')
# kagglehub returns a versioned path like .../versions/1
# CREMA-D AudioWAV is directly in that directory
print(path)
")
echo "CREMA-D at: $CREMAD_PATH"

# ── Run training with auto-retry ────────────────────────────────────────
echo "Starting training on H200..."
python scripts/train.py \
  --cremad_root "$CREMAD_PATH" \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-5 \
  --audio_encoder wavlm \
  --gradient_accumulation_steps 1 \
  --pretrain_epochs 5 \
  --amp \
  --output_dir checkpoints \
  --target_f1 0.78 \
  --max_retries 2 \
  --resume_epochs 10

echo "=== Training complete! ==="
echo "Checkpoints saved to: checkpoints/"
ls -la checkpoints/

# ── Summary ────────────────────────────────────────────────────────────
if [ -f checkpoints/best_model_meta.json ]; then
  python3 -c "import json; m=json.load(open('checkpoints/best_model_meta.json')); print(f'Best val F1: {m[\"best_val_f1\"]:.4f} (epoch {m[\"epoch\"]})')"
fi
