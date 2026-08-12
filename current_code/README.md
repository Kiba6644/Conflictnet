# ConflictNet v2

**Speaker-normalised cross-modal emotion conflict detection** with temporal context, interpretability, and multi-label subtype classification.

> Built on top of HuBERT-CLAP. ~70% reused from existing libraries; ~1,400 lines of novel code.

---

## Architecture

```
Audio (wav) ──► Emotion2Vec / WavLM / wav2vec2 ──► ProjectionHead ──┐
                                                                      ├──► FusionGate ──► TransformerTemporalContext ──► ConflictClassifier
Text (str) ───► DeBERTa-v3 + LoRA ──────────────► ProjectionHead ──┤                                                     ├── logits_type (sarcasm/suppression/deception)
                                                                      │                                                     ├── severity [0,1]
ECAPA-TDNN ──► SpeakerNormalizer (z-score) ─────────────────────────┘                                                     └── conflict_flag (bool)
                                                       │
              MFA TextGrids ──► WordLevelDivergence ───┘ (optional)
```

**Losses (jointly optimised via Kendall 2018 uncertainty weighting):**
1. Context-Gated InfoNCE contrastive loss (audio ↔ text alignment)
2. Multi-label BCE for conflict subtypes
3. Severity MSE regression
4. Self-supervised swap detection (pre-training phase)

---

## Quick Start

```bash
conda create -n conflictnet python=3.10 -y
conda activate conflictnet
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
```

### Reproduce baseline (IEMOCAP)
```bash
python scripts/train.py \
    --iemocap_root /data/iemocap \
    --audio_encoder emotion2vec \
    --epochs 30 \
    --pretrain_epochs 5 \
    --output_dir checkpoints/emotion2vec_run1
```

### Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/emotion2vec_run1/best_model.safetensors \
    --iemocap_root /data/iemocap \
    --fairness \
    --output_dir results/
```

### Run tests (no GPU needed)
```bash
pytest tests/ -v
```

---

## Project Structure

```
conflictnet/
├── configs/
│   └── default.yaml            # Hydra config: all hyperparameters
├── data/
│   ├── datasets.py             # IEMOCAP + MUStARD++ loaders + collate_fn
│   └── synthetic.py            # StarGANv2-VC conflict pair generation
├── models/
│   ├── encoders/
│   │   ├── audio.py            # Emotion2Vec, WavLM, wav2vec2
│   │   └── text.py             # DeBERTa-v3-large + LoRA
│   ├── speaker_norm/
│   │   └── speaker_norm.py     # ECAPA-TDNN + prosody z-score + cold-start
│   ├── temporal/
│   │   └── temporal.py         # Causal Transformer over dialogue turns ⭐
│   ├── alignment/
│   │   ├── alignment.py        # ProjectionHead + ContextGatedContrastiveLoss ⭐
│   │   └── word_divergence.py  # MFA word-level divergence features ⭐
│   ├── classifier/
│   │   └── classifier.py       # Multi-label subtype + severity head ⭐
│   └── conflictnet.py          # Full model assembly + MultiTaskLoss + SwapObjective
├── training/
│   ├── trainer.py              # Training loop + warmup cosine scheduler + WandB
│   └── curriculum.py           # CurriculumSampler ⭐
├── evaluation/
│   ├── metrics.py              # WAcc, macro-F1, per-type AP/AUC, severity MAE
│   ├── fairness.py             # FairLearn demographic parity + equalized odds
│   ├── attribution.py          # Captum integrated gradients (text + audio)
│   └── llm_baseline.py         # GPT-4o text-only ceiling
├── scripts/
│   ├── train.py                # CLI: train
│   ├── evaluate.py             # CLI: evaluate
│   └── generate_synthetic.py  # CLI: StarGANv2-VC data generation
├── tests/
│   └── test_components.py      # Unit tests (all components, no GPU required)
└── requirements.txt
```

⭐ = novel contribution

---

## Novel Components (~1,400 lines)

| Component | File | Lines | Novelty |
|-----------|------|-------|---------|
| Speaker normalization + cold-start | `models/speaker_norm/speaker_norm.py` | ~230 | ⭐ Novel |
| Transformer temporal context | `models/temporal/temporal.py` | ~130 | ⭐ Novel |
| Context-gated contrastive loss | `models/alignment/alignment.py` | ~110 | ⭐ Novel |
| Multi-label classifier + severity | `models/classifier/classifier.py` | ~90 | ⭐ Novel |
| Word-level divergence (MFA) | `models/alignment/word_divergence.py` | ~155 | ⭐ Novel |
| Self-supervised swap objective | `models/conflictnet.py` | ~50 | ⭐ Novel |
| Curriculum sampler | `training/curriculum.py` | ~60 | Semi-novel |
| Multi-task uncertainty loss | `models/conflictnet.py` | ~30 | Impl. |
| Full model assembly + forward | `models/conflictnet.py` | ~200 | Engineering |

---

## Datasets

| Dataset | Use | Source |
|---------|-----|--------|
| IEMOCAP | Primary train/eval | USC (request) |
| CREMA-D | Augmentation | HuggingFace |
| MUStARD++ | Sarcasm labels | GitHub |
| MELD | Dialogue context | HuggingFace |
| GoEmotions | Text pre-training | HuggingFace |
| VoxCeleb1/2 | Speaker embeddings | robots.ox.ac.uk |
| MUSAN | Noise augmentation | openslr.org |

---

## 12-Week Execution Plan

| Week | Milestone |
|------|-----------|
| 1 | Fork HuBERT-CLAP → reproduce baseline → swap DistilBERT→DeBERTa |
| 2 | Swap wav2vec2→Emotion2Vec → compare all 3 audio encoders |
| 3 | Add ECAPA-TDNN speaker norm → z-score pipeline |
| 4 | Build Transformer temporal context → integrate |
| 5 | Multi-label classifier + severity head |
| 6 | MFA word alignment → per-word divergence features |
| 7 | Self-supervised swap pre-training → synthetic data via StarGANv2-VC |
| 8 | Multi-task loss balancing → curriculum learning → full training |
| 9 | Captum integrated gradients → attribution maps |
| 10 | Evaluation — fairness, latency, LLM baseline, human eval |
| 11 | Ablation studies → error analysis |
| 12 | Paper writing + architecture diagram |

---

## Dependencies

```
torch>=2.1            transformers>=4.36    peft>=0.7
speechbrain>=1.0      captum>=0.7           fairlearn>=0.9
librosa>=0.10         praat-parselmouth>=0.4 torchaudio>=2.1
scikit-learn>=1.3     hydra-core>=1.3       wandb
audiomentations>=0.34 evaluate              datasets>=2.16
safetensors>=0.4      funasr  # optional
# Via conda: montreal-forced-aligner
# Via git clone: github.com/yl4579/StarGANv2-VC
```
