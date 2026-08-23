# ConflictNet v2

**Speaker-normalized cross-modal emotion conflict detection** with multi-turn dialogue context, multi-dataset integration, robust DDP multi-GPU scaling, and interpretability.

> Built on top of HuBERT-CLAP architecture. Features multi-dataset unified training across IEMOCAP, MUStARD++, CREMA-D, and MELD.

---

## 🌟 Key Updates & Improvements

Compared to earlier versions, ConflictNet v2 includes major architectural and pipeline upgrades:

* 📚 **Multi-Dataset Unified Pipeline:** Full support for loading and joint training across **IEMOCAP**, **MUStARD++**, **CREMA-D**, and **MELD** datasets simultaneously.
* ⚡ **Robust DDP Multi-GPU Execution:**
  * **Zero NCCL Deadlocks:** Clean separation of inference evaluation (`_model_for_eval`) to prevent rank desynchronization and collective timeouts.
  * **Rank-Zero Cache Warming:** Automatic single-rank downloading of HuggingFace and SpeechBrain models before multi-process initialization.
  * **Exact Gradient Accumulation Sync:** Guarantees synchronization on epoch boundaries even when batch counts are indivisible by accumulation steps.
* 🎙️ **Flexible Audio Backends:** Dynamic support for **Emotion2Vec**, **WavLM**, **FunASR**, and **wav2vec2** audio encoders with fallback mechanisms.
* 🗣️ **Speaker Normalization:** ECAPA-TDNN speaker embeddings combined with prosody z-score normalization.
* 🧠 **Dialogue Temporal Context Transformer:** Causal Transformer modeling past conversational turns with an isolated turn context cache (`ContextCache`).
* 🎯 **Multi-Task & Focal Loss Engine:** Multi-label Focal BCE loss targeting hard emotion cases, Context-Gated InfoNCE contrastive alignment, Severity MSE, and Kendall uncertainty loss weighting.
* 💾 **Safe Checkpointing & Auto-Retries:** `safetensors` model serialization and automated learning rate decay retry loops targeting goal F1 scores.

---

## 📐 Architecture Overview

```
Audio (wav) ──────► Emotion2Vec / WavLM / wav2vec2 ──► ProjectionHead ──┐
                                                                         ├──► FusionGate ──► TransformerTemporalContext ──► ConflictClassifier
Text (str) ───────► DeBERTa-v3 + LoRA ────────────────► ProjectionHead ──┤                                                     ├── logits_type (sarcasm/suppression/deception)
                                                                         │                                                     ├── severity [0,1]
ECAPA-TDNN ───────► SpeakerNormalizer (z-score) ─────────────────────────┘                                                     └── conflict_flag (bool)
                                                         │
                 MFA TextGrids ──► WordLevelDivergence ───┘ (optional)
```

### Multi-Task Objectives
1. **Focal BCE Loss:** Focuses optimization on hard minority conflict types (anger, disgust, fear).
2. **Context-Gated InfoNCE:** Cross-modal contrastive alignment between speech and text modalities.
3. **Severity MSE Regression:** Estimates conflict severity score.
4. **Self-Supervised Swap Objective:** Detects audio-text modality mismatches during pre-training.

---

## 📊 Supported Datasets

| Dataset | Modality | Primary Annotations |
| :--- | :--- | :--- |
| **IEMOCAP** | Audio + Text | Multi-speaker dyadic dialogue emotions |
| **MUStARD++** | Audio + Text | Sarcasm & implicit sentiment in TV dialogue |
| **CREMA-D** | Audio + Text | Multi-actor emotional speech expressions |
| **MELD** | Audio + Text | Multi-party conversational emotion & sentiment |

---

## 🚀 Quick Start

### 1. Installation

```bash
conda create -n conflictnet python=3.10 -y
conda activate conflictnet
pip install -r requirements.txt

# Optional: Montreal Forced Aligner for word-level alignment
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
```

### 2. Multi-Dataset Training (Single GPU or DDP)

#### Single-GPU Run:
```bash
python scripts/train.py \
    --iemocap_root /path/to/IEMOCAP \
    --mustard_root /path/to/MUStARD \
    --cremad_root /path/to/CREMA-D \
    --meld_root /path/to/MELD \
    --audio_encoder emotion2vec \
    --epochs 30 \
    --pretrain_epochs 5 \
    --output_dir checkpoints/multi_dataset_run
```

#### DDP Multi-GPU Run (e.g. 2 GPUs via `torchrun`):
```bash
torchrun --nproc_per_node=2 scripts/train.py \
    --iemocap_root /path/to/IEMOCAP \
    --meld_root /path/to/MELD \
    --audio_encoder emotion2vec \
    --epochs 30 \
    --amp \
    --output_dir checkpoints/ddp_run
```

### 3. Evaluation & Metrics

Evaluate trained models for Weighted Accuracy (WAcc), Macro-F1, AP, AUC, and Fairness metrics:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/multi_dataset_run/best_model.safetensors \
    --iemocap_root /path/to/IEMOCAP \
    --fairness \
    --output_dir results/
```

### 4. Run Unit Tests (No GPU Required)

```bash
pytest tests/ -v
```

---

## 🛠️ Repository Structure

```
ConflictNet-main/
├── configs/
│   └── default.yaml            # Hydra configuration
├── data/
│   ├── datasets.py             # IEMOCAP, MUStARD++, CREMA-D, MELD loaders & collate
│   └── synthetic.py            # Synthetic pair generation
├── models/
│   ├── encoders/
│   │   ├── audio.py            # Emotion2Vec, WavLM, FunASR, wav2vec2
│   │   └── text.py             # DeBERTa-v3-large + LoRA
│   ├── speaker_norm/
│   │   └── speaker_norm.py     # ECAPA-TDNN + prosody z-score
│   ├── temporal/
│   │   └── temporal.py         # Multi-turn dialogue Transformer context
│   ├── alignment/
│   │   ├── alignment.py        # Context-Gated InfoNCE Contrastive Loss
│   │   └── word_divergence.py  # MFA word-level divergence features
│   ├── classifier/
│   │   └── classifier.py       # Subtype logits & severity heads
│   └── conflictnet.py          # Master ConflictNet architecture & multi-task loss
├── training/
│   ├── trainer.py              # DDP trainer, warmup scheduler, context cache, evaluation
│   └── curriculum.py           # Curriculum learning sampler
├── evaluation/
│   ├── metrics.py              # WAcc, Macro-F1, AUC, AP evaluation
│   ├── fairness.py             # Demographic parity & equalized odds
│   ├── attribution.py          # Captum integrated gradients
│   └── ood_probe.py            # Out-of-distribution evaluation
├── scripts/
│   ├── train.py                # Main training CLI with auto-retry target F1
│   ├── evaluate.py             # Evaluation CLI
│   └── compute_prosody_stats.py# Prosody z-score computation utility
├── FUTURE_UPGRADES.md          # Technical roadmap to 90%+ accuracy
├── README(old).md               # Legacy README backup
└── requirements.txt
```

---

## 📖 Roadmap & Future Upgrades

See [FUTURE_UPGRADES.md](file:///c:\Users\Nithi\Documents\Github\ConflictNet-main\FUTURE_UPGRADES.md) for architectural plans on pushing model performance to 90%+ Accuracy/F1, including cross-attention fusion, modality dropout, and backbone scaling.
