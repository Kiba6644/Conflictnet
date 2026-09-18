# ConflictNet v2 — Codebase Architecture & Technical Reference

> [!IMPORTANT]
> **To all future AI agents and engineers**: Read this document before inspecting individual files. This file documents the complete system architecture, data division protocols, custom distributed samplers, standalone patches, resolved bugs, and directory structure. You do **not** need to re-scan all 75+ files in the repository to understand how the codebase works.

---

## 1. High-Level Architecture Overview

ConflictNet v2 is an end-to-end multimodal framework designed for detecting affective conflict, sarcasm, and emotional divergence across dyadic and multiparty spoken dialogues.

```
                    ┌─────────────────────────┐
                    │      Input Dialogue     │
                    │   (Audio + Transcript)  │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Audio Encoder   │  │   Text Encoder   │  │   Speaker Norm   │
│(WavLM/Emotion2Vec│  │(DeBERTa-v3-large │  │ (ECAPA-TDNN 192d │
│  + Audio Proj)   │  │    + LoRA r=32)  │  │ + Prosody Z-score│
└─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
          │                     │                     │
          └──────────┬──────────┘                     │
                     ▼                                │
      ┌─────────────────────────────┐                 │
      │  Cross-Modal Attention      │                 │
      │ (Audio ↔ Text Token Masked) │                 │
      └──────────────┬──────────────┘                 │
                     ▼                                │
      ┌─────────────────────────────┐                 │
      │ Word Divergence (MFA CTC)   │                 │
      └──────────────┬──────────────┘                 │
                     ▼                                │
      ┌───────────────────────────────────────────────┴┐
      │        Fusion Gate / MoE Router                │
      │  (Audio + Text + Speaker + Word Divergence)    │
      └──────────────────────┬─────────────────────────┘
                             ▼
              ┌─────────────────────────────┐
              │ Temporal Dialogue Context   │
              │(Transformer + Speaker Roles)│
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │     Classification Head     │
              │ (6-Emotion / Conflict Flag /│
              │  Severity Regression Head)  │
              └─────────────────────────────┘
```

### Core Components

1. **Audio Pipeline (`models/audio/encoder.py`)**:
   - Backends supported: `emotion2vec` (default), `wavlm`, `wavlm_weighted` (layer-weighted combination), and `wav2vec2`.
   - Feature projections: Projects pooled acoustic representations to shared `embed_dim` (256). Optionally returns frame-level embeddings for word divergence cross-attention.
   - DDP Pre-extraction bypass: When running under multi-GPU DDP, heavy sequential extractors (e.g. FunASR) can be pre-extracted into `.pt` feature dictionaries to prevent NCCL thread timeouts.

2. **Text Pipeline (`models/text/encoder.py`)**:
   - Backbone: `microsoft/deberta-v3-large` fine-tuned via Low-Rank Adaptation (LoRA rank $r=32$, $\alpha=32$).
   - Returns both pooled utterance representation and token-level embeddings.
   - Text projection layer maps raw DeBERTa dimension (1024) to shared `embed_dim` (256).

3. **Speaker Normalization (`models/speaker_norm.py`)**:
   - Disentangles expressive speaker traits from true conflict using two signals:
     - 192-dimensional speaker acoustic embedding from SpeechBrain ECAPA-TDNN.
     - 3-dimensional pre-computed prosody z-scores ($F_0$, RMS energy, speech rate), normalized strictly against speaker baseline statistics.
   - Bypasses internal DDP buffer all-reduces by supporting precomputed speaker embeddings (`precomputed_speaker_embed`).

4. **Word-Level Divergence (`models/fusion/word_divergence.py`)**:
   - Computes acoustic-lexical semantic mismatch across aligned words.
   - Receives MFA (Montreal Forced Aligner) or CTC Wav2Vec2 word boundaries and aligns acoustic frames against text subwords using cross-attention.

5. **Cross-Modal Attention & Fusion (`models/fusion/gate.py`)**:
   - Pre-fusion audio-text cross-modal attention masks padding tokens to prevent attention leakage.
   - Fusion strategies:
     - `GatedFusion`: Context-aware sigmoid gating between audio and text conditioned on speaker features.
     - `MoEFusion`: Dynamic mixture-of-experts routing audio, text, and speaker representations.
     - `AdaptiveRouter`: Sparsely activates expert branches with entropy regularization.

6. **Temporal Dialogue Transformer (`models/temporal/dialogue_transformer.py`)**:
   - Models dialogue history across conversational turns.
   - Incorporates learned speaker-role positional embeddings (Speaker A vs. Speaker B) and dialogue turn indices.
   - State cached incrementally in `data/context_cache.py` during inference and training.

7. **Classification & Severity Heads (`models/classifier/classifier.py`)**:
   - Multi-label subtype head predicting 6 emotion classes or 3 conflict classes.
   - Continuous severity regression head ($[0, 1]$) with Sigmoid activation.
   - Speaker-adaptive threshold network: Computes a learned positive offset $[\tau_{\text{base}}, \tau_{\text{base}} + \Delta]$ from speaker features to prevent expressive speakers from triggering false-positive conflict flags.

8. **Loss Formulation**:
   - Multi-task loss dynamically weighted using `AutomaticWeightedLoss` (Kendall et al. homoscedastic uncertainty weighting).
   - InfoNCE contrastive loss aligns audio and text embeddings (with sarcasm samples explicitly masked to prevent forcing contradictory modalities together).

---

## 2. Dataset Division Protocols & Splitting Integrity

Preventing conversational and speaker leakage across train/val/test splits is strictly enforced across all datasets.

### A. MELD (Multimodal EmotionLines Dataset)
- **Structure**: Multiparty conversational dialogues from the TV show *Friends*.
- **Emotion Classes (6 standard ConflictNet mapping)**:
  - `0`: Anger (Conflict)
  - `1`: Disgust (Conflict)
  - `2`: Fear (Conflict)
  - `3`: Happiness / Joy (Non-conflict)
  - `4`: Neutral (Non-conflict, ~59% majority)
  - `5`: Sadness (Non-conflict)
- **Dialogue-Stratified 80/20 Split**:
  - `MELDDataset` in `data/datasets.py` groups samples by `dialogue_id`.
  - The split is executed at the **dialogue level**, ensuring that all turns of a conversation stay in either the train split or the validation split. No dialogue is ever split across sets.
- **Minority Dialogue Oversampling**:
  - Because MELD is dominated by Neutral (~59%) and rare classes like Fear represent only ~2.5%, training applies dialogue-level oversampling: conversations containing rare conflict emotions (Anger, Disgust, Fear) are sampled with higher probability during training.
- **Feature Caching**:
  - Precomputed audio representations are cached into a `.pt` dictionary (`utt_id -> {audio_embed, speaker_embed}`) via `fix_datasets.py` to allow lightning-fast training without disk audio decode bottlenecks.

### B. MUStARD (Multimodal Sarcasm Detection)
- **Structure**: Dyadic sarcasm benchmark from sitcoms (*Friends*, *The Big Bang Theory*).
- **Splitting**: **Speaker-Stratified 80/20 Split**.
  - Partitioning ensures that speakers in the validation split do not appear in the training split, forcing the model to learn multimodal sarcasm patterns rather than memorizing individual character quirks.

### C. CREMA-D (Crowd-sourced Emotional Multimodal Actors)
- **Structure**: 7,442 clips from 91 ethnically diverse actors (48 male, 43 female).
- **Splitting**: **Actor-Stratified 70/15/15 Split**.
  - Partitioned by actor ID (`utt_id[:6]` / `Ses01F...`), guaranteeing that actors in the test set are completely unseen during training.
- **Dynamic Severity Calculation (`data/datasets.py:890-917`)**:
  - Conflict emotions (`anger`, `disgust`, `fear`) derive continuous severity from actor intensity ratings:
    $$\text{severity} = \begin{cases} 0.33 & \text{if Low (LO)} \\ 0.66 & \text{if Medium (MD)} \\ 1.00 & \text{if High (HI)} \end{cases}$$
  - Non-conflict emotions (`neutral`, `happy`, `sad`) assign $\text{severity} = 0.0$.

### D. IEMOCAP
- **Structure**: 5 dyadic sessions of improvised and scripted emotional dialogues.
- **Splitting**: Standard Session 5 held-out evaluation (Sessions 1–4 for training, Session 5 for testing).

---

## 3. Data Samplers & DDP Streaming (`data/samplers.py`)

Temporal context modeling requires conversational turns to be processed in strict chronological sequence. Standard random batch sampling breaks dialogue continuity.

### `DialogueDistributedBatchSampler`
- **Chronological Dialogue Ordering**:
  - Groups utterance indices by dialogue.
  - Generates batches such that all turns of a conversation are visited sequentially on the same GPU rank.
  - Enables `ContextCache` to maintain an active history buffer across turns without manual state tracking.
- **DDP Rank Partitioning**:
  - Assigns complete dialogues across distributed GPU ranks.
  - Pads dialogue batches evenly across ranks so all workers execute the exact same number of training steps, avoiding NCCL all-reduce hangs or barrier deadlocks.

---

## 4. Standalone Patches & Tool Scripts

The codebase includes specialized standalone utilities designed for training acceleration, evaluation, and data preparation:

| Script | Purpose & Usage |
|---|---|
| `fix_datasets.py` | Patches `data/datasets.py` to support loading pre-extracted `.pt` audio feature dictionaries (e.g. `meld_features.pt`). Automatically detects `.pt` files and injects fast tensor lookup loaders. |
| `run_eval_local.py` | Standalone, lightweight evaluation runner. Allows offline inference and metric reporting on CPU or a single GPU without requiring multi-GPU torchrun environments. |
| `scripts/eval_ablation.py` | Fast 5-way ablation evaluator. Tests Audio-only, Text-only, Concat, Gated, and MoE Router modalities on a trained checkpoint by dynamically monkey-patching `model.fuse`, avoiding retraining. |
| `scripts/run_mfa_meld.py` | In-memory CTC Wav2Vec2 forced aligner. Automatically computes word boundaries and generates Praat `.TextGrid` files for MELD without requiring external Kaldi / MFA installations. |
| `scripts/run_meld_kaggle.py` | Automated dual-T4 / P100 Kaggle environment runner. Sets up paths, handles HuggingFace/SpeechBrain caching, and launches DDP training with automatic retries. |
| `scripts/compute_prosody_stats.py` | Extracts speaker $F_0$, RMS energy, and speech rate statistics. Saves `*.train_only.json` computed strictly over training utterances to guarantee zero validation data leakage. |

---

## 5. Technical Post-Mortem: Resolved Bugs & Quirks

### 1. The Metric Collapse Bug (`training/trainer.py`) — RESOLVED
- **Root Cause**:
  In earlier runs, `training/trainer.py` contained an uncalibrated prior division:
  ```python
  # BROKEN:
  balanced_scores = probs / class_priors
  y_pred_cls = np.argmax(balanced_scores, axis=1)
  ```
  In MELD, Fear has a prior of ~0.025, while Neutral is ~0.59. Dividing sigmoid probabilities by $0.025$ multiplied Fear probabilities by $40\times$. This caused `y_pred_cls` to predict Fear for **100% of samples**.
  - Macro-F1 collapsed to $0.0487 / 6 = \mathbf{0.0065}$.
  - Weighted-F1 collapsed to $0.025 \times 0.0487 = \mathbf{0.0008}$.
  - Because `val/f1_weighted` was locked at $0.0008$, it never exceeded the initial checkpoint's `best_val_f1` ($0.4005$). As a result, `best_model.safetensors` was **never saved**, early stopping aborted training prematurely at Epoch 43, and the final evaluation evaluated an over-decayed checkpoint with LR at $9.75 \times 10^{-8}$.
- **Resolution**:
  - Replaced with standard multi-class argmax for single-label emotion datasets (`np.argmax(probs, axis=1)`).
  - Restores true Weighted-F1 ($\sim 0.45\text{--}0.55$) and Macro-F1 ($\sim 0.35\text{--}0.45$), enabling proper `is_best` checkpoint tracking and preventing premature early stopping.

### 2. Binary Conflict Thresholding (`val/f1_binary: 0.0000`) — RESOLVED
- **Root Cause**:
  Under BCE loss with extreme class imbalance, raw sigmoid probabilities for minority conflict emotions (anger, disgust, fear) peak around $0.20\text{--}0.35$. A rigid $0.5$ threshold resulted in zero positive predictions, producing `val/f1_binary: 0.0000`.
- **Resolution**:
  - `training/trainer.py` sweeps binary thresholds $\tau \in [0.05, 0.95]$ on validation outputs to find the optimal decision boundary ($\tau \approx 0.35$).
  - `val/f1_binary` now reports the calibrated score ($\text{F1} \approx 0.33$), while `val/f1_binary_raw05` is logged for reference.
  - Optimal thresholds are stored in `best_model_meta.json` under `best_binary_thresh` and automatically loaded by `scripts/evaluate.py`.
  - Fixed `f1_macro_cal` in trainer sweeps so it tracks true macro F1 at optimal threshold.

### 3. `evaluation/metrics.py: preds_type` UnboundLocalError — RESOLVED
- **Root Cause**: `preds_type` was only defined in the `else` (multi-label) branch. When evaluating single-label datasets (`is_single_label=True`), iterating through `preds_type[:, i]` raised `UnboundLocalError`.
- **Resolution**: Initialized `preds_type = (probs_type >= type_threshold).astype(int)` prior to the branch.

### 4. DDP Barrier Deadlocks on Exceptions — RESOLVED
- **Root Cause**: In multi-GPU DDP runs, if Rank 0 encountered an OOM or data error during validation metric broadcast, non-zero ranks remained blocked forever waiting on `torch.distributed.broadcast`.
- **Resolution**: Added sentinel broadcast in `trainer.py:729` sending `[-1.0, ...]` to unblock worker ranks before re-raising exceptions.

### 5. Multi-Class Cross-Entropy with Class Weighting & Label Smoothing — IMPLEMENTED
- **Problem**: Multi-label Focal BCE treated 6 mutually-exclusive emotion classes as independent binary problems. Minority classes (Fear ~2.5%, Disgust ~3%) received near-zero probability output (<0.05).
- **Resolution**:
  - For single-label emotion datasets (MELD, CREMA-D, IEMOCAP), routed loss to `nn.functional.cross_entropy` with inverse-frequency class weights `class_weights = torch.tensor([1.5, 3.5, 3.5, 1.0, 0.4, 2.2])` and label smoothing `0.05`.
  - Multi-label datasets (MUStARD, CASE) retain Focal BCE.
  - In `ConflictClassifier`, single-label datasets compute normalized Softmax probabilities (`torch.softmax(logits_type, dim=-1)`), while multi-label datasets output Sigmoid probabilities.
  - Added `--no_class_weights` CLI flag to allow unweighted cross-entropy ablation.

### 6. Stochastic Modality Dropout (Audio-Text Regularization) — IMPLEMENTED
- **Problem**: DeBERTa-v3 is significantly stronger than raw acoustic backends on textual emotion clues, causing text dominance where acoustic emotion representations are ignored during multimodal fusion.
- **Resolution**:
  - In `ConflictNet.fuse`, stochastically zero out audio ($p=0.15$) or text ($p=0.15$) embeddings with mutually-exclusive random masks during training.
  - Forces the network to learn robust acoustic emotion representations from audio alone when text is dropped, while keeping InfoNCE contrastive alignment inputs intact.
  - Inactive during evaluation (`self.training == False`).
  - Configurable via `--modality_dropout <float>` (default: 0.15).

### 7. Multi-Party Conversational Speaker Modeling — IMPLEMENTED
- **Problem**: `SpeakerRoleEmbedding` previously supported only 2 dyadic speakers (`nn.Embedding(2, embed_dim)`). Multiparty conversations (e.g. MELD 6 Friends characters) had character identities collapsed.
- **Resolution**:
  - Extended `SpeakerRoleEmbedding` to `num_speakers=16` with clamped indices `[0, 15]`.
  - Added speaker role extraction in `data/datasets.py:_collate_core` mapping MELD characters (Chandler=0, Joey=1, Monica=2, Phoebe=3, Rachel=4, Ross=5) and hashing other speakers.
  - Updated `ContextCache` to store and retrieve `(turn_index, turn_embed, speaker_role)` tuples with backwards-compatible `return_roles=True`.
  - Assembled sequence-aligned `full_speaker_roles` `(B, T_ctx + 1)` in `ConflictNet.forward` and passed to `TransformerTemporalContext`.

### 8. `NameError: embed_dim_val` in `trainer.train_epoch` — RESOLVED
- **Root Cause**: `embed_dim_val` definition was accidentally dropped during a line replacement in `train_epoch`.
- **Resolution**: Defined `_model_inner = getattr(self.model, "module", self.model)` and `embed_dim_val = getattr(_model_inner, "embed_dim", 256)` before `get_batch_context()`. Verified with live `train_epoch` execution.

---

## 6. Directory Map & File Index

```
ConflictNet-main/
├── CODEBASE_STRUCTURE.md      # THIS FILE: Master reference for codebase & patches
├── ARCHITECTURE.md            # High-level architecture specification
├── AUDIT_FINDINGS.md          # Historical audit logs
├── fix_datasets.py            # Precomputed .pt audio feature dictionary patch
├── run_eval_local.py          # Standalone offline local evaluation runner
├── requirements.txt           # Python dependencies
├── pyrightconfig.json         # Language server configuration
│
├── configs/                   # Experiment and ablation YAML configs
│   ├── default.yaml           # Full ConflictNet model configuration
│   ├── ablate_no_temporal.yaml
│   ├── ablate_no_speaker_norm.yaml
│   ├── ablate_no_word_div.yaml
│   ├── ablate_no_cross_attn.yaml
│   ├── ablate_no_adaptive_threshold.yaml
│   └── ablate_no_baseline_subtract.yaml
│
├── data/                      # Dataset loaders, augmentations, and samplers
│   ├── datasets.py            # MELD, MUStARD, CREMA-D, IEMOCAP, CASE loaders & split logic
│   ├── samplers.py            # DialogueDistributedBatchSampler (chronological turn batching)
│   ├── context_cache.py       # Rolling dialogue history buffer
│   ├── augmentation.py        # SpecAugment and audio time-stretch / noise augmentations
│   └── synthetic.py           # Synthetic dialogue generator for unit testing
│
├── models/                    # Model architecture implementation
│   ├── conflictnet.py         # Top-level ConflictNet nn.Module
│   ├── experiment_config.py   # Dataclass configuration and CLI parser sync
│   ├── checkpoint_utils.py    # SafeTensors & PyTorch checkpoint state load/extract helpers
│   ├── audio/
│   │   └── encoder.py         # Emotion2Vec, WavLM, Wav2Vec2 multi-backend audio encoders
│   ├── text/
│   │   └── encoder.py         # DeBERTa-v3-large encoder with LoRA adaptation
│   ├── speaker_norm/
│   │   └── speaker_norm.py    # ECAPA-TDNN speaker projection + prosody z-score fusion
│   ├── fusion/
│   │   ├── gate.py            # GatedFusion, MoEFusion, and AdaptiveRouter
│   │   └── word_divergence.py # MFA / CTC cross-attention word-level divergence head
│   ├── temporal/
│   │   └── dialogue_transformer.py # Temporal Transformer over dialogue turns
│   └── classifier/
│       └── classifier.py      # Multi-label classifier, severity head, speaker-adaptive threshold
│
├── training/                  # Training loop & optimization
│   ├── trainer.py             # ConflictNetTrainer (train step, DDP broadcast, metric evaluation)
│   └── losses.py              # AutomaticWeightedLoss (homoscedastic multi-task uncertainty)
│
├── evaluation/                # Evaluation metrics, auditing, and explainability
│   ├── metrics.py             # compute_all_metrics, threshold tuning, WAcc, Macro/Weighted F1
│   ├── calibration.py         # Temperature scaling & Platt calibration
│   ├── fairness.py            # Demographic parity and equalized odds fairness audits
│   ├── attribution.py         # Integrated Gradients token & audio saliency attribution
│   ├── latency.py             # End-to-end inference latency profiler
│   ├── ood_probe.py           # Out-of-domain speaker generalization probe
│   └── human_eval.py          # Human correlation evaluation utilities
│
└── scripts/                   # CLI execution scripts
    ├── train.py               # Main training entry point (torchrun DDP / single-GPU)
    ├── evaluate.py            # Main evaluation CLI (with calibrated threshold support)
    ├── benchmark.py           # Benchmark comparison against baseline models
    ├── eval_ablation.py       # 5-way ablation runner (Audio-only, Text-only, Concat, Gated, MoE)
    ├── run_mfa_meld.py        # CTC Wav2Vec2 in-memory forced alignment for MELD
    ├── run_meld_kaggle.py     # Automated Kaggle runner script
    ├── compute_prosody_stats.py # Extract speaker baseline statistics (*.train_only.json)
    └── export_onnx.py         # ONNX model export for production deployment
```

---

## 7. Recommended Workflows for Agents

1. **Running Training on MELD**:
   ```bash
   python scripts/train.py \
       --meld_root /path/to/meld \
       --pt_dir /path/to/precomputed_features \
       --batch_size 16 \
       --epochs 35 \
       --lr 3e-5 \
       --audio_encoder wavlm_weighted \
       --output_dir checkpoints/
   ```
2. **Evaluating a Trained Checkpoint**:
   ```bash
   python scripts/evaluate.py \
       --checkpoint checkpoints/best_model.safetensors \
       --meld_root /path/to/meld \
       --output_dir results/
   ```
   *Note: `scripts/evaluate.py` will automatically read `best_binary_thresh` from `best_model_meta.json`.*

3. **Running Modality Ablations**:
   ```bash
   python scripts/eval_ablation.py \
       --checkpoint checkpoints/best_model.safetensors \
       --meld_root /path/to/meld
   ```

4. **Fast Prototyping / Debugging with Sample Capping**:
   ```bash
   python scripts/train.py \
       --meld_root /path/to/meld \
       --max_samples 500 \
       --batch_size 16 \
       --epochs 3
   ```
   *Note: `--max_samples` (aliases: `--max_sample_size`, `--max-samples`, `--max-sample-size`, `--meld_max_samples`) caps dataset size while preserving complete dialogues and stratified conflict distributions.*

