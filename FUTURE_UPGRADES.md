# ConflictNet v3: Roadmap to 98% Accuracy & Macro-F1

This document is the living technical roadmap for pushing **ConflictNet** beyond **98% accuracy and macro-F1** across multi-modal conflict benchmarks (IEMOCAP, MUStARD++, CREMA-D, MELD, CMU-MOSEI, CASE 2026).

Upgrades are ordered by priority. **P0 fixes are not optional** — the audit revealed data leakage and label fabrication bugs that are silently capping every metric reported so far. No architecture improvement will exceed the ceiling set by corrupted training signals.

---

## ✅ P0 — Data Integrity & Leakage Fixes (COMPLETED)

> All bugs from `AUDIT_FINDINGS.md` have been patched. Listed here for historical record.

- ✅ **P0-A** Speaker ID extraction fixed (`utt_id[:6]` → `"Ses01F"`)
- ✅ **P0-B** Prosody z-scores now computed train-split-only (`prosody_stats.train_only.json`)
- ✅ **P0-C** `ctx_cache.clear()` added before `evaluate()` in `training/trainer.py`
- ✅ **P0-D** Conflict type labels corrected per emotion (anger→deception, frustrated→suppression; BCE masked for missing labels)
- ✅ **P0-E** Severity MSE loss masked for datasets without real annotations
- ✅ **P0-F** MUStARD++ and CREMA-D splits are now speaker-stratified



---

## 1. Backbone Upgrades & Capacity Scaling (High Impact)

### 1-A: Audio Backbone — Large Model Upgrade
- **Current:** `emotion2vec_plus_large` / `wavlm-large`
- **Next Tier:**
  - **Whisper Large v3** encoder-only — 1500M params, trained on 680K hours of multilingual speech. Captures fine-grained prosody that even emotion2vec misses.
  - **HuBERT X-Large** (`facebook/hubert-xlarge-ls960-ft`) — 1B param self-supervised audio model with richer frame-level representations for word-level divergence computation.
- **Integration:** Add `whisper_encoder` and `hubert_xlarge` backends to `models/encoders/audio.py`'s `build_audio_encoder()` factory.
- **Estimated Gain:** +2–3% on sarcasm (subtle tonal cues), +1–2% on suppression.

### 1-B: Text Backbone — Higher LoRA Capacity
- **Current:** DeBERTa-v3-Large, LoRA r=16, α=32
- **Upgrade:**
  - Scale to **r=32, α=64** and add `key_proj` + `output` projection as additional LoRA targets (currently only `query_proj` + `value_proj`).
  - Alternatively explore **ModernBERT-large** (2024): 8192 context, rotary position embeddings, Flash Attention — better for long multi-turn dialogue windows.
- **Estimated Gain:** +1.5–2.5%

### 1-C: Dual Audio Encoder Fusion
- Run **both** Emotion2Vec+ and WavLM-Large in parallel. Concatenate their projected embeddings before cross-modal attention rather than picking one.
- `audio_embed = concat(proj_emotion2vec(h_e2v), proj_wavlm(h_wavlm))` → dimension `256 + 256 = 512`, then projected back to `256` via a learned gate.
- **Rationale:** Emotion2Vec captures utterance-level emotional category; WavLM captures frame-level phonetic detail. Together they provide complementary acoustic signal.
- **Estimated Gain:** +2–3%

---

## 2. Cross-Modal Architecture Upgrades

### 2-A: Multi-Layer Bi-directional Cross-Modal Attention
- **Current:** Single-layer CrossModalAttention (4 heads).
- **Upgrade:** Stack **2–3 cross-modal attention layers**, each with residual connections. Also increase heads from 4 → 8.
- **Estimated Gain:** +1.5–2%

### 2-B: Modality Dropout (ModalDrop) to Prevent Text Dominance
- During training, zero out text embeddings with `p_text = 0.15` and audio embeddings with `p_audio = 0.15`.
- Prevents modality collapse where the model ignores acoustic prosody and relies almost entirely on textual features.
- **Implementation:** Add `ModalDrop` module in `models/alignment/alignment.py`, applied before CrossModalAttention.
- **Estimated Gain:** +1–1.5%

### 2-C: Token-Level (Not CLS-Level) Cross-Modal Attention
- **Current:** CLS token pooled embedding per modality used as single query/key.
- **Upgrade:** Pass full sequence of DeBERTa token embeddings `(B, L, 1024)` → projected to `(B, L, 256)` and full audio frame embeddings as key/value sequences. Each text token attends to the most acoustically relevant audio frame.
- **Why this matters for 98%:** Sarcasm is often carried by a specific word ("Oh, REALLY?") whose acoustic embedding diverges maximally from the text embedding. Token-level attention surfaces this per-word signal automatically.
- **Estimated Gain:** +2–4% (biggest single architectural upgrade)

### 2-D: Gated Multi-Modal Mixture-of-Experts Fusion
- Replace the current linear Fusion Gate with a **MoE fusion layer**:
  - 4 expert MLPs (256 → 256), each specialising in a different conflict subtype signature.
  - A gating network selects/weights experts based on `speaker_feat` and `word_div` features.
- **Estimated Gain:** +1–2%

---

## 3. Temporal Context Upgrades

### 3-A: Extended Dialogue Window with Hierarchical Compression
- **Current:** 4–8 turns (max 16 configured).
- **Target:** 8–16 turns active window with hierarchical compression for older turns (older turns summarised into a single "history vector" via average pooling).
- **Estimated Gain:** +1–1.5% on multi-turn sarcasm and passive-aggressive suppression.

### 3-B: Turn-Level Contrastive Learning
- Add an auxiliary loss that pulls together embeddings of emotionally consistent turns and pushes apart turns where emotional state flips.
- **Rationale:** Suppression often presents as a sudden prosodic drop after conflict escalation. The temporal model should learn this trajectory.
- **Estimated Gain:** +0.5–1%

### 3-C: Relative Position Encoding (RoPE / ALiBi) in Temporal Transformer
- Replace learned absolute positional embeddings with **RoPE** or **ALiBi** for the Temporal Transformer.
- Allows generalisation to unseen context window lengths at inference time.
- **Estimated Gain:** Marginal accuracy, but significantly improves OOD long-form conversation generalisation.

---

## 4. Word-Level Divergence Upgrades

### 4-A: Extended Word Divergence Feature Vector
- **Current:** 8-dim aggregate vector (max, mean, std, ratio, top3 positions, word count).
- **Add features:**
  - `skewness(d(w))` — asymmetry of divergence distribution (high skew = one explosive word)
  - `entropy(d(w))` — uniform vs. concentrated divergence
  - `turn_position_of_max` — is the peak divergence at utterance start/middle/end?
  - Per-phoneme divergence variance (requires phoneme-level MFA alignment)
- **Target:** Expand to 12–16 dim vector.
- **Estimated Gain:** +0.5–1%

### 4-B: Soft CTC Alignment Replacing Hard MFA
- **Current:** Hard MFA `.TextGrid` forced alignment (offline, brittle, ~15% failure rate on noisy audio).
- **Upgrade:** Replace with **Wav2Vec2 CTC forced alignment** via `torchaudio.functional.forced_align()` (torchaudio 2.1+). Online, no preprocessing pipeline required, robust to disfluencies.
- **Estimated Gain:** +1–2% on datasets where MFA alignment fails silently.

### 4-C: Prosody-Text Co-embedding Divergence
- Train a small GPT-2-scale transformer on `(text → prosody sequence)` prediction.
- The PLM's predicted prosody serves as "expected prosody". The residual between PLM prediction and actual prosody is the conflict signal — directly formalising the ConflictNet hypothesis.
- **Estimated Gain:** +1.5–2.5% (most principled formalisation of the sarcasm/deception signal).

---

## 5. Loss Function & Training Upgrades

### 5-A: Focal Loss for Class Imbalance
- Apply **Focal BCE** (γ = 2.5–3.0) to the conflict type multi-label loss:
  - Down-weights easy non-conflict examples, forces gradient to focus on hard ambiguous boundaries.
- **Estimated Gain:** +0.5–1% on minority class (deception is least represented).

### 5-B: Extended Self-Supervised Pre-Training + Masked Audio Modelling
- **Current:** 5 pretrain epochs (swap objective only).
- **Target:** 10–15 pretrain epochs + **Masked Audio Modelling (MAM)**: randomly mask 15% of audio frames; predict masked frame embeddings from context + text. Forces deeper acoustic-textual alignment.
- **Estimated Gain:** +1–2%

### 5-C: Label Smoothing + Multimodal Mixup
- **Label Smoothing:** Apply `ε = 0.1` to BCE labels. Prevents overconfident outputs; improves calibration.
- **Multimodal Mixup:** Interpolate audio + text embeddings between two same-label training samples (`λ ~ Beta(0.4, 0.4)`).
- **Estimated Gain:** +0.5–1%

### 5-D: Contrastive Prototype Learning
- Maintain a **prototype bank** (running mean of per-class embeddings) in the embedding space.
- Add auxiliary prototype loss: push each sample's fused embedding toward its class prototype and away from opposing prototypes.
- **Estimated Gain:** +0.5–1%

---

## 6. Data & Augmentation Pipeline Upgrades

### 6-A: Integrate Synthetic Conflict Data (StarGANv2-VC)
- `scripts/generate_synthetic.py` exists but is not wired into the training pipeline.
- **Action:** Integrate StarGANv2-VC-generated samples as a dedicated synthetic training split with 20% weight in curriculum sampler. Target +5,000–10,000 synthetic utterances per conflict subtype.
- **Estimated Gain:** +1.5–2.5% (especially deception, which is severely underrepresented).

### 6-B: Cross-Dataset Transfer Pre-Training
- Pre-train on all datasets simultaneously before fine-tuning on the target benchmark. Use **dataset-specific task heads** during pre-training, then discard heads and fine-tune shared backbone.
- **Estimated Gain:** +1–2%

### 6-C: Test-Time Augmentation (TTA)
- At inference, run each utterance through 3–5 augmented variants (pitch shift, speed jitter, SpecAugment) and average predicted probabilities.
- **No training change required.**
- **Estimated Gain:** +0.5–1%

### 6-D: Advanced Audio Augmentation
- **Current:** Speed, noise, SpecAugment.
- **Add:** Pitch shifting (±2 semitones), room impulse response (RIR) convolution, telephone codec simulation (G.711), background babble from MUSAN.
- **Estimated Gain:** +0.5–1% on OOD generalisation.

---

## 7. Speaker Normalisation Upgrades

### 7-A: Split-Aware Prosody Normalisation (See P0-B)
- After fixing the pipeline, verify that cold-start fallback statistics are computed **train-split-only** before being used at validation time.

### 7-B: Gender-Conditional Prosody Normalisation
- Always apply a **gender-conditional prior** as an additive bias to z-scores (female speakers have naturally higher F₀ — same z-score means different things).
- Learn gender-conditioned mean offsets as trainable parameters.
- **Estimated Gain:** +0.5–1% (fairness-accuracy double win).

### 7-C: Emotional State Trajectory Normalisation
- Track not just neutral baseline (current EMA) but also **per-speaker angry baseline** and **per-speaker suppressed baseline**.
- Conflict is relative to the speaker's own emotional range, not just their neutral centroid.
- **Estimated Gain:** +0.5–1%

---

## 8. Calibration & Threshold Optimisation

### 8-A: Temperature Scaling Calibration
- After training, fit a single scalar temperature `T` via NLL minimisation on the val set to calibrate output probabilities.
- **Estimated Gain:** +0.3–0.7%

### 8-B: Per-Class Threshold Tuning from Calibration Sweep
- Feed the optimal per-type thresholds from `evaluation/calibration.py` back into inference as defaults. Store in checkpoint `_meta.json` and load in `serve/model.py`.
- **Estimated Gain:** +0.5–1%

### 8-C: Platt Scaling / Isotonic Regression
- Apply **Platt scaling** (logistic regression on logits) per conflict type as a calibration post-processing step.
- **Estimated Gain:** +0.3–0.5%

---

## 9. Training Infrastructure & Regularisation

### 9-A: Stochastic Depth (DropPath)
- Apply **DropPath** to Temporal Transformer and Cross-Modal Attention layers during training (survival rate = 0.9).
- Reduces overfitting on small IEMOCAP (10K utterances).

### 9-B: Exponential Moving Average (EMA) of Model Weights
- Maintain EMA copy of model weights (decay = 0.9999 at batch level). Use EMA model at eval/inference.
- **Estimated Gain:** +0.5–1%

### 9-C: Layer-wise Learning Rate Decay (LLRD)
- Apply different LRs per layer group:
  - Frozen audio encoder: `lr = 0`
  - DeBERTa lower layers: `lr = 1e-5`
  - DeBERTa LoRA adapters: `lr = 2e-5`
  - Projection heads, fusion, temporal: `lr = 5e-5`
  - Classifier head: `lr = 1e-4`
- Prevents catastrophic forgetting while allowing aggressive fine-tuning of task-specific modules.
- **Estimated Gain:** +0.5–1%

### 9-D: Gradient Checkpointing for Larger Batch Size
- Enable `model.gradient_checkpointing_enable()` on DeBERTa to allow batch sizes of 64–128. Larger batches improve InfoNCE quality (more in-batch negatives).
- **Estimated Gain:** +0.5–1%

---

## 10. Novel Research-Grade Upgrades

### 10-A: Knowledge Distillation from Audio-LLM Teacher
- Use **WavLLM** or **Emotion-LLaMA** as a frozen teacher model. Train ConflictNet (student) to match the teacher's soft conflict logits via KD loss.
- **Estimated Gain:** +2–3% (KD from larger model is consistently one of the highest-leverage techniques)

### 10-B: Retrieval-Augmented Conflict Detection
- Build a FAISS index of training embeddings post-training. At inference, retrieve top-K most similar utterances and augment the classifier with their labels as soft pseudo-labels.
- **Estimated Gain:** +0.5–1.5% especially on rare conflict subtypes (deception is data-scarce).

### 10-C: Prosody Language Model (PLM) Auxiliary Task
- Train a small GPT-2-scale transformer on `(text → prosody sequence)` prediction as an auxiliary task during pre-training.
- The PLM's predicted prosody serves as "expected prosody". The residual = conflict signal.
- **Estimated Gain:** +1.5–2.5% (fundamentally more principled than the current word-level divergence heuristic).

### 10-D: Graph Neural Network for Multi-Party Conversation Modelling
- For datasets with 3+ speakers (MELD, CASE 2026), model inter-speaker influence with a **GNN**:
  - Nodes = speaker turns, Edges = temporal adjacency + same-speaker connections
  - Node features = fused ConflictNet embeddings
  - GNN propagates conflict signals across speakers (conflict is often reactive/contagious)
- **Estimated Gain:** +1–2% on multi-party datasets.

### 10-E: Uncertainty-Aware Inference with Monte Carlo Dropout
- At inference, enable dropout and run N=20 forward passes. Report mean prediction + variance.
- High variance → flag for human review in production serving.
- **No training change required.**

---

## 11. Evaluation & Fairness Upgrades

### 11-A: Fix FairLearn API Compatibility (Q5)
- Update `evaluation/fairness.py` for FairLearn ≥ 0.9. Extend auditing beyond gender → also audit by **age group**, **dialect/accent**, and **emotion intensity quartile**.

### 11-B: Cross-Dataset Zero-Shot Generalisation Benchmarking
- Implement formal cross-dataset transfer evaluation:
  - Train on IEMOCAP → evaluate on MUStARD++ (zero-shot).
  - Train on MELD → evaluate on IEMOCAP (zero-shot).
- No formal cross-dataset zero-shot evaluation currently exists in the codebase.

### 11-C: Reliability Diagrams (Confidence-Accuracy Curves)
- Extend `evaluation/calibration.py` to generate reliability diagrams per conflict subtype. Directly expose overconfidence in high-accuracy models trained on small data.

---

## Summary Roadmap to 98%

| Priority | Feature / Upgrade | Current State | Target State | Est. Gain |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | Speaker ID Fix (F1) | `utt_id[:5]` → `"Ses01"` | `utt_id[:6]` → `"Ses01F"` | Foundational |
| **P0** | Prosody Leakage Fix (L1+L2) | Train/val stats contaminated | Train-only stats | Foundational |
| **P0** | Context Cache Clear (L3) | Cache persists across epochs | `clear()` before eval | Foundational |
| **P0** | Conflict Type Labels (F3) | All mapped to suppression | Proper per-emotion mapping | Foundational |
| **P0** | Severity Labels (F2) | Hardcoded constants | Masked or acoustic proxy | Foundational |
| **1** | Token-Level Cross-Modal Attn (2-C) | CLS-only query | Full sequence Q/K/V | +2–4% |
| **1** | Dual Audio Encoder (1-C) | Single encoder | Emotion2Vec + WavLM concat | +2–3% |
| **1** | KD from WavLLM Teacher (10-A) | No teacher | Audio-LLM soft targets | +2–3% |
| **1** | Whisper Encoder Backend (1-A) | — | Whisper Large v3 option | +2–3% |
| **2** | Synthetic Data Integration (6-A) | Scripted but unused | +10K synthetic utterances | +1.5–2.5% |
| **2** | Prosody LM Aux Task (10-C / 4-C) | Word divergence heuristic | PLM-predicted residual | +1.5–2.5% |
| **2** | CTC Soft Alignment (4-B) | Hard MFA TextGrid | `torchaudio.forced_align()` | +1–2% |
| **2** | Multi-Layer Cross-Attn (2-A) | 1 layer, 4 heads | 3 layers, 8 heads | +1.5–2% |
| **2** | LoRA Capacity Scaling (1-B) | r=16, α=32 | r=32, α=64, +key+output | +1.5–2.5% |
| **2** | Extended Pre-Training + MAM (5-B) | 5 epochs, swap only | 15 epochs + MAM | +1–2% |
| **3** | EMA Model Weights (9-B) | No EMA | EMA decay=0.9999 | +0.5–1% |
| **3** | Per-Class Threshold Tuning (8-B) | Fixed 0.5 | Val-set calibrated per type | +0.5–1% |
| **3** | LLRD (9-C) | Uniform LR | Layer-wise decay | +0.5–1% |
| **3** | Gradient Checkpointing (9-D) | batch=32 | batch=128, more negatives | +0.5–1% |
| **3** | ModalDrop (2-B) | No dropout | p=0.15 per modality | +1–1.5% |
| **3** | Retrieval Augmentation (10-B) | No retrieval | FAISS top-K neighbour labels | +0.5–1.5% |
| **3** | Test-Time Augmentation (6-C) | Single pass | 5× augmented average | +0.5–1% |
| **3** | Focal Loss γ=2.5–3.0 (5-A) | BCE, γ=2.0 | Focal BCE, γ=2.5–3.0 | +0.5–1% |
| **3** | GNN Multi-Party Modelling (10-D) | Independent utterances | GNN over speaker graph | +1–2% |
| **4** | Emotional State Trajectory Norm (7-C) | Neutral baseline only | Angry + suppressed baselines | +0.5–1% |
| **4** | Stochastic Depth (9-A) | No DropPath | DropPath, survival=0.9 | Minor |
| **4** | Label Smoothing + Mixup (5-C) | Hard labels | ε=0.1 smooth + multimodal mixup | +0.5–1% |

**Cumulative Estimated Gain (P0 fixes + all Priority 1–3 improvements):** +18–35% over the leakage-affected baseline → targeting **98%+** on a clean evaluation.

---

## Implementation Order

```
Phase 1 (Weeks 1–2): Fix the foundations
  └── P0-A through P0-F (data integrity + leakage)

Phase 2 (Weeks 3–4): High-impact architecture
  └── Token-level cross-modal attention (2-C)
  └── Dual audio encoder (1-C)
  └── Multi-layer cross-attn (2-A) + ModalDrop (2-B)
  └── LoRA r=32 + key+output targets (1-B)

Phase 3 (Weeks 5–6): Training improvements
  └── Extended pre-training + MAM (5-B)
  └── EMA weights (9-B) + LLRD (9-C) + grad checkpointing (9-D)
  └── Synthetic data integration (6-A)
  └── Focal loss + label smoothing (5-A, 5-C)

Phase 4 (Weeks 7–8): Novel research features
  └── Prosody LM auxiliary task (10-C / 4-C)
  └── CTC soft alignment replacing MFA (4-B)
  └── Knowledge distillation from WavLLM (10-A)
  └── GNN for multi-party conversations (10-D)

Phase 5 (Week 9+): Calibration + retrieval
  └── Threshold calibration from val set (8-A–8-C)
  └── FAISS retrieval augmentation (10-B)
  └── Test-time augmentation (6-C)
```
