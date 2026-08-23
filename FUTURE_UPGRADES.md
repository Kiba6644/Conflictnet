# ConflictNet v2: Roadmap to 90%+ Accuracy & Macro-F1

This document outlines the technical strategy, architectural modifications, and data pipeline enhancements required to push **ConflictNet v2** beyond **90%+ accuracy and macro-F1** on multi-modal emotion and conflict benchmarks (IEMOCAP, MUStARD++, CREMA-D, MELD).

---

## 1. Backbone Upgrades & Capacity Scaling (High Impact)

### Audio Backbone
* **Current:** Base audio models (`emotion2vec_base`, `wav2vec2-base`).
* **Upgrade Target:** Upgrade to `emotion2vec_plus_large` or `microsoft/wavlm-large`.
* **Rationale:** Acoustic emotion conflict often relies on micro-prosodic cues (pitch contours, intensity spikes, jitter). Large pre-trained acoustic models capture these subtle variations far better than base models.

### Text Backbone & LoRA Optimization
* **Current:** `microsoft/deberta-v3-large` with LoRA rank \(r=8, \alpha=16\).
* **Upgrade Target:** Scale LoRA capacity to \(r=16\) or \(r=32\) with \(\alpha=32\).
* **Rationale:** In spoken sarcasm and conflict, textual semantics carry up to ~70% of the predictive signal. Higher LoRA ranks enable DeBERTa to fine-tune its deeper representation layers for conversational dynamics without full fine-tuning overhead.

---

## 2. Modality Dropout (ModalDrop) to Prevent Text Dominance

In cross-modal architecture, networks often suffer from **modality collapse**—where the model relies overwhelmingly on text tokens while ignoring acoustic prosody.

### Implementation Strategy
* Apply **Modality Dropout** during training:
  * Zero out text embeddings with probability \(p_{\text{text}} = 0.15\).
  * Zero out audio embeddings with probability \(p_{\text{audio}} = 0.15\).
* **Benefit:** Forces the downstream fusion gate and temporal transformer to build robust, standalone acoustic features rather than treating audio as a secondary signal.

---

## 3. Bi-directional Cross-Modal Attention Fusion

Replace or augment the current element-wise gated fusion with **Bi-directional Cross-Attention**:

\[
\text{Query}_{\text{text}} = \mathbf{H}_{\text{text}} \mathbf{W}_Q, \quad \text{Key}_{\text{audio}} = \mathbf{H}_{\text{audio}} \mathbf{W}_K, \quad \text{Value}_{\text{audio}} = \mathbf{H}_{\text{audio}} \mathbf{W}_V
\]

* **Mechanism:** Allows text word tokens to attend directly to aligned acoustic frames (e.g., matching a sarcastic tone to specific words like *"Oh, REALLY?"*).
* **Output:** Fused cross-modal vectors passed into the Transformer Temporal Context module.

---

## 4. Extended Dialogue Temporal Context

* **Current Window:** 4–8 dialogue turns.
* **Target Window:** Expand context window to **8–12 dialogue turns** with causal positional embeddings.
* **Rationale:** Conversational conflict and passive-aggressive suppression build up over multiple turns. Capturing longer conversational trajectories improves multi-turn sarcasm detection.

---

## 5. Loss Function & Focal Loss Hyperparameter Tuning

* **Focal Loss Scaling:** Increase Focal Loss \(\gamma\) from \(2.0 \to 2.5\) or \(3.0\) for `focal_bce_loss`:
  \[
  \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
  \]
  Down-weights easy non-conflict examples even further, forcing gradients to focus on hard, ambiguous conflict examples.
* **Self-Supervised Pre-Training:** Increase `--pretrain_epochs` from 5 to 10 epochs. Pre-training with the Context-Gated InfoNCE contrastive loss and Swap objective establishes tight cross-modal alignment before downstream fine-tuning.

---

## 6. Speaker ID & Prosody Normalization Integrity

* Ensure per-speaker ID extraction (`Ses01F` vs `Ses01M`) is strictly isolated across training splits so ECAPA-TDNN speaker z-scores operate on pure per-speaker moments.
* Use strict split-wise prosody normalization to eliminate any data leakage between train and validation sets.

---

## Summary Roadmap

| Feature / Upgrade | Current State | Target State (90%+ Accuracy) | Estimated Gain |
| :--- | :--- | :--- | :--- |
| **Audio Encoder** | Emotion2Vec Base / WavLM Base | `emotion2vec_plus_large` / `WavLM-Large` | +2.5–4.0% |
| **Text Encoder** | DeBERTa-v3 Large (\(r=8\)) | DeBERTa-v3 Large (\(r=16 / r=32\)) | +1.5–2.5% |
| **Fusion Mechanism** | Gated Element-Wise | Bi-directional Cross-Attention + Modality Dropout | +2.0–3.5% |
| **Pre-training Epochs** | 5 Epochs | 10 Epochs (Swap + InfoNCE) | +1.0–2.0% |
| **Temporal Window** | 4–8 Turns | 8–12 Turns | +1.0–1.5% |
