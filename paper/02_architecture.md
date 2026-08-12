# Model Architecture

ConflictNet processes a paired audio waveform and text transcript through a shared embedding space, applies speaker-invariant normalization and cross-modal alignment, contextualizes within dialogue history, and produces multi-label conflict predictions with severity estimates. Figure 1 illustrates the full pipeline.

## 3.1 Encoders and Projection

**Audio encoder.** We use Emotion2Vec+ (iic/emotion2vec_plus_large) as the primary audio backbone, producing 768-dimensional utterance-level embeddings from 16 kHz waveforms (max 10 seconds). Emotion2Vec+ is chosen over WavLM and wav2vec2 because its self-supervised pre-training on emotional speech corpora yields representations that are more sensitive to prosodic nuance (Peng et al., 2024). The encoder is frozen during training to prevent catastrophic forgetting.

**Text encoder.** We use DeBERTa-v3-large (He et al., 2023) with LoRA fine-tuning (Hu et al., 2022) at rank r=16, alpha=32, targeting query_proj and value_proj. The [CLS] token from the final hidden layer provides a 1024-dimensional text representation. LoRA reduces trainable text parameters from 390M to ~0.5M (0.13%), mitigating overfitting on moderate-sized conflict datasets. We retain DeBERTa-v3-large rather than a larger LLM (Mistral-7B, Phi-4) due to compute constraints; the 0.5M trainable adapter parameters are sufficient for the paper's scope.

**Projection heads.** Both encoder outputs are projected to a shared 256-dimensional space via identical MLPs: Linear(encoder_dim → 512) + GELU + LayerNorm + Linear(512 → 256) + LayerNorm. This matches the HuBERT-CLAP projection architecture (Liang et al., 2024).

## 3.2 Speaker-Invariant Normalization

Speaker identity is the primary confound in prosody-based conflict detection. We address this with a three-component speaker normalization module.

**ECAPA-TDNN speaker embedding.** A frozen speechbrain ECAPA-TDNN (Desplanques et al., 2020) extracts a 192-dimensional speaker embedding from the raw waveform. This embedding captures speaker identity without task-specific fine-tuning.

**Prosody features.** We extract three prosody statistics per utterance using librosa (fallback from parselmouth): mean fundamental frequency (F0), mean energy (RMSE), and speaking rate (syllables/second estimated via energy envelope peak counting). These are computed offline and provided as a 3-dimensional z-score vector (or baseline-subtracted deviation).

**Cold-start fallback hierarchy.** For each speaker, we maintain online Welford statistics (mean, variance, count) for the prosody features. For speakers with ≥5 reference utterances, we use their own statistics for normalization. Below this threshold, we fall back to gender-group statistics, then to the nearest VoxCeleb k-means cluster centroid (k=20), and finally to global corpus statistics. This ensures robust normalization even for speakers seen only once.

**Baseline-subtract mode (default).** Instead of standard z-score normalization (deviation from speaker mean), we track an exponential moving average (EMA) of neutral-speaking prosody. For IEMOCAP, neutral utterances are those labeled "neu"; for other datasets, they are non-conflict utterances. The EMA update for a non-conflict utterance is:

```
μ_neutral ← (1 - η) · μ_neutral + η · prosody_features
```

with learning rate η = 0.2. The deviation features are then δ = prosody - μ_neutral. This measures *how far the speaker is from their neutral baseline* rather than from their average, which is more interpretable and empirically more robust. An ablation study (§5) confirms that baseline-subtract outperforms standard z-score.

The speaker embedding (192-d) and prosody deviation (3-d) are concatenated and projected to 256-d via Linear(195 → 512) + GELU + LayerNorm + Linear(512 → 256) + LayerNorm.

## 3.3 Cross-Modal Attention

Prior work injects dialogue context into audio and text embeddings *independently* (each modality attending to context without cross-modal interaction). We replace this with **direct cross-modal attention**: each modality attends to the other's embedding, with optional dialogue context appended to the key/value sequence.

Concretely, let `a ∈ ℝ^D` and `t ∈ ℝ^D` be the projected audio and text embeddings for the current utterance, and let `C ∈ ℝ^{T×D}` be the sequence of T prior-turn fused embeddings (when operating in dialogue mode). We construct:

```
K/V_audio = [t, C]    (text K/V: current text + optional context)
K/V_text  = [a, C]    (audio K/V: current audio + optional context)
```

Two independent multi-head attention layers (4 heads, D=256) then perform:

```
a' = a + LayerNorm(MultiHeadAttn(Q=a, K=K/V_audio, V=K/V_audio))
t' = t + LayerNorm(MultiHeadAttn(Q=t, K=K/V_text,  V=K/V_text))
```

Using two independent layers allows each modality to learn modality-specific cross-modal patterns. The context_seq parameter is optional — when C is None, the module acts as pure cross-modal alignment without dialogue history, enabling single-utterance inference. A cold-start guard inserts a neutral zero vector as an unmasked K/V entry when all context positions are padding, preventing attention collapse on the first dialogue turn.

## 3.4 Fusion and Temporal Context

**Fusion gate.** The normalized audio embedding a', text embedding t', and speaker feature s are concatenated and passed through a gated MLP:

```
f = MLP_gate([a', t', s])  where MLP_gate: ℝ^{3D} → ℝ^D
```

The gate has structure Linear(3D → 2D) + GELU + LayerNorm + Linear(2D → D) + LayerNorm.

**Temporal context module.** When dialogue history is available (up to 8 turns), a causal Transformer encoder (2 layers, 4 heads, feed-forward dimension 1024, Pre-LayerNorm) contextualizes the current fused embedding within the turn sequence. Learned positional encodings (not sinusoidal) capture turn order, and learned speaker role embeddings (SPK_A=0, SPK_B=1) capture inter-speaker dynamics. Causal masking prevents future information leakage. The output at the last position serves as the context-pooled representation for classification.

## 3.5 Optional Word-Level Divergence

When forced alignment via Montreal Forced Aligner (McAuliffe et al., 2017) is available, we compute word-level divergence features: per-word cosine similarity between audio frame embeddings and text token embeddings in the shared 256-d space, aggregated to 8 statistics (mean, std, min, max, skew, kurtosis, rate of divergence change, and the maximum single-word divergence). These features provide a fine-grained signal when accurate word boundaries are available. When MFA is not available, the classifier operates on fused embeddings alone.

## 3.6 Conflict Classifier

The classifier receives the context-pooled fused embedding (optionally concatenated with word-level divergence features, total dimension 264) and produces three outputs through a shared MLP (264 → 512 → 256):

1. **Multi-label conflict types.** Three independent sigmoid units predict sarcasm, suppression, and deception (not mutually exclusive).

2. **Severity regression.** A Linear(256 → 1) + Sigmoid head predicts continuous severity in [0, 1].

3. **Speaker-adaptive conflict flag.** A binary flag is derived by thresholding the maximum subtype probability. Critically, the threshold is per-sample: a small MLP (256 → 64 → 1, Tanh, sigmoid × 0.3) predicts an offset in [0, 0.3] from the speaker representation s. The effective threshold is 0.5 + offset, so expressive speakers receive higher thresholds (fewer false positives) and reserved speakers receive lower thresholds (fewer false negatives). This is the first application of speaker-adaptive thresholds to conflict detection to our knowledge.

## 3.7 Training Objectives

**Context-gated contrastive loss.** We adopt an InfoNCE loss (Oord et al., 2018) with a context-adaptive temperature τ. A small MLP (256 → 64 → 1) takes the pooled dialogue context and predicts Δlog τ, modulating the temperature per-batch:

```
τ_eff = exp(log τ_base + Δlog τ_mean)
```

This allows the model to soften the contrastive objective when the dialogue context is inherently ambiguous (e.g., extended sarcastic exchanges) and sharpen it for clearly sincere utterances.

**Conflict separation loss.** For pairs labeled as conflict (any subtype present), we add a hinge penalty:

```
L_sep = max(0, sim(a, t) + margin)  where margin = 0.5
```

This explicitly pushes audio and text embeddings apart for conflict pairs — the core inductive bias that conflict = audio-text divergence. For non-conflict pairs, only the standard InfoNCE loss applies, pulling aligned pairs together.

**Multi-task uncertainty weighting.** Following Kendall et al. (2018), we learn log σ² per task and combine losses as:

```
L_total = Σ_i (exp(-log σ²_i) · L_i + log σ_i)
```

This eliminates manual loss weighting. During pre-training, four tasks are balanced: contrastive, type BCE, severity MSE, and swap detection. During fine-tuning, the swap objective is removed (three tasks).

## 3.8 Self-Supervised Pre-Training

Before supervised fine-tuning, we pre-train for 5 epochs using only the swap detection objective. For each batch, 30% of samples are randomly chosen as "swap" examples; of these, 50% have their audio swapped with another sample (audio from utterance A, text from utterance B) and 50% have their text swapped symmetrically. A binary classifier (Linear(2D → 1)) predicts swapped (1) vs. matched (0). Using both swap types prevents the model from learning a trivial text-only shortcut. This pre-training forces the model to learn cross-modal alignment without any conflict labels, enabling training on unlabeled audio-text corpora.
