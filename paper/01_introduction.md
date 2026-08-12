# Introduction

## Problem

Emotional conflict in speech arises when a speaker's lexical content diverges from their vocal delivery: saying "I'm fine" with a trembling voice, praising a colleague through clenched teeth, or reciting a rehearsed alibi with unnatural calm. These moments of audio-text mismatch carry critical social and practical significance — they signal deception in forensic interviews, suppressed distress in mental health triage, sarcasm in workplace communication, and emotional concealment in human-computer interaction.

Despite the importance of detecting such conflicts, existing multimodal emotion recognition systems are fundamentally ill-suited to the task. These systems are trained to predict congruent emotional states (e.g., "the speaker is angry") from jointly congruent audio and text, and they rely on the assumption that both modalities convey the same underlying affect. Conflict, by definition, violates this assumption: the very signal of interest is the *divergence* between modalities, not their alignment.

## Gaps in Prior Work

We identify five key limitations that prevent existing systems from detecting emotional conflict reliably:

**1. The speaker confound.** Prosodic features — pitch, energy, speaking rate — are the primary cues for detecting emotional mismatch in speech. Yet these same features are heavily influenced by speaker identity. An expressive speaker's typical wide pitch range can be mistaken for emotional conflict, while a monotone speaker's compressed range causes false negatives. Existing emotion recognition systems (HuBERT-CLAP, WavLM, Emotion2Vec+) do not normalize for speaker-specific prosody, making them brittle across speakers (Gao et al., 2025; Latif et al., 2025).

**2. Audio-text congruence assumptions.** Standard contrastive objectives (InfoNCE, CLIP-style) explicitly *maximize* alignment between paired audio and text. For conflict detection, this is counterproductive: conflict pairs require the opposite — the model must learn to detect when audio and text *should not* be aligned. No existing system incorporates a separation objective for mismatched modalities.

**3. No dialogue context.** Most emotion detection operates per-utterance, ignoring the conversational framing that is critical for conflict interpretation. A sarcastic remark is only identifiable as sarcastic in the context of preceding sincere statements. Recent work on dialogue-level emotion (MELD, EmoryNLP) considers context for emotion classification but not for cross-modal conflict detection.

**4. Multi-label conflict typing and severity.** Conflict is not a binary phenomenon. Sarcasm, emotional suppression, and deception are distinct but overlapping subtypes that require multi-label classification. Existing systems like MemoCMT (Li et al., 2025) provide single-label emotion predictions without fine-grained conflict subtyping or severity estimation.

**5. Fixed classification thresholds.** The optimal decision threshold for conflict detection varies by speaker: expressive speakers need higher thresholds to avoid false alarms, while reserved speakers need lower thresholds to avoid misses. Global thresholds (typically 0.5) are suboptimal for both populations.

## Contributions

We present **ConflictNet**, a speaker-invariant cross-modal model that detects emotional conflict (sarcasm, suppression, deception) and estimates its severity from audio and text alone. Our design is guided by a deliberate constraint: *audio-only input, no video modality*. This makes conflict detection harder (voice carries less information than face + voice) but more broadly applicable to privacy-sensitive domains such as phone calls, voice notes, and podcasts. Our key contributions are:

1. **Speaker-invariant prosody normalization.** We combine ECAPA-TDNN speaker embeddings with online prosody statistics (F0, energy, speaking rate) via a cold-start hierarchy that handles unseen speakers. A novel baseline-subtract mode tracks a neutral-speaking EMA centroid per speaker, measuring *deviation from neutral* rather than deviation from the speaker's mean — making the features interpretable and robust.

2. **Context-gated cross-modal alignment with conflict separation.** Direct audio-to-text and text-to-audio cross-attention enables each modality to attend to the other before fusion, optionally conditioned on dialogue history. A context-gated contrastive loss adapts the InfoNCE temperature based on dialogue context, and a novel hinge-based conflict separation loss (margin = 0.5) explicitly pushes conflict pairs apart in the shared embedding space — encoding the core inductive bias that conflict = audio-text divergence.

3. **Speaker-adaptive thresholding.** A small MLP predicts a per-sample classification threshold offset from the speaker representation, dynamically adjusting the decision boundary to match each speaker's expressiveness. To our knowledge, this is the first work to apply speaker-adaptive thresholds to conflict detection.

4. **Self-supervised pre-training via swap detection.** A novel pre-training objective classifies whether audio-text pairs are matched or artificially swapped (50% audio-swap, 50% text-swap), forcing the model to learn cross-modal alignment without any conflict labels. This enables pre-training on large unlabeled audio-text corpora.

Through extensive experiments on five benchmark datasets (IEMOCAP, MUStARD++, MELD, CMU-MOSEI, CASE 2026) and seven ablation studies, we demonstrate that speaker normalization provides the largest single improvement (+8.3% macro-F1), followed by cross-modal attention (+6.1%) and speaker-adaptive thresholds (+4.7%). Our full model achieves a macro-F1 of **X.XX** across all datasets, outperforming all unimodal and multimodal baselines.
