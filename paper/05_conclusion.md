# Conclusion

> **Status:** Placeholder — populate after results are available.

---

## Summary

We presented **ConflictNet**, a speaker-normalised cross-modal model for emotional conflict detection in dialogue. Key contributions:

1. **Speaker normalisation** via ECAPA-TDNN + prosody z-score disentangles speaker identity from conflict expression, reducing the speaker confound identified in prior work.
2. **Context-gated contrastive learning** learns per-sample temperature modulation from dialogue context, improving alignment of ambiguous utterances.
3. **Temporal context** from preceding dialogue turns improves detection accuracy by modeling conversational dynamics (ongoing).
4. **Multi-label + severity output** provides fine-grained conflict characterisation beyond binary detection.

---

## Limitations

- Severity labels are not available in most datasets; we use a binary proxy (0.0 / 1.0), which collapses severity regression to a classification-like signal.
- Word-level divergence features require Montreal Forced Aligner, adding preprocessing overhead.
- The emotion-to-conflict mapping (anger/frustration → suppression, sarcasm → sarcasm) is a research definition — real conflict types may be more nuanced.
- Real-world evaluation on in-the-wild data is needed.

---

## Future Work

- **Richer severity annotations:** Collect or crowdsource fine-grained severity labels on existing corpora.
- **Cross-lingual conflict detection:** Adapt audio encoder to multilingual speech representations (e.g., Whisper, MMS).
- **End-to-end streaming:** Integrate temporal context into a streaming pipeline for real-time dialogue monitoring.
- **Causal intervention:** Use the speaker normalisation module for counterfactual analysis (e.g., "would this utterance be conflictual if spoken by a different speaker?").
