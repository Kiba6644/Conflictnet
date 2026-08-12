# ConflictNet Audit Findings

> Comprehensive code audit conducted on `main`.  
> All claims verified against source code at the line numbers shown.

---

## CRITICAL: Data Integrity Issues

### F1 — Speaker ID Extraction is Broken

- **File:** `data/datasets.py:260`
- **Code:** `"speaker_id": utt_id[:5]`
- **Problem:** IEMOCAP utterance IDs look like `Ses01F_impro01_F000`. `utt_id[:5]` extracts `"Ses01"` — the **session prefix** — not the speaker identity. IEMOCAP Session 1 has two speakers (`Ses01F` and `Ses01M`), both get `speaker_id="Ses01"`. This conflates distinct speakers, making speaker normalization, LOSO, and all speaker-dependent logic unreliable.
- **Impact:** Every experiment that depends on per-speaker separation operates on conflated groups. The speaker normalizer ECAPA-TDNN is computing statistics across 2 speakers per session.

### F2 — Severity Labels are Hardcoded Constants

- **File:** `data/datasets.py` (lines 259, 411, 496, 627)
- **Values:**
  | Dataset | Conflict | Non-conflict |
  |---------|----------|-------------|
  | IEMOCAP | 0.7 | 0.1 |
  | MUStARD++ | 0.8 | 0.1 |
  | CREMA-D | 0.5 | 0.1 |
  | MELD | 0.6 | 0.1 |
- **Problem:** No dataset provides real severity annotations. The MSE severity loss trains against meaningless constants with no per-sample variation within the conflict class.
- **Impact:** Reported severity MAE is not meaningful. The model is learning to predict dataset-specific constants.

### F3 — Conflict Type Labels are Fabricated

- **File:** `data/datasets.py:258`
- **Code:** `"conflict_type_labels": [0, int(conflict), 0]`
- **Problem:** All IEMOCAP/CREMA-D/MELD/CMU-MOSEI conflict is mapped to `[0, 1, 0]` (suppression). Only MUStARD++ has real sarcasm labels (`[1, 0, 0]`). The multi-label classifier learns a bias toward one class for most datasets.
- **Impact:** Per-type AP/AUC metrics are only meaningful for MUStARD++ and CASE datasets.

---

## CRITICAL: Data Leakage Paths

### L1 — Prosody Statistics Contaminated Across Splits

- **File:** `scripts/compute_prosody_stats.py`
- **Lines:** 87-100, 131-141, 103-114 (scanners), 209-260 (aggregation + z-score computation)
- **Problem:** All four dataset scanners collect audio from **every split**:
  - `scan_iemocap`: globs `Session*` — **includes Session 5** (used as validation)
  - `scan_meld`: iterates `("train", "dev", "test")` explicitly
  - `scan_mustard`: `rglob("*.wav")` on entire root — no split awareness
- At lines 241-260, per-utterance z-scores are computed using per-speaker statistics that include data from the opposite split. Every validation utterance's z-score `(f0 − μ) / σ` is normalized by moments that include that same utterance's data.

### L2 — Same Contaminated Z-Score Dict Used for Train and Val

- **File:** `scripts/train.py:114-115`
- **Code:**
  ```python
  train_collate = make_collate_fn(augmentor=augmentor, prosody_lookup=prosody_lookup)
  val_collate   = make_collate_fn(prosody_lookup=prosody_lookup)
  ```
- **Problem:** Both DataLoaders receive the same `prosody_lookup` dict (loaded from a `.zscores.json` file computed on all splits). During validation, the model sees z-scores normalized with statistics that include validation data. Information flows bidirectionally across splits.

### L3 — Context Cache Never Cleared Between Train and Eval

- **File:** `training/trainer.py`
- **Lines:** 96-99 (cache init), 156-163 (train reads), 187-189 (train writes), 255-258 (eval reads)
- **Problem:** `self.ctx_cache` persists across the entire training lifecycle. After `train_epoch()` populates the cache with conversation context, `evaluate()` immediately reads from the same cache. No `self.ctx_cache.clear()` call exists anywhere.
- **Latent:** Current datasets happen to have non-overlapping `conversation_id`s across splits (IEMOCAP sessions 1-4 vs 5 have different dialogues). MUStARD++, CREMA-D, CMU-MOSEI don't set `conversation_id` at all. The leak is architecturally dormant but one config change from activation.

### L4 — Dataset Splits Not Speaker-Stratified

- **File:** `data/datasets.py`
- **MUStARD++ (lines 360-361):** `all_keys[:n_train]` — splits by JSON key order (utterance ID), not by speaker. Same speaker can appear in both train and val.
- **CREMA-D (line 476):** `all_wavs[:n_train]` — splits by filename order. Safe by accident (sorts by actor ID, cutoff falls between actors) but not explicitly guaranteed.

### L5 — ColdStart Fallback Can Accumulate Cross-Split Data

- **File:** `models/speaker_norm/speaker_norm.py:275-281`
- **Problem:** `compute_prosody_z_scores()` unconditionally updates `_speaker_registry` and `cold_start` for every utterance it processes. Currently dormant (online path not activated) but documented as intended flow.

---

## HIGH: Correctness Bugs

### B1 — Benchmark Model Loading Uses Wrong Architecture

- **File:** `scripts/benchmark.py:172-175`
- **Code:**
  ```python
  model = ConflictNet(
      audio_encoder_name=ckpt.get("audio_encoder", "emotion2vec"),
      embed_dim=ckpt.get("embed_dim", 256),
  )
  ```
- **Problem:** `load_checkpoint_state()` for `.safetensors` returns a flat tensor dict via `safetensors.torch.load_file()`. No key `"audio_encoder"` exists. `ckpt.get("audio_encoder", "emotion2vec")` **always** returns the default `"emotion2vec"`. The actual config is in sidecar `_meta.json` but never read.
- **Same bug** in `serve/model.py:47-57`.

### B2 — Context Cache Zero-Fallback Missing Device

- **File:** `data/context_cache.py:75-76`
- **Code:**
  ```python
  embeds = torch.zeros(B, 1, embed_dim)                    # Missing device=
  padding = torch.ones(B, 1, dtype=torch.bool)             # Missing device=
  ```
- **Problem:** When `max_len == 0` (first turn in a conversation), tensors are created on CPU even when `self.device` is `"cuda"`. The non-fallback path (lines 80-81) correctly includes `device=self.device`. This causes a device mismatch at `trainer.py:161-162`.

### B3 — Context-Adaptive Temperature Averaged Across Batch

- **File:** `models/alignment/alignment.py:217`
- **Code:** `tau = (self.log_tau + delta_tau).exp().mean()`
- **Problem:** Per-sample `delta_tau` is computed but `.mean()` reduces to a scalar. Sample A's contrastive objective is modulated by sample B's dialogue context.

### B4 — Word Count Mismatch Silently Truncated

- **File:** `models/alignment/word_divergence.py:291-292`
- **Code:** `n = min(wa.size(0), wt.size(0))` then `wa[:n], wt[:n]`.
- **Problem:** When audio and text word embedding counts differ (MFA vs tokenization misalignment), the shorter list is silently truncated. No warning logged. Remaining words may be misaligned.

---

## MEDIUM: Code Quality Issues

| ID | File | Line | Issue |
|----|------|------|-------|
| Q1 | `models/encoders/text.py` | 22 | Tokenizer stored on model, unused in `forward()` — serialization hazard |
| Q2 | `models/encoders/audio.py` | 184-193 | Emotion2Vec fallback produces variable-length output vector |
| Q3 | `models/checkpoint_utils.py` | 73 | `getattr(torch, "load")` obfuscation; use `# nosec` + `weights_only=True` |
| Q4 | `scripts/run_mfa_alignment.py` | 152-181 | TextGrid fallback parser depends on fragile line ordering |
| Q5 | `evaluation/fairness.py` | 65 | FairLearn MetricFrame.by_group version-dependent format |
| Q6 | `evaluation/latency.py` | 40,46 | Population std (N) instead of sample std (N-1); off-by-one percentile |
| Q7 | `scripts/compute_difficulties.py` | 82-84 | Only supports .safetensors, no .pt fallback |
| Q8 | `scripts/evaluate.py` | 75-89, 107 | Evaluation uses self-contaminated z-scores |
| Q9 | `evaluation/ood_probe.py` | 147, 166, 217 | OOD probe uses leaky z-scores |
| Q10 | `serve/config.py` | 45 | Fragile `"str" in str(field_type)` type detection |

---

## Root Cause Analysis

| Issue | Root Cause | What It Invalidates |
|-------|-----------|-------------------|
| F1 (speaker_id) | `utt_id[:5]` truncates to session prefix | All speaker-dependent results |
| F2 (severity) | Hardcoded constants, not annotations | Severity MSE loss, reported MAE |
| F3 (type labels) | Hardcoded to suppression slot | Multi-label type predictions |
| L1+L2 (prosody leakage) | No split awareness in stats pipeline | Every quantitative metric on paper |
| L3 (context cache) | No clear between train/eval | Dialogue-dependent model results |
| B1 (benchmark) | Config not read from checkpoint | Benchmark results, serving predictions |

---

## Fix Priority

```
P0 — Fix foundations first
├── F1: Fix speaker_id extraction
├── L1+L2: Fix prosody stats pipeline (train-only statistics)
├── F2+F3: Fix fabricated labels

P1 — Fix correctness bugs
├── L3: Add ctx_cache.clear() before evaluate()
├── B1: Fix benchmark/serving model loading (read meta.json)
├── B2: Add device=self.device to zero-context fallback
├── L4: Add speaker-stratified dataset splitting

P2 — Model quality improvements
├── B3: Fix or document temperature averaging
├── B4: Add warning on word count mismatch
├── Q1-Q10: Code quality fixes
├── Expansion tasks from original plan
```
