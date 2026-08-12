# Results

> **Status:** Placeholder — to be populated after training runs with the corrected codebase.
> Once the audit fixes (P0–P2) are applied, re-run all experiments and fill the tables below.

---

## Main Results

Conflict detection performance on held-out test sets (weighted F1, macro-F1, and severity MAE).

| Model | IEMOCAP (WAcc) | IEMOCAP (F1) | MUStARD++ (F1) | CREMA-D (F1) | Severity MAE |
|-------|----------------|--------------|----------------|---------------|--------------|
| HuBERT-CLAP (baseline) | — | — | — | — | — |
| ConflictNet (ours) | — | — | — | — | — |
| _vs. prior SOTA_ | — | — | — | — | — |

> **Dataset definitions:**
> - IEMOCAP: session-level leave-one-session-out (4 train, 1 test).
> - MUStARD++: speaker-stratified 80/20 split.
> - CREMA-D: speaker-stratified 80/20 split (actors not shared across splits).

---

## Ablation Study

Effect of each component on weighted F1 (IEMOCAP held-out test).

| Variant | F1 | \(\Delta\) |
|---------|-----|------------|
| Full ConflictNet | — | — |
| – Speaker normalisation | — | — |
| – Temporal context | — | — |
| – Word-level divergence | — | — |
| – Cross-modal attention | — | — |
| – Context-gated contrastive | — | — |
| – Speaker-adaptive threshold | — | — |
| – Baseline subtract | — | — |

---

## Fairness Audit

Demographic parity and equalised-odds differences across gender groups (IEMOCAP).

| Metric | Male | Female | Max gap |
|--------|------|--------|---------|
| F1 score | — | — | — |
| Selection rate | — | — | — |
| DP diff | — | — | — |
| EO diff | — | — | — |

---

## Latency

| Batch size | Avg (ms) | Std (ms) |
|------------|----------|----------|
| 1 | — | — |
| 8 | — | — |
| 32 | — | — |

Hardware: —

---

## LLM Baseline Comparison (MUStARD++)

| Model | F1 | Notes |
|-------|-----|-------|
| GPT-4o (text-only) | — | Zero-shot, prompt from §3 |
| ConflictNet (audio+text) | — | — |

---

## Sample Efficiency

| % training data | F1 |
|----------------|-----|
| 10% | — |
| 25% | — |
| 50% | — |
| 100% | — |
