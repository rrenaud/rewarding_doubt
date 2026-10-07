# 26. Cosine gate: one and two epochs without overfitting (Llama-3-8B)

**When / where:** Oct 7, 2026. Minimal trainer, rank-4 LoRA on layers 16–31, W = 1, offline on the 8,000-question
cache, 2 seeds, RunPod (`runs/runpod/llama-r4-ep*-cos*`). Constant-rate baselines reuse earlier runs
([23](23-llama-lora-rank.md), [25](25-llama-lr-schedule-2k.md)): a constant-rate run's first N steps do not depend on
its planned length.

**Compute (estimated):** 8 runs, 1.5 GPU-hours if run one after another, about $1 (RunPod). See `scripts/sweep_costs.py`.


## Question and motivation

Before spending on full-split runs with cosine decay, check at small scale whether cosine helps when the data is not
repeated much. Gate (fixed before results): at 1 epoch, cosine must match or beat constant on dev Brier and AUROC
beyond seed noise, without a larger train–dev gap.

## Inputs

Cosine decay to 0 over 250 steps (1 epoch) or 500 steps (2 epochs), peak lr {3e-4, 1e-4}.

## Results (trainer's dev evaluation, mean of 2 seeds)

| | Brier | AUROC | ECE | answer KL |
|---|---|---|---|---|
| **1 epoch (step 250)** | | | | |
| cosine 3e-4 | 0.154 ± 0.002 | 0.837 | 0.033 | 0.005 |
| constant 3e-4 (rank sweep) | 0.141 ± 0.001 | 0.874 | 0.068 | 0.011 |
| cosine 1e-4 | 0.165 | 0.819 | 0.047 | 0.004 |
| constant 1e-4 (steps 200 / 300) | 0.159 / 0.162 | 0.830 / 0.848 | | 0.003 |
| **2 epochs (step 500)** | | | | |
| cosine 3e-4 | 0.131 | 0.883 | 0.041 | 0.007 |
| constant 3e-4 | 0.127–0.130 | 0.891 | 0.043–0.062 | 0.024–0.033 |
| cosine 1e-4 | 0.144 | 0.854 | 0.030 | 0.003 |
| constant 1e-4 | 0.141 | 0.874 | 0.064 | 0.004 |

Train vs dev Brier at the end (dumps): equal within 0.006 for every cosine run (0.143–0.186).

## What we learned

- **The gate fails:** at 1 epoch cosine is worse on ranking (AUROC 0.837 vs 0.874) and Brier; at 2 epochs it still
  trails on AUROC.
- **These short runs are underfit, not overfit** (train = dev), and cosine's decay simply moves the weights less.
  Its lower ECE and drift come with less learning.
- Full-split runs use a constant rate (with weight averaging).

## Ratings

- **Motivation ★★★** — a pre-registered gate protecting a larger spend. (Its first launch duplicated constant-rate
  runs we already had; those were cancelled before starting.)
- **Knowledge: medium** — a clear decision; the result is specific to short horizons, where decay costs progress.
