# 23. LoRA rank (Llama-3-8B, late half)

**When / where:** Oct 7, 2026. Minimal trainer, LoRA on layers 16–31, lr 3e-4, W = 1, alpha = rank (scale 1), offline,
500 steps, 2 seeds; ranks 1–4 on RunPod 4090s (`runs/runpod/llama-lorafull16-r*`), rank 8 from [21](21-llama-late-layers.md).

**Compute (estimated):** 6 runs, 1.4 GPU-hours if run one after another, about $1 (RunPod); includes the later per-example dump reruns. See `scripts/sweep_costs.py`.


## Question and motivation

Is rank 8 needed? A smaller adapter might drift less with the same calibration.

## Inputs

Rank ∈ {1, 2, 4} (and 8).

## Results (step 500; trainer's dev evaluation)

| rank | Brier (seeds) | AUROC (seeds) | answer KL | regen accuracy (base 0.667) |
|---|---|---|---|---|
| 1 | 0.137, 0.130 | 0.880, 0.881 | 0.005 | 0.673, 0.661 |
| 2 | 0.128, 0.131 | 0.887, 0.887 | 0.006 | 0.661, 0.667 |
| **4** | **0.128, 0.127** | **0.893, 0.891** | 0.02 | 0.665, 0.671 |
| 8 | 0.129, 0.128 | 0.885, 0.893 | 0.15 | 0.659, 0.659 |

## What we learned

- **Rank 4 matches rank 8's calibration with about 8× less answer drift,** and kept accuracy where rank 8 lost a
  point. Rank 2 is close; rank 1 is slightly worse.
- Peak memory is set by activations (20.4 GB at `--accumulate 4`), not rank.

## Ratings

- **Motivation ★★** — a standard capacity question, cheaply run.
- **Knowledge: medium–high** — rank 4 became the default; drift scales with rank more than calibration does.
