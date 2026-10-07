# 24. Residual-stream biases on top of rank-4 LoRA (Llama-3-8B)

**When / where:** Oct 7, 2026. Minimal trainer, rank-4 LoRA on layers 16–31 (lr 3e-4, W = 1) plus bias vectors added
to the residual stream after each MLP (and attention) in layers 16–31 with their own learning rate (`--bias-lr`);
offline, 500 steps, 2 seeds, RunPod (`runs/runpod/llama-lora16r4-*`), per-example dumps.

## Question and motivation

User hypothesis: simple learned residual additions might add modelling power that LoRA lacks.

## Inputs

{MLP-output bias, attention + MLP bias} × bias lr {3e-5, 1e-4, 3e-4 (MLP only)}.

## Results (per-example dumps, final weights; lower loss is better)

| arm | train loss | train Brier | dev Brier | dev AUROC | train answer KL | regen accuracy |
|---|---|---|---|---|---|---|
| **rank 4 alone** | **−9.717** | **0.139** | 0.149 | 0.892 | 0.084 | 0.668 |
| + MLP bias, 3e-5 | −9.658 | 0.147 | 0.155 | 0.890 | 0.084 | 0.662 |
| + MLP bias, 1e-4 | −9.634 | 0.148 | 0.149 | 0.893 | 0.108 | 0.661 |
| + MLP bias, 3e-4 | −9.529 | 0.159 | 0.159 | 0.893 | 0.159 | 0.668 |
| + attention + MLP, 3e-5 | −9.636 | 0.148 | 0.152 | 0.892 | 0.100 | 0.666 |
| + attention + MLP, 1e-4 | −9.634 | 0.140 | 0.145 | 0.891 | 0.123 | 0.664 |

## What we learned

- **No added modelling power:** every bias arm fits the training set slightly worse than LoRA alone, even counting
  only the confidence terms; dev is within seed noise; the biases mainly add drift.
- **Capacity is not the bottleneck at 500 steps:** train and dev losses are nearly equal, and extra parameters
  (biases, or rank 4 vs 2) barely improve the training fit.

## Ratings

- **Motivation ★★** — a clear hypothesis, tested by training fit (the right measure of capacity).
- **Knowledge: medium** — a clean negative and the "capacity is not the bottleneck" observation.
