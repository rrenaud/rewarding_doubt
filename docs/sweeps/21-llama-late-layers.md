# 21. Late-layer LoRA depth (Llama-3-8B)

**When / where:** Oct 7, 2026. Minimal trainer, rank-8 LoRA on all 7 projections, lr 3e-4, W = 1, offline, 500
steps, 2 seeds (`runs/minimal/llama-lorafull{8,16,24}-*`, `llama_latelora_configs.json`). Learning curves:
`docs/learning_curves.html`.

## Question and motivation

Full LoRA on Llama needs 53 GB per step (or gradient accumulation). Training only later layers stops the backward
pass earlier: cheaper, and maybe less drift. How shallow can it go?

## Inputs

Layers {0–31 (full), 8–31 (`--accumulate 2`), 16–31, 24–31}.

## Results (step 500)

| layers | Brier | AUROC | answer KL | regen accuracy | peak memory |
|---|---|---|---|---|---|
| 0–31 | 0.136 | 0.883 | 0.78 | 0.661 | 53 GB (33 GB with accumulate 2) |
| 8–31 | 0.134 | 0.886 | 0.49 | 0.649 | |
| **16–31** | **0.128** | **0.889** | 0.15 | 0.659 | 34.6 GB |
| 24–31 | 0.138 | 0.879 | 0.05 | 0.661 | |

Elbow: full LoRA reaches 90% of its Brier gain in 50–100 steps; small adapters take 250–350.

## What we learned

- **The late half (16–31) is as good as or better than full LoRA** at a fifth of the drift and two-thirds of the
  memory; it fits a 48 GB card without accumulation.
- The last quarter (24–31) is close behind on Llama (unlike Qwen's last quarter, [09](09-fast-adapter-ablation.md)).

## Ratings

- **Motivation ★★** — motivated by cost, with Qwen's depth result as a prior.
- **Knowledge: high** — set the late-half adapter used for everything after, including the paper comparison.
