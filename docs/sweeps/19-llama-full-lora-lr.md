# 19. Full LoRA on Llama: penalty and the released schedule (Llama-3-8B)

**When / where:** Oct 6 (night), 2026. Minimal trainer, all-layer rank-8 LoRA, offline, 500 steps, 2 seeds
(`runs/minimal/llama-lorafull-*`, `llama_lorafull_configs.json`).

## Question and motivation

After [18](18-llama-small-adapters.md): is our learner buggy on Llama, or is drift adapter-dependent? Run the paper's
own adapter (full LoRA) with our schedule and with the released schedule.

## Inputs

lr 3e-4 with W ∈ {0, 1} (our schedule: 32 questions, one update); lr 1e-5 with the released schedule (batch 8,
4 passes of minibatch 4), W = 0.

## Results (step 500; base regenerated accuracy 0.656–0.669)

| arm | Brier | AUROC | answer KL | regen accuracy | best Brier (step) |
|---|---|---|---|---|---|
| lr 3e-4, W=1 | 0.136 | 0.883 | 0.78 | 0.661 | 0.130 (300) |
| lr 3e-4, W=0 | 0.216 | 0.727 | 62.4 | (answers broken) | 0.126 (250) |
| released schedule, lr 1e-5, W=0 | 0.138 | 0.888 | 0.94 | 0.646 | 0.134 (425) |

## What we learned

- **The learner is fine:** full LoRA with W = 1 holds accuracy within ±2 points up to about 0.8 nats, and the
  released schedule reaches the same calibration. Drift tolerance depends on the adapter, not a bug.
- **Unpenalized full LoRA at 3e-4 peaks at step 250 and then destroys the answers** (KL 62).

## Ratings

- **Motivation ★★★** — a decisive check of a specific suspicion (a learner bug), using the paper's configuration.
- **Knowledge: high** — established full LoRA + W = 1 as the Llama baseline.
