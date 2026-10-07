# 07. Fast-loop schedule: KL mode, batch size and learning rate (Qwen-2.5-3B)

**When / where:** Oct 5–6, 2026. `modal_repro/fast_loop.py` on cached base-model answers, one L40S per run;
[`docs/fast_loop_log.md`](../fast_loop_log.md) sections 1, 3, 4; configs in `runs/fast/`.

**Compute (estimated):** 8 runs, 2.5 GPU-hours if run one after another, about $6 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

The released schedule (8 questions per batch, 4 passes in minibatches of 4, lr 1e-5) took 48 minutes for 1,000 steps
at 35% GPU use. Can fewer, larger updates match it in minutes?

## Inputs

1. Released schedule, KL adaptive (target 6) vs fixed 0.05, 1,000 steps.
2. One Adam update per batch: (batch 32; lr 1e-5, 3e-5, 1e-4) and (batch 16; lr 3e-5), 300–600 steps.
3. Length bucketing on/off at batch 32, lr 1e-4.

One seed each.

## Results (Brier / ECE / AUROC on cached dev answers)

| schedule | final | time |
|---|---|---:|
| released, adaptive KL, 1,000 steps | 0.137 / 0.037 / 0.874 | 45.8 min |
| released, fixed KL 0.05, 1,000 steps | 0.133 / 0.025 / 0.877 | 48.1 min |
| batch 32, lr 1e-5, 300 steps | 0.183 / 0.037 / 0.788 | 5.5 min |
| batch 32, lr 3e-5 | 0.155 / 0.056 / 0.842 | 5.5 min |
| **batch 32, lr 1e-4** | **0.117 / 0.035 / 0.905** | 5.4 min |
| batch 16, lr 3e-5, 600 steps | 0.140 / 0.058 / 0.871 | 5.1 min |
| batch 32, lr 1e-4, bucketed | 0.122 / 0.056 / 0.908 | 4.7 min |

## What we learned

- **With 8× fewer updates per question, the learning rate must rise** about 10×: lr 1e-4 at one update per 32
  questions beats the 48-minute released schedule in 5 minutes. (Later, [08](08-fast-optimizers.md) moved it to 3e-4.)
- KL mode barely mattered on this horizon.
- Bucketing cut padding from 16.9% to 0.2% and time by 13%, with no measurable effect on the metrics.

## Ratings

- **Motivation ★★★** — a speed goal with an explicit quality bar (match the released schedule).
- **Knowledge: high** — set the fast loop's schedule used by every later Qwen and Llama sweep.

## Caveats

One seed per point; frozen cached answers (no drift measured until [11](11-answer-kl-weight.md)).
