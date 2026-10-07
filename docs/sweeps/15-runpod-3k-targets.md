# 15. 3,000-step online runs: answer-KL target (Qwen-2.5-3B, RunPod)

**When / where:** Oct 6 (evening), 2026. Minimal trainer online, layers 18–35, lr 1e-2, one seed, RTX 4090s
(`runs/runpod/online-*`); [`docs/minimal_trainer.md`](../minimal_trainer.md) "3,000-step online runs".

**Compute (estimated):** 5 runs, 3.7 GPU-hours if run one after another, about $3 (RunPod). See `scripts/sweep_costs.py`.


## Question and motivation

The 300-step winners ([13](13-online-trials.md), [14](14-stock-params.md)) at 10× the horizon: which answer-KL target
keeps answers and calibration over 3,000 steps?

## Inputs

Hooked attention bias with target {0.5, 1, 2}; stock `v_proj` bias with target 2.

## Results

- **Targets 0.5 and 1 wrecked their runs by step ~1,800:** the answer KL stayed above target (at this lr Adam's step
  does not shrink as the weight grows, so the KL has a floor of a few nats) and the unbounded controller drove the
  weight to 1e22–1e33. The controller was then bounded to [1e-3, 10] with a moving average.
- Target 2 (mean of steps 1k–3k): hooked bias Brier 0.122, AUROC 0.909, accuracy 0.422–0.458; `v_proj` bias 0.108,
  0.919, accuracy 0.404–0.428 (1–3.6 points below start).

## What we learned

- **A multiplicative controller needs bounds** when the target can be below the achievable floor.
- Over long runs the controller's weight sinks and drift accumulates for `v_proj` (→ [17](17-weight-floor.md)).

## Ratings

- **Motivation ★★** — a reasonable horizon test; one seed per target.
- **Knowledge: medium** — mostly a controller bug and its fix; the calibration numbers are single-seed.
