# 12. Offline search with the cached top-k reference (Qwen-2.5-3B, minimal trainer)

**When / where:** Oct 6 (evening), 2026. Minimal trainer, cached answers, top-64 reference for the answer-KL penalty,
300 steps, 2 seeds; [`docs/minimal_trainer.md`](../minimal_trainer.md) "Search with the cached top-k reference"
(`runs/minimal/hs-*`, 22 arms).

**Compute (estimated):** 44 runs, 4.1 GPU-hours if run one after another, about $9 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

With drift measurable ([11](11-answer-kl-weight.md)), find the best calibration among configurations that keep the
answers. Cached top-k reference log-probs make the penalty free of an extra forward pass, so the search is cheap.

## Inputs

Adapter {all-layer LoRA, attention bias 18–35} × lr (LoRA 3e-4, 1e-3; bias 3e-3, 1e-2, 3e-2) × penalty (fixed W
0.03–1, or an adaptive answer-KL target of 1 nat). "Safe" (fixed before the results): answer KL ≤ 2, regenerated
accuracy no more than 1.5 points below step 0, malformed ≤ 2%.

## Results (best and instructive arms, step 300)

| arm | Brier | AUROC | answer KL | Δ accuracy | safe |
|---|---|---|---|---|---|
| LoRA, lr 3e-4, W=1 | 0.108 | 0.920 | 0.29 | +0.015 | yes |
| bias, lr 3e-2, W=0.3 | 0.113 | 0.914 | 2.57 | −0.007 | no (KL) |
| LoRA, lr 3e-4, W=0.3 | 0.120 | 0.905 | 0.61 | +0.012 | yes |
| bias, lr 1e-2, target 1 | 0.120 | 0.910 | 1.19 | +0.012 | yes |
| LoRA, lr 3e-4, target 1 | 0.221 | 0.730 | 0.72 | +0.028 | one seed collapsed |
| LoRA, lr 1e-3, W=0.3 | 0.246 | 0.697 | 2.86 | +0.021 | no |

## What we learned

- **Offline, LoRA with W = 1 is best** (Brier 0.108, AUROC 0.920) and safe.
- **The penalty also steadies LoRA's confidence:** too little of it, or lr 1e-3, and a seed collapses.
- **An adaptive target suits small adapters, not LoRA:** LoRA's KL (≈ 0.3) sits below a 1-nat target, so the
  controller weakens the penalty and destabilizes the run.

## Ratings

- **Motivation ★★★** — a constrained search with the safety rule fixed in advance.
- **Knowledge: medium** — the ranking it produced did not survive the move online ([13](13-online-trials.md)),
  but the adaptive-target finding did.
