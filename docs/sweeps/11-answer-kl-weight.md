# 11. Protecting the answers: answer-KL penalty weight (Qwen-2.5-3B)

**When / where:** Oct 6, 2026. Fast loop (4-bit) and the minimal trainer (bf16), default schedule, 2 seeds;
[`docs/fast_loop_log.md`](../fast_loop_log.md) section 10, [`docs/minimal_trainer.md`](../minimal_trainer.md)
"Answer drift" (`runs/minimal/akl-*`).

## Question and motivation

A 3,000-step attention-bias run in the released pipeline fell to 0.315 dev accuracy: adapters trained only on the
confidence also move the answers. Measure the drift (answer KL to the base model; regenerated dev accuracy) and
penalize it with weight W.

## Inputs

W ∈ {0, 0.1, 1, 10} for the attention bias (18–35, lr 1e-2); W ∈ {0, 1} for all-layer LoRA (lr 3e-4).

## Results (minimal trainer, step 300; base regenerated accuracy 0.426)

| arm | Brier | AUROC | answer KL | regen accuracy @300 | malformed |
|---|---|---|---|---|---|
| bias, W=0 | 0.114 | 0.912 | 11.5 | 0.217 | 18% |
| bias, W=0.1 | 0.115 | 0.908 | 1.19 | 0.423 | 0.2% |
| bias, W=1 | 0.128 | 0.900 | 0.78 | 0.439 | 0% |
| bias, W=10 | 0.156 | 0.843 | 0.89 | 0.430 | 0.7% |
| LoRA, W=0 | 0.112 | 0.917 | 5.66 | 0.404 (0.223 at step 200) | 7.1% |
| LoRA, W=1 | 0.112 | 0.912 | 0.28 | 0.439 | 0.6% |

The fast loop showed the same pattern (bias W=0: regen 0.139; LoRA W=1: 0.403 vs base 0.396).

## What we learned

- **Unpenalized, the adapters wreck the answers** even though only the confidence is trained; every earlier
  calibration number was of the confidence on the base model's answers.
- **A small penalty is nearly free:** W = 0.1 (bias) and W = 1 (LoRA) keep accuracy at base with no measurable
  calibration cost; W = 10 costs calibration and protects no better.
- The two measurements (answer KL, regenerated accuracy) became standard in every later sweep.

## Ratings

- **Motivation ★★★** — triggered by an observed failure, with a clear mechanism to test.
- **Knowledge: high** — the most consequential finding of the fast-loop phase; it reframed the problem as
  calibration under an answer-drift constraint.
