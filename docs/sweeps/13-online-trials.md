# 13. Online trials of the offline winners (Qwen-2.5-3B, minimal trainer)

**When / where:** Oct 6 (evening), 2026. `--online`: each batch's questions answered by the policy, graded, trained
on, with a live reference; 300 steps, 2 seeds; [`docs/minimal_trainer.md`](../minimal_trainer.md) "Online trials"
(`runs/minimal/on-*`).

## Question and motivation

Offline training uses the base model's answers forever. Online, the policy trains on its own (drifted) answers, so
drift could compound. Do the offline winners of [12](12-offline-search-topk.md) survive?

## Inputs

Attention bias (lr 1e-2, adaptive target 1; lr 3e-3, W=0.03; no penalty) and all-layer LoRA (lr 3e-4, W=1; no penalty).

## Results (step 300)

| arm | Brier | AUROC | mean Brier, steps 150–300 | answer KL | worst Δ regen accuracy | malformed |
|---|---|---|---|---|---|---|
| **bias, lr 1e-2, target 1** | **0.110** | **0.915** | 0.125 | 1.11 | −0.002 | 0.5% |
| bias, lr 3e-3, W=0.03 | 0.136 | 0.896 | 0.128 | 0.83 | −0.006 | 0.8% |
| LoRA, lr 3e-4, W=1 | 0.159 | 0.851 | 0.141 | 0.33 | −0.004 | 0.5% |
| LoRA, no penalty | 0.112 | 0.921 | 0.113 | 3.99 | −0.148 | 4% |
| bias, no penalty | 0.127 | 0.899 | 0.132 | 15.4 | −0.323 | 33–42% |

## What we learned

- **Unpenalized online training compounds drift:** the bias's answers fall apart (accuracy −32 points, answers growing
  from 7 to 26–35 tokens).
- **With the penalty, answers hold online too** (every penalized arm within half a point).
- **The offline ranking flipped:** the attention bias with a 1-nat target was best online, better than offline;
  LoRA with W = 1, best offline, was unsteady online (ECE swinging 0.02–0.16).

## Ratings

- **Motivation ★★★** — tests whether offline selection transfers to the real training mode.
- **Knowledge: high** — offline and online rankings can disagree; online selection needs online trials.
