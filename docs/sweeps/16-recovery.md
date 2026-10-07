# 16. Recovering a drifted model's answers (Qwen-2.5-3B)

**When / where:** Oct 6 (evening), 2026. `--init` from the 3,000-step online `v_proj` run (Brier 0.108, answer KL 2.06,
accuracy 0.418 vs base 0.426), offline, fixed answer-KL weight, 300 steps, 2 seeds;
[`docs/minimal_trainer.md`](../minimal_trainer.md) "Recovering" (`runs/minimal/recover-*`).

## Question and motivation

How much penalty does it take to undo drift after the fact, and does it cost the calibration?

## Inputs

lr {1e-2, 3e-3} × W {0, 0.3, 1, 3, 10, 30} (3e-3 only with W 3 and 10).

## Results (step 300)

| lr, W | regen accuracy | Brier | AUROC | answer KL |
|---|---|---|---|---|
| 1e-2, 0 | 0.416 | 0.106 | 0.919 | 3.13 |
| 1e-2, 1 | 0.433 | 0.116 | 0.912 | 0.97 |
| 1e-2, 10 | 0.440 | 0.124 | 0.895 | 0.83 |
| **3e-3, 3** | **0.435** | **0.108** | **0.911** | 1.31 |
| 3e-3, 10 | 0.434 | 0.107 | 0.910 | 1.28 |

## What we learned

- **With W ≥ 1 the answers return to base within 50 steps;** beyond that W barely matters.
- **At lr 3e-3, recovery is nearly free** (Brier 0.107–0.108): only part of the 2 nats of drift hurt the answers.
- Suggested a gentle penalized polish phase after online training (not tried online).

## Ratings

- **Motivation ★★** — a direct question about an observed drift, cheaply answered.
- **Knowledge: medium** — showed drift is partly harmless and reversible; superseded in practice by the weight
  floor ([17](17-weight-floor.md)), which avoids the drift in one phase.
