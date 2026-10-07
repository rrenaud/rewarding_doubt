# 17. A floor on the adaptive answer-KL weight (Qwen-2.5-3B)

**When / where:** Oct 6 (night), 2026. Stock `v_proj` bias 18–35, lr 1e-2; offline 1,000 steps, 2 seeds
(`runs/minimal/floor-*`), then online 300 steps (2 seeds) and 3,000 steps on RunPod (one seed);
[`docs/minimal_trainer.md`](../minimal_trainer.md) "A floor on the adaptive weight".

## Question and motivation

In the 3,000-step `v_proj` run the adaptive weight sank to ~0.001 and the answers drifted past the knee (about 2 nats
per answer). Does a floor on the weight hold the answers, and at what calibration cost?

## Inputs

Target {1.5, 2} × floor {0.001, 0.1, 0.3, 1, 3}.

## Results (offline, steps 500–1,000)

| target, floor | regen accuracy @1,000 | Brier | AUROC | answer KL | late weight |
|---|---|---|---|---|---|
| 1.5, 0.001 | 0.420 | 0.113 | 0.916 | 1.68 | 0.037 |
| 2, 0.001 | 0.420 (0.388 at 500) | 0.113 | 0.916 | 2.26 | 0.013 |
| **1.5, 0.1** | **0.430** | 0.118 | 0.910 | 1.05 | 0.100 |
| 2, 0.1 | 0.427 | 0.114 | 0.911 | 1.08 | 0.100 |
| 1.5, 1 | 0.439 | 0.133 | 0.889 | 0.33 | 1.000 |
| 1.5, 3 | 0.435 | 0.144 | 0.867 | 0.20 | 3.000 |

Online 3,000 steps, target 1.5 / floor 0.1 vs target 2 / no floor: Brier 0.112 vs 0.108, AUROC 0.912 vs 0.919, answer
KL 1.39 vs 2.54, late accuracy at or above base vs dips to 0.404.

## What we learned

- **From a floor of 0.1 up, the weight sits on the floor** and the target stops mattering: in practice this is a
  fixed weight of 0.1.
- **Floor 0.1 is the knee:** accuracy at base with KL near 1 nat for about 0.004 Brier; higher floors buy little
  accuracy for up to 0.03 Brier.

## Ratings

- **Motivation ★★★** — a targeted fix for a diagnosed failure, with the trade-off measured.
- **Knowledge: medium–high** — simplified the controller to a constant and located the trade-off's knee on Qwen.
