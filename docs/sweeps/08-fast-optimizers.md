# 08. Optimizers: Adam, Muon, Scaled AdamW, PoLoRA (Qwen-2.5-3B, fast loop)

**When / where:** Oct 6, 2026. Fast loop, all-layer rank-8 LoRA, batch 32, one pass, bucketed, fixed KL 0.05, 300
steps; [`docs/fast_loop_log.md`](../fast_loop_log.md) section 5; configs `runs/fast/{muon,lora_optim,seed_sweep}_configs.json`.

## Question and motivation

User-driven: is there a better optimizer for LoRA than Adam? Literature suggested factor-aware methods (PoLoRA,
Scaled AdamW) and Muon.

## Inputs

Muon lr {3e-5, 1e-4, 3e-4, 1e-3}; Scaled AdamW lr {3e-5, 1e-4, 3e-4}; PoLoRA lr {3e-3, 1e-2, 3e-2} (one seed each).
Then matched learning rates, 3 seeds: Adam {3e-4, 1e-3}, PoLoRA 3e-3, Scaled AdamW {3e-4, 1e-3}.

## Results (3 seeds, step 300, mean ± sd)

| arm | Brier | ECE | AUROC | Brier, mean of steps 150–300 |
|---|---|---|---|---|
| **Adam 3e-4** | **0.120 ± 0.004** | **0.040 ± 0.014** | **0.915 ± 0.005** | 0.125 ± 0.009 |
| PoLoRA 3e-3 | 0.124 ± 0.018 | 0.068 ± 0.039 | 0.910 ± 0.005 | 0.125 ± 0.001 |
| Scaled AdamW 3e-4 | 0.139 ± 0.029 | 0.072 ± 0.045 | 0.905 ± 0.012 | 0.129 ± 0.012 |
| Adam 1e-3 | 0.202 ± 0.062 | 0.040 ± 0.026 | 0.731 ± 0.150 | 0.196 ± 0.047 |
| Scaled AdamW 1e-3 | 0.208 ± 0.059 | 0.064 ± 0.030 | 0.669 ± 0.204 | 0.211 ± 0.054 |

Muon tied Adam at 1e-4 and collapsed at 3e-4 after leading at step 150.

## What we learned

- **No optimizer beat Adam at a matched learning rate.** The single-seed "wins" of Scaled AdamW and PoLoRA came from
  comparing them at higher rates than Adam had been run at.
- **Adam's best rate moved to 3e-4** (from 1e-4 in [07](07-fast-schedule.md)); 1e-3 is past the edge of stability
  (2 of 3 seeds collapse to a near-constant confidence).
- **ECE alone would have picked a collapsed run** (Muon at step 200: ECE 0.013, AUROC 0.566).
- PoLoRA was the most consistent across seeds, not better on average.

## Ratings

- **Motivation ★★** — literature-backed curiosity rather than a bottleneck; the objective (an 11-way distribution
  anchored by KL) gave little reason to expect optimizer gains.
- **Knowledge: medium** — a negative result plus two durable rules: compare at matched, tuned learning rates, and
  never select on ECE alone. The optimizer code was removed.
