# 03. Frozen answers and their learning rate (Llama-3-8B, exact + hinge vs PPO + hinge)

**When / where:** Oct 3, 2026. Follow-up to [01](01-llama-hparam-search.md). `--frozen-answers` generates the answer
with the adapter off, so training only shapes the confidence. Runs:
`runs/hparam-search-20261002T044831Z/{frozen_compare,frozen_lr}`; amendment in `PROTOCOL.md`.

## Question and motivation

On-policy finals occasionally damaged their answers (one exact seed in five collapsed). Does freezing the answers
remove that failure, at what cost, and does it move the best learning rate?

## Inputs

frozen_compare: the two tuned final configs with frozen answers, seeds 4–8, dev and test. frozen_lr: dev-only lr
sweep pooled with frozen_compare's seeds 4–5 at the tuned rate — exact {2e-5, 4e-5 tuned, 8e-5, 1.6e-4, 3.2e-4},
PPO {2.27e-5 tuned, 4.5e-5, 9e-5}; 2 seeds each.

## Results

| arm | dev Brier | dev AUROC | test Brier / ECE / AUROC |
|---|---|---|---|
| exact, frozen, tuned 4e-5 (5 seeds) | 0.169 | 0.793 | 0.180 / 0.078 / 0.797 (0/5 broken) |
| exact, on-policy final (healthy seeds) | – | – | ECE 0.052, AUROC 0.839 (1/5 collapsed) |
| exact frozen 2e-5 / 8e-5 | 0.171 / 0.171 | 0.756 / 0.768 | 0.174 / 0.171 (mean of 2) |
| exact frozen 1.6e-4 / 3.2e-4 | 0.246 / 0.250 | 0.517 / 0.500 | collapsed |
| PPO, frozen, tuned 2.27e-5 (5 seeds) | 0.205 | 0.687 | 0.217 / 0.142 / 0.693 |
| PPO frozen 4.5e-5 / 9e-5 | 0.195 / 0.174 | 0.736 / 0.762 | 0.200 / 0.182 |

## What we learned

- **Frozen answers trade peak quality for safety:** no broken seeds, but test AUROC 0.797 against 0.839 for healthy
  on-policy seeds.
- **Exact's tuned rate held** (2e-5 to 8e-5 is a flat plateau); 1.6e-4 and up collapse, the same boundary as in 01.
- **PPO's tuned rate was too low once answers were frozen** (9e-5 beats 2.27e-5 by 0.03 Brier); still behind exact.
- A hedging check ("None", "Unknown" answers) found it does not explain the frozen/on-policy gap (< 0.01 AUROC).

## Ratings

- **Motivation ★★★** — targeted at an observed failure, with a pre-stated choice rule.
- **Knowledge: medium** — quantified the safety/quality trade-off and confirmed the lr plateau.

## Caveats

The protocol said to score test only for a chosen rate different from the tuned one; test was scored for every rate.
No selection used it.
