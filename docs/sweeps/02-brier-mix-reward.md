# 02. Mixing the Brier score into the reward (Llama-3-8B, exact + hinge)

**When / where:** Oct 2, 2026. Follow-up to [01](01-llama-hparam-search.md), same pipeline and tuned exact + hinge config.
Runs: `runs/hparam-search-20261002T044831Z/{brier_sweep,brier_sweep2,brier_full}`, choice in `brier_choice.json`.

**Compute (estimated):** 26 runs, 5.7 GPU-hours if run one after another, about $13 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

Selection is by Brier, but the reward is a clipped log score. Would rewarding (1 − m)·log score + m·Brier
directly improve dev Brier?

## Inputs

m ∈ {0, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1}; 512 training questions, 128 steps (2 epochs), 2–4 seeds per value, dev
only. Rule fixed before sweep 2: lowest mean dev Brier, disqualified if any seed is ineligible. Then the chosen m vs
m = 0 at full scale (1,024 questions, 256 steps, seeds 6–8, dev and test).

## Results

| m | seeds | dev Brier | dev ECE | dev AUROC | note |
|---|---:|---|---|---|---|
| 0 | 4 | 0.164 | 0.074 | 0.790 | |
| 0.25 | 2 | 0.175 | 0.095 | 0.797 | a seed ineligible |
| 0.5 | 4 | 0.162 | 0.055 | 0.797 | chosen |
| 0.6 | 2 | 0.163 | 0.040 | 0.791 | |
| 0.75 | 2 | 0.241 | 0.175 | 0.644 | answers damaged |
| 1 | 2 | 0.202 | 0.126 | 0.764 | |

Full scale (test, 3 seeds): m = 0 Brier 0.156 / ECE 0.078 / AUROC 0.843; m = 0.5 0.169 / 0.077 / 0.800.

## What we learned

- **The small-scale pick did not replicate.** m = 0.5 won by 0.002 Brier on 512 questions and lost by 0.013 Brier
  and 0.043 AUROC at full scale. The log score stays.
- Large m (≥ 0.75) damages answers: the Brier reward's bounded penalty gives weaker pressure against confident
  wrong answers, and the policy drifts.

## Ratings

- **Motivation ★★** — a fair question (align reward with the selection metric), but the margin it could win was
  small and the sweep was too small to resolve it.
- **Knowledge: low** — a negative result, plus a methodological one: 2–4-seed differences of a few thousandths are
  noise; confirm at full scale before adopting.
