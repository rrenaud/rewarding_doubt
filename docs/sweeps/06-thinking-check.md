# 06. Unscored thinking before the confidence (Llama-3-8B)

**When / where:** Oct 2, 2026. `modal_repro/thinking_llama.py`; design and results in
[`docs/thinking_experiment.md`](../thinking_experiment.md). Not a hyperparameter sweep proper: an arm comparison with
one KL-weight variation.

**Compute (estimated):** 18 runs, 5.6 GPU-hours if run one after another, about $13 (Modal L40S); recorded spend. See `scripts/sweep_costs.py`.


## Question and motivation

If the model writes a short "check" before stating its confidence, does the confidence get better?

## Inputs

512 training questions, 128 steps, tuned exact + hinge settings. Arms: no check, fixed filler text, frozen base-model
check, trained check with β ∈ {0.05, 0.005}; G = 4 checks per question for frozen and trained; 2–4 seeds.

## Results (dev, sampled confidence)

| arm | seeds | Brier | ECE | AUROC |
|---|---:|---|---|---|
| no check | 4 | 0.176 | 0.058 | 0.783 |
| filler | 2 | 0.181 | 0.082 | 0.782 |
| frozen check | 4 | 0.167 | 0.067 | 0.823 |
| trained check, β 0.05 | 4 | 0.177 | 0.080 | 0.792 |
| trained check, β 0.005 | 2 | 0.177 | 0.099 | 0.815 |

## What we learned

- Training the check gave nothing measurable in 128 steps; the policy-gradient signal on the check is weak
  (advantages 0.02–0.06).
- The frozen arm's small AUROC gain mostly survives removing the check at test time: likely a data-augmentation
  effect of four contexts per answer, not reasoning.
- The β variation made no difference at this scale.

## Ratings

- **Motivation ★★** — an idea worth one cheap test; the design anticipated its confounds.
- **Knowledge: low** — a clean negative at reduced scale; the β axis taught nothing.
