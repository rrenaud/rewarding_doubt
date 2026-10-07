# 22. Beating the paper: online F1-label runs, checkpoint selection, longer training (Llama-3-8B)

**When / where:** Oct 7, 2026. Minimal trainer online with F1 labels (`--online --grading f1`), lr 3e-4, W = 1,
adapters saved every 250 steps; Modal L40S, continued on RunPod A100s. Released evaluation (`subset.py evaluate`) on
the 512 dev questions and the full validation set; `runs/minimal/llama-paper-release-full-metrics.json`.

**Compute (estimated):** 4 runs, 7.4 GPU-hours if run one after another, about $16 (Modal L40S + RunPod); released evaluations estimated: 9 dev runs at 5 min, 4 full-set runs at 45 min. See `scripts/sweep_costs.py`.


## Question and motivation

Do our best settings beat the paper's Llama numbers (ECE 0.0226, AUROC 0.859) under the paper's own evaluation?

## Inputs

Late-half (16–31) vs all-layer LoRA (rank 8), 1,000 steps, one seed. Then: late-half continued to 2,000 steps;
late-half at rank 4, 2,000 steps. Checkpoint chosen by dev ECE (released evaluation).

## Results

Released evaluation on the dev questions (selection):

| adapter | ECE | AUROC | accuracy |
|---|---|---|---|
| late-half, step 500 / 750 / **1,000** | 0.066 / 0.057 / **0.031** | 0.858 / 0.845 / **0.862** | 0.650 / 0.666 / 0.652 |
| all-layer, step 500 / 750 / 1,000 | 0.069 / 0.048 / 0.064 | 0.866 / 0.831 / 0.827 | 0.654 / 0.645 / 0.637 |
| late-half continued, step 1,250 / 2,000 | 0.048 / 0.084 | 0.849 / 0.836 | 0.659 / 0.642 |
| late-half rank 4, step 1,000 / 2,000 | 0.041 / 0.082 | 0.841 / 0.826 | 0.657 / 0.662 |

Full validation set (11,313 questions):

| model | ECE | AUROC | Brier | accuracy |
|---|---|---|---|---|
| untrained Llama-3-8B | 0.303 | 0.625 | 0.303 | 0.661 |
| paper | **0.0226** | 0.859 | – | – |
| **late-half, step 1,000** | 0.031 | **0.877** | 0.132 | 0.635 |
| all-layer, step 500 | 0.063 | 0.867 | 0.140 | 0.646 |

## What we learned

- **We beat the paper's AUROC (0.877 vs 0.859), not its ECE (0.031 vs 0.023),** at a cost of 2.6 points of accuracy.
  Held-out (excluding dev) numbers are the same.
- **Training longer online made it worse;** the curves oscillate between checkpoints. The cause turned out to be
  overfitting to the 8,000 cached questions ([25](25-llama-lr-schedule-2k.md)), not the learning rate.
- Of the accuracy loss, numeric answers (3.7% of questions) lose 11 points and account for about 15%; the rest is
  diffuse. Wrong-format outputs rose only 0.5% → 0.7%.

## Ratings

- **Motivation ★★★** — the project's headline comparison, under the paper's protocol.
- **Knowledge: high** — the headline result, and the observation that led to the overfitting diagnosis. One seed
  per arm; the step-1,000 checkpoint may be partly lucky (dev ECE resolves only to about ±0.03).
