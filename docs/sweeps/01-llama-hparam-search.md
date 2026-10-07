# 01. Hyperparameter search: exact + hinge vs PPO + hinge (Llama-3-8B, released pipeline)

**When / where:** Oct 1–2, 2026. Llama-3-8B (4-bit, Unsloth), released `Train.py` and our `exact_llama.py`, one L40S
per run on Modal. Protocol fixed before any result: [`runs/hparam-search-20261002T044831Z/PROTOCOL.md`](../../runs/hparam-search-20261002T044831Z/PROTOCOL.md);
interactive page: `runs/hparam-search-20261002T044831Z/hparam_search.html`.

## Question and motivation

Compare the exact objective with PPO fairly: each arm tuned with the same budget, one selection rule, a dev split
for selection and a test split touched once.

## Inputs

| | exact + hinge | PPO + hinge |
|---|---|---|
| swept | lr log-uniform [3e-6, 3e-4]; updates per batch {1, 2, 4, 8}; hinge weight [0.1, 10]; hinge threshold {0.9, 0.95, 0.99} | same, plus cliprange {0.1, 0.2, 0.3}, vf_coef [0.03, 0.3] |
| fixed | batch 8, 1,024 training questions, 128 steps (1 epoch) per search run, no KL | same |

Round 1: 16 random configurations per arm (seed 1). Then four greedy one-dimensional stages around the current point
(lr ×{1/3, 1/2, 1, 2, 3}; updates per batch; lr again; hinge / clip / vf), seed 1, 128 steps. Final: the chosen
config, 256 steps, fresh seeds, dev and test (512 questions each). Selection: lowest dev Brier among **eligible**
runs (amendment 2: wrong format ≤ 2%, accuracy ≥ base − 2 points).

## Results

Round 1 eligible learning rates — exact: 4e-6 to 8e-5 (ineligible: 2.8e-5 to 2.6e-4); PPO: 6e-6 to 1.2e-4.

| stage | exact: chosen point (dev Brier) | PPO: chosen point (dev Brier) |
|---|---|---|
| round 1 | lr 8.0e-5, 1 update, hinge 1.01 @ 0.9 (0.152) | lr 2.27e-5, 8 updates, clip 0.1, vf 0.078 (0.179) |
| 1: lr | kept 8.0e-5 (0.158; 4e-5: 0.169; 1.6e-4 ineligible) | kept 2.27e-5 (0.172; 6.8e-5 collapsed, AUROC 0.50) |
| 2: updates | 2 (0.162 vs 1: 0.183; 4 and 8 collapsed, AUROC 0.50) | 2 (0.180 vs 8: 0.242) |
| 3: lr | **4.0e-5** (0.158 vs 8e-5: 0.165) | kept 2.27e-5 |
| 4: hinge / clip | threshold 0.95 (0.154) | kept (0.180) |

Final (2 epochs = 256 steps, F1 labels), mean over seeds:

| arm | seeds | dev Brier / ECE / AUROC | **test** Brier / ECE / AUROC | test accuracy |
|---|---:|---|---|---|
| exact + hinge (lr 4e-5, 2 updates) | 3 | 0.159 / 0.052 / 0.804 | **0.155 / 0.052 / 0.839** | 0.631 |
| PPO + hinge (lr 2.27e-5, 2 updates; real seeds) | 3 | 0.206 / 0.133 / 0.717 | 0.204 / 0.128 / 0.757 | 0.636 |
| PPO + hinge, F1 labels in its reward | 5 | 0.188 / 0.098 / 0.713 | 0.203 / 0.114 / 0.721 | 0.639 |
| exact + hinge, seeds 7–8 (added for parity) | 2 | 0.244 / 0.161 / 0.649 | – | 0.589 (a seed damaged its answers) |

## What we learned

- **Exact beat PPO at matched tuning effort** on test: AUROC 0.839 vs 0.757, ECE 0.052 vs 0.128. Giving PPO F1
  labels did not close the gap.
- **The selection rule had a loophole** (found in round 1): runs that destroyed their answers (accuracy 9.5%, or
  94–98% malformed) scored the best Brier, because the evaluation drops malformed replies. Eligibility filters
  became standard in every later sweep.
- **Collapse is the failure mode above the optimum:** 4 and 8 updates per batch at 8e-5, and lr ≥ 1.6e-4, give
  AUROC 0.50 (one confidence for every answer).
- **Stage decisions rested on single-seed differences of 0.004–0.02 Brier** (stage 3 halved the lr on 0.158 vs
  0.165), below the seed noise later measured (±0.01). The tuned point is "a good region", not an optimum.
- The tuned exact config (lr 4e-5) **did not transfer to long runs**: it diverged at step 57 in 4,000-step runs
  (see [05](05-llama-long-runs.md)).

## Ratings

- **Motivation ★★★** — pre-registered protocol, equal budget per arm, dev/test separation, amendments dated before
  the results they affected.
- **Knowledge: high** — the method comparison, the eligibility rule, and the collapse boundary. Weakened by
  single-seed greedy stages and a 128-step horizon.

## Caveats

1,024 questions and 128–256 steps: a short-horizon, small-data search. The first PPO final reused one seed and was
rerun (`final_ppo`); the table uses the rerun.
