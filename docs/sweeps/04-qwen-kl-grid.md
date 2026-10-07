# 04. KL settings grid (Qwen-2.5-3B, released code + patch)

**When / where:** Oct 3–4, 2026. Released `Train.py` with the `--objective exact` patch, full TriviaQA training set,
3,000 steps, one RTX 4090 per run on RunPod (`runs/runpod/klgrid-*`). Write-up:
[`patches/PR_DESCRIPTION.md`](../../patches/PR_DESCRIPTION.md), commit 472cf22.

## Question and motivation

The obvious objection to "exact beats PPO": maybe PPO is held back by its KL constraint. Loosen it and see.

## Inputs

KL setting ∈ {adaptive target 6 (released default), adaptive target 20, fixed 0.05, none} × objective {exact, ppo};
lr 1e-5, batch 8, released schedule, one seed each. Companion seed check: 1,024 questions, 256 steps, 3 seeds per
objective.

## Results

Mean over snapshots at steps 1,000–3,000, 512 dev questions (ECE / AUROC):

| KL setting | ppo | exact |
|---|---|---|
| adaptive, target 6 (default) | 0.106 / 0.864 | **0.035 / 0.894** |
| adaptive, target 20 | 0.128 / 0.848 | 0.051 / 0.863 |
| fixed 0.05 | 0.162 / 0.817 | 0.055 / 0.883 |
| none | 0.114 / 0.849 | 0.071 / 0.872 |

Seed check (3 seeds): exact ECE 0.056 ± 0.021, AUROC 0.803; ppo 0.257 ± 0.018, 0.781. Time per iteration 2.3 s vs 15.1 s.

## What we learned

- **PPO is not limited by its KL constraint:** the released default is PPO's best setting, and exact is ahead at
  every setting.
- **For exact, the adaptive controller is best over 3,000 steps but harmful over very long runs:** on a 21,900-step
  run it kept raising β to hold 6 nats and pulled the confidence back to the base model's near-constant "10"
  (ECE 0.032 → 0.222 by step 15,000); a fixed β did not.

## Ratings

- **Motivation ★★★** — answers a specific, anticipated objection to the main claim.
- **Knowledge: high** — closes off the KL explanation and exposes the adaptive controller's long-run failure.

## Caveats

One seed per cell for the long runs; ECE on 512 questions resolves only to about 0.04.
