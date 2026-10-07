# 05. Long runs: learning rate and regularizer at 4,000 steps (Llama-3-8B)

**When / where:** Oct 2–3, 2026. Our `exact_llama.py` and the released PPO (`subset.py`), full TriviaQA training set,
4,000 steps, RTX 4090s on RunPod (`runs/runpod/long-*`, `long2-*`, `long3-*`). Curves: [`docs/long_runs.html`](../long_runs.html).

**Compute (estimated):** 12 runs, 32.6 GPU-hours if run one after another, about $24 (RunPod). See `scripts/sweep_costs.py`.


## Question and motivation

Do the search's short-horizon winners ([01](01-llama-hparam-search.md)) hold over a long run, and does exact keep
its lead over PPO there?

## Inputs

| run | objective | lr | regularizer | seeds |
|---|---|---|---|---|
| long / long2 exact-tuned | exact | 4e-5 (tuned), 2 updates | hinge | 2 |
| long2 exact-1e5 | exact | 1e-5, 2 updates | hinge | 2 |
| long2 ppo-paper | PPO | 1e-5 (paper) | released KL | 2 |
| long3 exact-kl | exact | 1e-5, 4 updates | KL (β 0.05) | 2 |
| long3 ppo-kl | PPO, frozen answers | 1e-5 | KL | 2 |

## Results

- **exact-tuned (4e-5) diverged** in both seeds (the stability monitor stopped them; seed 2's format broke at
  step 57, reproducibly).
- **exact 1e-5 + hinge:** Brier 0.146–0.149, ECE 0.047–0.057, AUROC 0.815–0.825 at step 4,000.
- **exact 1e-5 + KL:** dev ECE 0.038 at step 4,000 (2 seeds).
- **PPO paper recipe:** still improving at step 2,000 (AUROC 0.806, ECE 0.121) when it ran out of memory on 24 GB;
  PPO + KL reached ECE 0.150 by step 3,500 (one seed; the other ran out of memory).

## What we learned

- **A learning rate tuned on 128-step runs can be unstable over 4,000 steps.** The search's 4e-5 broke the format
  almost immediately here; the paper's 1e-5 was stable.
- Exact keeps a large ECE lead over PPO at this horizon.
- PPO on a 24 GB card runs out of memory partway through (steps 951–2,138).

## Ratings

- **Motivation ★★** — a natural confirmation step, not a designed sweep (two learning rates, two regularizers).
- **Knowledge: medium** — the horizon-transfer failure of the tuned lr is the lasting lesson; the PPO arm is
  incomplete.
