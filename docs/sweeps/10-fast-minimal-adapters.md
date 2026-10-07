# 10. How little has to train: o-LoRA depth and bias vectors (Qwen-2.5-3B, fast loop)

**When / where:** Oct 6, 2026. Fast loop, default schedule, 2 seeds; [`docs/fast_loop_log.md`](../fast_loop_log.md) section 9.

**Compute (estimated):** 26 runs, 2.3 GPU-hours if run one after another, about $5 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

Following [09](09-fast-adapter-ablation.md): can calibration be learned with no low-rank matrices at all, just a
trained vector added to each layer's output?

## Inputs

| what trains | layers | learning rates |
|---|---|---|
| o LoRA | all, 18–35, 27–35 | 3e-4 |
| gate-preactivation biases | all | 3e-4, 3e-3, 3e-2 |
| MLP-output bias | 18–35 | 3e-3, 1e-2, 3e-2 |
| MLP-output bias | 27–35 | 1e-2 |
| attention-output bias | 18–35, 27–35 | 1e-2 |
| attention + MLP bias | 18–35, 27–35 | 1e-2 |

## Results (step 300)

| arm | params | Brier | AUROC |
|---|---:|---|---|
| all-layer LoRA (baseline) | 15.0M | 0.120 | 0.915 |
| o LoRA 18–35 | 590k | 0.118 | 0.905 |
| gate biases, lr 3e-3 | 396k | 0.114 | 0.913 |
| gate biases, lr 3e-4 / 3e-2 | 396k | 0.151 / 0.161 | 0.851 / 0.847 |
| **attention-output bias 18–35, lr 1e-2** | **36.9k** | **0.114** | **0.908** |
| MLP-output bias 18–35, lr 1e-2 | 36.9k | 0.130 | 0.902 |
| attention + MLP bias 18–35, lr 1e-2 | 73.7k | 0.128 | 0.908 |
| anything in layers 27–35 | 18–37k | 0.166–0.203 | 0.827–0.836 |

## What we learned

- **A constant vector per layer nearly suffices on Qwen:** 36.9k attention-bias values (0.25% of LoRA's) match LoRA's
  Brier. A fixed shift improves ranking, not only calibration (AUROC 0.787 → 0.91), through later layers' nonlinearity.
- **Bias vectors want learning rates 10–100× LoRA's** (best 3e-3 to 1e-2), with a narrow window.
- Attention output beats MLP output; both together did not help. The last quarter fails for every adapter type.

## Ratings

- **Motivation ★★** — an interesting minimality question; the bias learning rates were swept only coarsely.
- **Knowledge: medium** — true on the proxy, but [11](11-answer-kl-weight.md) showed these adapters destroy the
  answers at these rates, and on Llama the same adapters were unusable ([18](18-llama-small-adapters.md)).
