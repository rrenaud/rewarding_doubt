# 09. Which LoRA adapters are needed: components and depth (Qwen-2.5-3B, fast loop)

**When / where:** Oct 6, 2026. Fast loop, default schedule (Adam 3e-4, 300 steps), 2 seeds per arm (baseline 3);
[`docs/fast_loop_log.md`](../fast_loop_log.md) sections 6–8.

## Question and motivation

Rank-8 LoRA on all 7 projections and 36 layers (15M parameters) may be far more than calibration needs. Which parts
matter, and can a partial adapter be cheaper?

## Inputs

Components (all layers): all 7; MLP (gate, up, down); attention (q, k, v, o); o + down; q + v; each of gate, up, down,
o, q, v, k alone. Depth (all 7 projections): layers 0–35, 0–17, 18–35, 27–35, 32–35.

## Results (step 300)

| adapters | params | Brier | AUROC |
|---|---:|---|---|
| all 7, all layers | 15.0M | 0.120 | 0.915 |
| gate, up, down | 11.3M | 0.113 | 0.913 |
| o alone | 1.2M | 0.115 | 0.912 |
| down alone | 3.8M | 0.121 | 0.907 |
| q alone / v alone / k alone | 0.7–1.2M | 0.129 / 0.129 / 0.145 | 0.887 / 0.897 / 0.870 |
| layers 0–17 / 18–35 | 7.5M | 0.117 / 0.117 | 0.908 / 0.917 |
| layers 27–35 | 3.7M | 0.152 | 0.855 |
| layers 32–35 | 1.7M | 0.233 | 0.826 |

## What we learned

- **One projection that writes into the residual stream (o, or any MLP projection) is enough;** q, k, v alone lag.
- **Either half of the network is enough; the last quarter is not.** Placement matters more than size: the last
  4 layers (1.7M) fail where q + v everywhere (1.8M) matches the baseline.
- Smaller adapters are slower to converge at a shared lr, not worse by step 300.
- Speed: partial adapters only got cheaper after cutting the backward pass at the first trained layer and turning
  gradient checkpointing off (found here: Unsloth re-enabled it); layers 18–35 then cost 28% less per step.

## Ratings

- **Motivation ★★** — a sensible capacity/speed question, run as a broad ablation.
- **Knowledge: high** — set the late-half adapters used later on Llama ([21](21-llama-late-layers.md)), and the
  "placement over size" rule held up. Found two speed bugs.

## Caveats

Frozen cached answers: the ablation measured calibration, not what the adapters did to the answers
([11](11-answer-kl-weight.md)).
