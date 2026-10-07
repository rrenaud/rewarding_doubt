# 18. Small adapters on Llama: bias vectors and norm gains (Llama-3-8B, minimal trainer)

**When / where:** Oct 6 (night), 2026. Minimal trainer on the Llama cache (8,000 questions, bf16), offline, 500 steps,
layers 16–31, 2 seeds (`runs/minimal/llama-*`, configs `llama_configs.json`, `llama_drift_configs.json`); KL vs
accuracy plot `docs/answer_kl_vs_accuracy_llama.png`.

**Compute (estimated):** 26 runs, 3.1 GPU-hours if run one after another, about $7 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

Port Qwen's best small adapters ([14](14-stock-params.md), [17](17-weight-floor.md)) to Llama at Qwen's learning
rates, and measure how answer drift costs accuracy there (unpenalized "drift" runs).

## Inputs

Penalized (target 1.5, floor 0.1): `o_proj` bias lr {3e-3, 1e-2}; hooked attention bias lr 1e-2; norm gains lr 3e-3.
Unpenalized: `o_proj` bias lr {3e-4, 1e-3}; norm gains lr {3e-3, 1e-2}.

## Results (step 500; base regenerated accuracy 0.669)

| arm | Brier | AUROC | answer KL | regen accuracy |
|---|---|---|---|---|
| `o_proj` bias, lr 3e-3, penalized | 0.180 | 0.795 | 4.9 | 0.534 |
| `o_proj` bias, lr 1e-2, penalized | 0.228 | 0.780 | 10.9 | 0.418 |
| hooked attention bias, lr 1e-2, penalized | 0.193 | 0.774 | 17.7 | 0.403 |
| norm gains, lr 3e-3, penalized | 0.140 | 0.870 | 0.24 | 0.651 |
| `o_proj` bias, lr 3e-4, unpenalized | 0.160 | 0.832 | 0.93 | 0.610 |
| `o_proj` bias, lr 1e-3, unpenalized | 0.165 | 0.839 | 19.5 | 0.262 |
| norm gains, lr 3e-3, unpenalized | 0.143 | 0.870 | 0.51 | 0.639 |
| norm gains, lr 1e-2, unpenalized | 0.133 | 0.881 | 16.2 | 0.295 |

## What we learned

- **Qwen's learning rates do not transfer to Llama:** the bias adapters wreck Llama's answers within 100 steps even
  with the penalty (the floor-0.1 weight cannot hold them).
- **Llama loses accuracy from the first tenths of a nat** with small adapters (about −2 points at 0.1–0.3 nats, −4 at
  0.5–1, −7 at 1–2), where Qwen was flat up to about 2 nats. The base model's seed-to-seed spread is ±0.007, so
  this is not noise.
- Norm gains are the least harmful small adapter on Llama (−2 points at 0.24 nats).

## Ratings

- **Motivation ★★** — a natural port, but at untested learning rates; it became a drift study.
- **Knowledge: high** — the "a nat costs more on Llama" finding redirected the Llama work to LoRA ([19](19-llama-full-lora-lr.md)).
