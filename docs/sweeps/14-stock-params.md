# 14. Stock parameters instead of hooks: `v_proj` bias and norm gains (Qwen-2.5-3B)

**When / where:** Oct 6 (evening), 2026. Minimal trainer, adaptive 1-nat answer-KL target, 300 steps, 2 seeds;
[`docs/minimal_trainer.md`](../minimal_trainer.md) "Stock parameters" (`runs/minimal/stock-*`).

**Compute (estimated):** 18 runs, 2.1 GPU-hours if run one after another, about $5 (Modal L40S). See `scripts/sweep_costs.py`.


## Question and motivation

The attention-output bias needs a forward hook (and a patched decoding loop in Unsloth). Qwen's `v_proj` bias has the
same effect (attention weights sum to 1) as an ordinary checkpoint weight. Does it work as well?

## Inputs

`v_proj` bias (layers 18–35, or all) at lr {3e-3, 1e-2, 3e-2}; RMSNorm gains (18–35) at lr {1e-3, 3e-3}; online and
offline; the hooked bias as reference.

## Results

| arm | mode | Brier | AUROC | answer KL | worst Δ regen accuracy |
|---|---|---|---|---|---|
| hooked attention bias, lr 1e-2 | online | 0.110 | 0.915 | 1.11 | −0.002 |
| `v_proj` bias, lr 1e-2 | online | 0.116 | 0.910 | 1.27 | −0.006 |
| norm gains, lr 3e-3 | online | 0.115 | 0.913 | 0.86 | −0.020 |
| `v_proj` bias, lr 3e-2 | online | 0.134 | 0.892 | 1.28 | −0.018 |
| `v_proj` bias, lr 1e-2 | offline | 0.125 | 0.902 | 1.35 | −0.037 |
| `v_proj` bias, lr 3e-3 | offline | 0.144 | 0.879 | 1.08 | −0.041 |
| `v_proj` bias, all 36 layers, lr 1e-2 | offline | 0.219 | 0.849 | 3.91 | −0.043 |

## What we learned

- **Online, the stock `v_proj` bias nearly matches the hooked bias** with an eighth of the parameters and no hooks.
- **Online beats offline** again for small adapters.
- **"About 1 nat of answer KL is safe" is false in general:** offline `v_proj` and norm gains lost 3–4 points at
  1.1–1.35 nats. The cost per nat depends on the directions moved; accuracy has to be measured.

## Ratings

- **Motivation ★★** — an engineering simplification (no hooks) with a mechanistic argument.
- **Knowledge: medium** — a usable hook-free adapter, and the end of KL as a stand-in for accuracy.
