# 20. LoRA on `o_proj` only (Llama-3-8B)

**When / where:** Oct 6 (night), 2026. Minimal trainer, offline, 500 steps, 2 seeds (`runs/minimal/llama-lorao*`,
`llama_lorao_configs.json`).

## Question and motivation

On Qwen, `o` alone matched all seven projections ([09](09-fast-adapter-ablation.md)). Is a cheap `o`-only LoRA
enough on Llama?

## Inputs

Layers {16–31, all} × lr {3e-4, 1e-3, 3e-3} × W {0, 0.3, 1} (not a full grid).

## Results (step 500)

| arm | Brier | AUROC | answer KL | regen accuracy |
|---|---|---|---|---|
| 16–31, lr 3e-4, W=1 | 0.149 | 0.857 | 0.007 | 0.674 |
| 16–31, lr 1e-3, W=1 | 0.141 | 0.873 | 0.17 | 0.664 |
| 16–31, lr 3e-3, W=1 | 0.289 | 0.528 | 161 | broken |
| all, lr 1e-3, W=1 | 0.321 | 0.699 | 74 | broken (best Brier 0.131 at step 350) |
| all, lr 1e-3, W=0 | 0.208 | 0.555 | 32 | 0.025 |
| all, lr 3e-3, W ∈ {0.3, 1} | 0.37–0.49 | 0.51 | 160–180 | broken |

## What we learned

- `o`-only LoRA is safe in the late half up to lr 1e-3, but reaches only Brier 0.141 / AUROC 0.873 (full LoRA
  0.136 / 0.883 at 4× the parameters).
- **On all 32 layers, lr 1e-3 destroys the answers even with W = 1:** the penalty cannot hold a fast, small adapter.
- The stability edge sits between 1e-3 and 3e-3 for this adapter.

## Ratings

- **Motivation ★** — carried a Qwen result to Llama with a loose grid; the cost question was better answered by
  late-layer LoRA ([21](21-llama-late-layers.md)).
- **Knowledge: low–medium** — confirmed "small adapters are fragile on Llama"; no adopted setting.
