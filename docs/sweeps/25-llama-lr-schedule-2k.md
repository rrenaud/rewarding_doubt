# 25. Learning-rate schedule and weight averaging over 2,000 steps (Llama-3-8B)

**When / where:** Oct 7, 2026. Minimal trainer, rank-4 LoRA on layers 16–31, W = 1, offline on the 8,000-question
cache, 2,000 steps (8 passes over the data), 2 seeds, RunPod (`runs/runpod/llama-r4-2k-*`), per-example dumps.

## Question and motivation

Online runs got worse past step 1,000 and oscillated between checkpoints. Hypothesis: a constant learning rate tuned on
500-step runs leaves a noise floor; decay or averaging should let long runs settle.

## Inputs

{constant 3e-4, constant 1e-4, cosine 3e-4 → 0}; every run also evaluates the average of its weights over steps
1,000–2,000 (`--weight-avg-start 1000`).

## Results

Trainer's dev evaluation, mean of 2 seeds:

| arm | best region (Brier / AUROC) | step 2,000 | weight average, steps 1,000–2,000 |
|---|---|---|---|
| constant 3e-4 | step 1,000: 0.125 / 0.895 | 0.153 / 0.883 (answer KL 0.38) | 0.130 / 0.896 (KL 0.18) |
| constant 1e-4 | steps 1,000–1,500: 0.126 / 0.89 | 0.152 / 0.877 (KL 0.04) | **0.127 / 0.897** (KL 0.015) |
| cosine 3e-4 | step 500: 0.129 / 0.892 | 0.170 / 0.879 | 0.150 / 0.886 |

Per-example dumps at step 2,000 (exact-match labels): train Brier 0.029–0.071 against dev 0.165–0.181; at 500 steps
the two were equal (0.13–0.15).

## What we learned

- **The hypothesis was wrong; the runs overfit.** By step 2,000 each run has nearly memorized which of its 8,000
  training answers are correct. Even cosine, whose lr reaches 0, gets worse late.
- **Cosine overfits most** (its clean late steps fit the training answers fastest).
- **Weight averaging acts like early stopping** and recovers most of the peak.
- This also explains the online paper runs ([22](22-llama-paper-comparison.md)): they re-ask the same 8,000
  questions. The lever is more distinct questions: the full training split (87,622) was cached next.

## Ratings

- **Motivation ★★★** — a specific hypothesis with a planned test (schedule, rate, averaging) and per-example dumps.
- **Knowledge: high** — falsified the hypothesis and found the real bottleneck (data repetition).
