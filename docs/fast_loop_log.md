# Experiment log: the fast approximate loop

October 5–6, 2026 (times PST). Qwen-2.5-3B, exact objective, `modal_repro/fast_loop.py`. Every run is on
one L40S on Modal; per-run outputs (`train.log`, `metrics.jsonl`, `curve.json`) and the config files
that launched them are in `runs/fast/`.

**Outcome.** A training experiment now takes about 5 minutes instead of 48: one Adam update per 32
length-bucketed questions at learning rate 3e-4, for 300 steps. That reaches dev Brier 0.120 ± 0.004
and AUROC 0.915 ± 0.005 (3 seeds), better than 1,000 steps of the released schedule (0.133 / 0.877).
Most of the gain is in by step 150. The speedup comes from computing logits only where they are
read, turning gradient checkpointing off, and replacing many small updates with fewer large ones.
Muon, Scaled AdamW and PoLoRA were tried as optimizers and did not beat Adam at a matched learning
rate; only Adam is kept in the code.

## Goal

Iterate on confidence objectives in minutes rather than hours, at the cost of some fidelity: no
generation during training, answers frozen at the base model's, dev scored from one forward pass.

## Setup

- **Cache** (`fast_loop.py cache`, built October 5, 22:27 PST): the base model answers 8,400 random
  TriviaQA training questions once with the released prompt and sampling (T = 0.6, top-p 0.9, stop at
  " Confidence"). 8,000 answers are kept, plus the 507 dev answers. Each stores its tokens, F1 and
  exact-match grades, and the reference model's log-probabilities of the 11 confidence levels.
  Training accuracy: exact match 0.324, F1 0.373. Dev accuracy (F1 > 0.5): 0.391.
- **Training**: the exact objective (released reward, −30 on the leftover mass, exact KL to the cached
  reference) with a fixed KL weight of 0.05 unless noted. Labels are exact match, the released default.
- **Dev metrics**: the confidence on each cached dev answer is the expected level Σ k·q(k) of the level
  distribution q; Brier, 11-bin ECE and AUROC as in the released evaluation. Untrained model: Brier
  0.566, ECE 0.573, AUROC 0.787.
- **Proxy check** (commit 7197c17): on 18 snapshots from earlier full runs, these fast metrics
  correlate with the full released-protocol dev scores at r = 0.75 (Brier) and r = 0.76 (AUROC),
  Spearman 0.72 / 0.80, but only about 0.5 for ECE. Method rankings were preserved and gaps
  compressed; PPO's ECE is understated on frozen answers. **Track Brier and AUROC; treat ECE as
  secondary.** ECE swings by 0.05–0.15 between evaluations of the same run throughout this log.

## 1. First runs: the released schedule (Oct 5, 22:28 PST)

Batches of 8 questions, 4 passes in minibatches of 4 (8 Adam updates per step), learning rate 1e-5,
as in the released PPO config.

| run | step 250 | step 1,000 | time for 1,000 steps |
|---|---|---|---:|
| adaptive KL (target 6) | 0.153 / 0.038 / 0.843 | 0.137 / 0.037 / 0.874 | 45.8 min |
| fixed KL 0.05 | 0.159 / 0.035 / 0.845 | 0.133 / 0.025 / 0.877 | 48.1 min |

Cells are Brier / ECE / AUROC. About 2.9 s per step, with the GPU at 35% utilization: far too slow
for a "fast" loop, even though nothing is generated.

## 2. Why it was slow (Oct 5–6)

A profiler (`fast_loop.py profile`, Modal entrypoint `fast_profile`) timed forward, loss, backward
and optimizer per question-update on cached rows (mean 164 tokens; 132 tokens are a prompt prefix
shared by every row).

| path | mb 4 | mb 8 | mb 16 | mb 32 |
|---|---:|---:|---:|---:|
| full-vocabulary logits at every position (before) | 84.9 | 50.0 | 50.9 | 75.1 |
| logits only at the read positions, checkpointing on | – | 42.6 | 28.5 | 29.6 |
| logits only at the read positions, checkpointing off | – | 29.5 | **20.6** | 21.5 |

Milliseconds per question-update. Peak memory at mb 16 with checkpointing off: 10.3 GB.

Three causes:

1. **Logits over the whole vocabulary (152k tokens) at every position.** The level probabilities read
   3 positions. The backward pass through the full logits grew with minibatch size. Fix:
   `level_logps` runs the inner model and applies the output head only at the read positions. Its
   level log-probabilities match `LevelScheme.batch` exactly (max difference 0.0).
2. **Overhead-bound small minibatches.** At minibatch 4, the CPU spent 2.6 s issuing kernels against
   0.55 s of GPU work. A step was 32 question-updates at about 85 ms each, which accounts for the
   2.9 s.
3. **Gradient checkpointing forced on.** Unsloth enables it even with `use_gradient_checkpointing=False`,
   so backward recomputes the forward pass. Fix: switched off on every module (`set_gradient_checkpointing`).

Two bugs found on the way, both from calling Unsloth's inner model directly:
- Without the `causal_mask` that Unsloth's `CausalLM` forward passes (xformers `LowerTriangularMask`),
  attention was bidirectional. Levels were off by up to 23 nats until the mask was passed.
- The inner model reads `_has_no_labels`, which only the outer forward sets; the first training run
  crashed on it.

**Check on the released schedule** (fixed KL 0.05, 250 steps): 11.3 min vs 11.9 before; Brier 0.155 vs
0.159, AUROC 0.849 vs 0.845. Barely faster: minibatches of 4 stay overhead-bound, so the schedule has
to change too.

## 3. Larger updates (Oct 6)

One pass, one Adam update per batch, random batches (no length bucketing yet), fixed KL 0.05.

| batch, lr | steps | step 150 | final | time |
|---|---:|---|---|---:|
| 32, 1e-5 | 300 | 0.448 / 0.460 / 0.782 | 0.183 / 0.037 / 0.788 | 5.5 min |
| 32, 3e-5 | 300 | 0.187 / 0.104 / 0.809 | 0.155 / 0.056 / 0.842 | 5.5 min |
| 32, 1e-4 | 300 | 0.173 / 0.151 / 0.878 | **0.117 / 0.035 / 0.905** | 5.4 min |
| 16, 3e-5 | 600 | – | 0.140 / 0.058 / 0.871 | 5.1 min |

With 8× fewer updates per question, the learning rate has to rise. Batch 32 at 1e-4 beat the
48-minute released-schedule run in 5.4 minutes (one seed).

## 4. Length bucketing (Oct 6)

Rows are right-padded to the longest in their batch. Over the 8,000 training rows (151 / 164 / 180 /
271 tokens at min / median / p90 / max), random batches of 32 are 16.9% padding.

- **Training on only the shortest answers was rejected.** Exact-match accuracy by length quartile,
  shortest first: 0.455, 0.323, 0.270, 0.248. The shortest half would shift the base rate the
  confidence learns, for only about 5% less compute (the 132-token shared prefix dominates).
- **Bucketing keeps the data and removes the padding.** Each epoch is shuffled, cut into windows of
  64 batches, sorted by length within each window, cut into batches, and the batch order is
  shuffled (`epoch_batches`, `--bucket-window`, default 64; 0 turns it off). Padding: 0.2%. Side
  effect: each batch is similar in length and therefore in accuracy.

Batch 32, lr 1e-4, 300 steps: 4.7 min bucketed vs 5.4 min random; 0.122 / 0.056 / 0.908 vs
0.117 / 0.035 / 0.905 (one seed each, within seed noise).

## 5. Optimizers (Oct 6)

The released code, and so the fast loop, uses plain Adam (`torch.optim.Adam`: betas 0.9 / 0.999, eps
1e-8, no weight decay, no schedule), as TRL 0.8.6's `PPOTrainer` builds it. The trainable parameters
are the rank-8 LoRA matrices only, all 2-D. All runs below: batch 32, one pass, bucketed, fixed KL
0.05, 300 steps.

### Muon on each LoRA factor

Implementation (since removed): Nesterov momentum 0.95, five quintic Newton–Schulz iterations to
orthogonalize each factor's update, scaled by 0.2·√max(rows, cols) (Moonlight's rule, so learning
rates are comparable with Adam's). Written by hand because the pinned PyTorch predates
`torch.optim.Muon`.

| lr | step 150 | step 300 |
|---|---|---|
| 3e-5 | 0.198 / 0.110 / 0.777 | 0.153 / 0.056 / 0.845 |
| 1e-4 | 0.134 / 0.066 / 0.884 | 0.117 / 0.042 / 0.901 |
| 3e-4 | 0.115 / 0.039 / 0.904 | collapsed: 0.314 / 0.277 / 0.551 |
| 1e-3 | collapsed from step 50 | 0.326 / 0.300 / 0.543 |

At 1e-4 Muon tied Adam at 1e-4; it was 30% slower per step (a Python loop over 252 matrices). At
3e-4 it was best at step 150 and then collapsed. At step 200, ECE was 0.013 with AUROC 0.566: one
confidence for every answer, calibrated on average and uninformative. ECE alone would have ranked
it first.

### LoRA-aware optimizers

Muon (and Adam) applied to A and B separately depend on the factorization: (cA, B/c) gives the same
B·A but a different step. Literature consulted:
- *Can Muon Fine-tune Adam-Pretrained Models?* ([arXiv 2605.10468](https://arxiv.org/pdf/2605.10468),
  ICML 2026): full fine-tuning with Muon suffers an optimizer mismatch on Adam-pretrained models;
  LoRA-Muon matches or beats LoRA-Adam, especially at low rank.
- PoLoRA ([arXiv 2607.17620](https://arxiv.org/abs/2607.17620), July 2026): per-factor Muon does not
  consistently beat Adam; PoLoRA reaches tuned Adam's loss in 1.2–1.7× fewer steps (1B–8B models,
  code and math instruction tuning).
- LoRA meets Riemannion ([arXiv 2507.12142](https://arxiv.org/abs/2507.12142v2), ICLR 2026): a Muon
  generalization on the fixed-rank manifold (not tried).

Two were implemented (since removed), with CPU tests on a toy low-rank regression:

- **Scaled AdamW** (Zhang & Pilanci, ICML 2024, their Algorithm 1): A's gradient multiplied by
  (BᵀB + 10⁻⁶I)⁻¹ and B's by (AAᵀ + 10⁻⁶I)⁻¹, then ordinary Adam moments on the preconditioned
  gradients.
- **PoLoRA** (Algorithm 1; β1 0.9, β2 0.99, relative damping 10⁻⁴, ε 10⁻¹²): momentum with look-ahead;
  diagonal curvature P (outputs) and Q (inputs) fitted to the factor gradients and normalized to a
  largest entry of 1; product-aware directions D_A = C_B^-½ msign(C_B^-½ M_A Q^-½) Q^-½ and
  D_B = P^-½ msign(P^-½ M_B C_A^-½) C_A^-½ with C_B = BᵀPB, C_A = AQAᵀ; each factor stepped by
  ρ = η / (‖A‖₂ + ‖B‖₂), which bounds the linearized merged update by η in spectral norm. The paper
  computes msign and C^-½ with Gram Newton–Schulz; at rank 8 they are 8×8 problems, so the
  implementation used exact batched `eigh`, with the 252 LoRA pairs grouped by shape. Neither added
  measurable time per step.

On the toy problem, a target of lower rank than the adapter (rank 2 vs rank 4) made PoLoRA crawl:
spectral steps move every singular direction equally, so the spare directions are pushed and then
undone. With a rank-6 target both optimizers converged to near the rank-4 limit.

One seed each:

| optimizer, lr | step 150 | step 300 |
|---|---|---|
| Scaled AdamW 3e-5 | 0.179 / 0.035 / 0.792 | 0.163 / 0.053 / 0.821 |
| Scaled AdamW 1e-4 | 0.162 / 0.084 / 0.857 | 0.142 / 0.070 / 0.886 |
| Scaled AdamW 3e-4 | 0.118 / 0.049 / 0.910 | 0.116 / 0.031 / 0.915 |
| PoLoRA 3e-3 | 0.121 / 0.045 / 0.902 | 0.110 / 0.032 / 0.917 |
| PoLoRA 1e-2 (paper's optimum) | 0.111 / 0.029 / 0.908 | 0.135 / 0.062 / 0.906 |
| PoLoRA 3e-2 | 0.136 / 0.050 / 0.888 | 0.123 / 0.035 / 0.886 |

Both looked better than Adam at 1e-4, but Adam had not been run above 1e-4.

### Matched learning rates, 3 seeds

| arm | Brier @150 | AUROC @150 | Brier @300 | ECE @300 | AUROC @300 | Brier, mean of steps 150–300 |
|---|---|---|---|---|---|---|
| **Adam 3e-4** | **0.116 ± 0.006** | **0.907 ± 0.007** | **0.120 ± 0.004** | **0.040 ± 0.014** | **0.915 ± 0.005** | 0.125 ± 0.009 |
| PoLoRA 3e-3 | 0.124 ± 0.009 | 0.904 ± 0.006 | 0.124 ± 0.018 | 0.068 ± 0.039 | 0.910 ± 0.005 | **0.125 ± 0.001** |
| Scaled AdamW 3e-4 | 0.126 ± 0.017 | 0.904 ± 0.005 | 0.139 ± 0.029 | 0.072 ± 0.045 | 0.905 ± 0.012 | 0.129 ± 0.012 |
| Adam 1e-3 | 0.176 ± 0.043 | 0.858 ± 0.033 | 0.202 ± 0.062 | 0.040 ± 0.026 | 0.731 ± 0.150 | 0.196 ± 0.047 |
| Scaled AdamW 1e-3 | 0.198 ± 0.076 | 0.669 ± 0.213 | 0.208 ± 0.059 | 0.064 ± 0.030 | 0.669 ± 0.204 | 0.211 ± 0.054 |

Mean ± sd over seeds 1–3; 4.6–4.9 min per run for every arm.

- **Adam at 3e-4 is best or tied on every metric.** The earlier advantage of Scaled AdamW and PoLoRA
  came from the higher learning rate.
- **PoLoRA is the most consistent across seeds** (late-Brier spread ±0.001 against ±0.009) but not
  better on average.
- **1e-3 is past the edge of stability** for Adam and Scaled AdamW: 2 of 3 seeds of each ended with
  AUROC 0.51–0.71, a near-constant confidence.

This matches PoLoRA's own finding for per-factor Muon. Our objective, a small 11-way distribution
anchored by a KL term, may simply be insensitive to the optimizer beyond its step size. The
optimizer code was removed; the descriptions above are the record.

## Defaults now in `fast_loop.py train`

`--batchsize 32 --passes 1 --minibatch 32 --lr 3e-4 --steps 300 --eval-every 50 --bucket-window 64`,
Adam, gradient checkpointing off. About 4.7 minutes on an L40S, including 7 dev evaluations. The
released schedule is `--batchsize 8 --passes 4 --minibatch 4 --lr 1e-5`.

## Caveats and open items

- **The new schedule is not yet validated against the full evaluation.** The proxy check used
  snapshots trained with the released schedule. Scoring a few of these final adapters with the full
  released protocol would confirm that the fast metrics still rank them correctly.
- Answers are frozen at the base model's, and dev is 507 answers; ECE below about 0.04 is not resolved.
- Most comparisons above are one seed; only section 5's last table has three.
- Not tried: computing the 132-token shared prefix once per minibatch (up to about 5× less forward
  compute; needs plain Hugging Face code, since Unsloth's training forward cannot take a cached
  prefix), and Riemannion.

## Reproducing

```bash
modal run modal_repro/app.py::fast_profile                                          # timing breakdown
modal run modal_repro/app.py::fast_train --configs runs/fast/seed_sweep_configs.json
```

Every config file in `runs/fast/` spells out its schedule. `first_configs.json`,
`speed_configs.json` and `schedule_configs.json` pass `--bucket-window 0`, since they ran before
bucketing. `muon_configs.json`, `lora_optim_configs.json` and the non-Adam arms of
`seed_sweep_configs.json` pass an `--optimizer` flag that no longer exists; they record what was
run but no longer launch.
