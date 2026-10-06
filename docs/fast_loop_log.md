# Experiment log: the fast approximate loop

October 5–6, 2026 (times PST). Qwen-2.5-3B, exact objective, `modal_repro/fast_loop.py`. Every run is on
one L40S on Modal; per-run outputs (`train.log`, `metrics.jsonl`, `curve.json`) and the config files
that launched them are in `runs/fast/`.

**Outcome.** A training experiment now takes about 5 minutes instead of 48: one Adam update per 32
length-bucketed questions at learning rate 3e-4, for 300 steps. That reaches dev Brier 0.120 ± 0.004
and AUROC 0.915 ± 0.005 (3 seeds), better than 1,000 steps of the released schedule (0.133 / 0.877).
Most of the gain is in by step 150. The speedup comes from computing logits only where they are
read and replacing many small updates with fewer large ones. Gradient checkpointing was meant to
be off as well, but every run until section 8 trained with it on; with it off, a step is another
28% faster (section 8).
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
   *Correction (section 8):* the dev scoring at step 0 turned it back on, so training runs kept it
   on until that was fixed; the profiler's numbers above are with it off.

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

## 6. Which adapters are needed (Oct 6)

`--lora-modules` (any of q, k, v, o, gate, up, down) and `--lora-layers` (an inclusive range of
Qwen-2.5-3B's 36 decoder layers) choose which LoRA adapters train. Every module still carries an
adapter; the others are frozen with B = 0, so they leave the model unchanged. Default schedule,
2 seeds per arm (mean ± sd); the baseline is the 3 seeds of Adam 3e-4 from section 5.

**Components, all layers:**

| adapters | trainable params | Brier @150 | AUROC @150 | Brier @300 | ECE @300 | AUROC @300 |
|---|---:|---|---|---|---|---|
| all 7 (baseline) | 15.0M | 0.116 ± 0.006 | 0.907 ± 0.007 | 0.120 ± 0.004 | 0.040 ± 0.014 | 0.915 ± 0.005 |
| gate, up, down | 11.3M | 0.121 ± 0.011 | 0.908 ± 0.001 | 0.113 ± 0.002 | 0.039 ± 0.014 | 0.913 ± 0.001 |
| q, k, v, o | 3.7M | 0.144 ± 0.029 | 0.892 ± 0.003 | 0.122 ± 0.006 | 0.048 ± 0.025 | 0.916 ± 0.001 |
| o, down | 4.9M | 0.136 ± 0.018 | 0.897 ± 0.002 | 0.118 ± 0.016 | 0.050 ± 0.012 | 0.911 ± 0.008 |
| q, v | 1.8M | 0.136 ± 0.002 | 0.885 ± 0.003 | 0.120 ± 0.003 | 0.048 ± 0.025 | 0.907 ± 0.003 |
| gate | 3.8M | 0.137 ± 0.009 | 0.895 ± 0.006 | 0.117 ± 0.005 | 0.043 ± 0.013 | 0.907 ± 0.005 |
| up | 3.8M | 0.144 ± 0.023 | 0.900 ± 0.002 | 0.118 ± 0.006 | 0.059 ± 0.021 | 0.913 ± 0.000 |
| down | 3.8M | 0.133 ± 0.003 | 0.893 ± 0.000 | 0.121 ± 0.018 | 0.064 ± 0.014 | 0.907 ± 0.008 |
| o | 1.2M | 0.145 ± 0.000 | 0.890 ± 0.005 | 0.115 ± 0.001 | 0.032 ± 0.019 | 0.912 ± 0.001 |
| q | 1.2M | 0.167 ± 0.005 | 0.855 ± 0.006 | 0.129 ± 0.002 | 0.044 ± 0.003 | 0.887 ± 0.003 |
| v | 0.7M | 0.163 ± 0.005 | 0.867 ± 0.006 | 0.129 ± 0.002 | 0.059 ± 0.009 | 0.897 ± 0.002 |
| k | 0.7M | 0.161 ± 0.009 | 0.835 ± 0.002 | 0.145 ± 0.001 | 0.052 ± 0.002 | 0.870 ± 0.001 |

**Depth, all 7 projections:**

| layers | trainable params | Brier @150 | AUROC @150 | Brier @300 | ECE @300 | AUROC @300 |
|---|---:|---|---|---|---|---|
| 0–35 (baseline) | 15.0M | 0.116 ± 0.006 | 0.907 ± 0.007 | 0.120 ± 0.004 | 0.040 ± 0.014 | 0.915 ± 0.005 |
| 0–17 | 7.5M | 0.130 ± 0.001 | 0.891 ± 0.009 | 0.117 ± 0.005 | 0.034 ± 0.010 | 0.908 ± 0.004 |
| 18–35 | 7.5M | 0.140 ± 0.009 | 0.896 ± 0.001 | 0.117 ± 0.007 | 0.067 ± 0.032 | 0.917 ± 0.003 |
| 27–35 | 3.7M | 0.167 ± 0.011 | 0.818 ± 0.025 | 0.152 ± 0.005 | 0.042 ± 0.018 | 0.855 ± 0.010 |
| 32–35 | 1.7M | 0.243 ± 0.001 | 0.805 ± 0.013 | 0.233 ± 0.002 | 0.189 ± 0.008 | 0.826 ± 0.002 |

- **A single projection is enough if it writes into the residual stream.** The attention output (o,
  1.2M parameters, 8% of the baseline) matches all seven at step 300, and so does any single MLP
  projection. q, k or v alone lag; k is weakest. q and o have the same size: 0.129 / 0.887 against
  0.115 / 0.912.
- **Smaller adapters are slower, not worse,** at the shared learning rate: every subset trails at
  step 150 and catches up by step 300, except q, k, v alone and the late-layer arms.
- **Either half of the network is enough; the last layers alone are not.** Layers 0–17 or 18–35
  match the baseline. The last 9 layers end at 0.152 / 0.855, and the last 4 barely move (Brier near
  0.24 at both steps 150 and 300). The last-4 arm (1.66M parameters) and q, v in every layer (1.84M)
  are the same size: one fails, the other matches the baseline. Placement matters, not size.
- Not checked: whether a higher learning rate closes the gap for the late-layer arms.
- These runs took 4.3–4.7 min each, the same as the baseline: frozen adapters still run, and (found
  later, section 7) the backward pass still went through every layer.

## 7. Making partial adapters cheaper (Oct 6)

Section 6's arms all cost the same per step, whatever they trained. `fast_loop.py profile-lora`
(Modal entrypoint `fast_profile_lora`) times one update at minibatch 32 with adapters trained in all
layers, layers 18–35 or layers 27–35:

| layers trained | backward | forward ms / q | backward ms / q | peak memory |
|---|---|---:|---:|---:|
| all 36 | full | 7.2 | 9.6 | 16.0 GB |
| 18–35 | full | 7.2 | 9.6 | 15.6 GB |
| 18–35 | stops at layer 18 | 7.2 | **4.9** | **9.1 GB** |
| 27–35 | full | 7.3 | 9.6 | 15.4 GB |
| 27–35 | stops at layer 27 | 7.2 | **2.5** | 11.9 GB |

- **Unsloth sends the backward pass through every layer.** Its attention picks a grouped-query
  layout with no backward kernel (xformers) when the incoming hidden state does not require grad,
  so it makes every layer's input require grad. The output of layer 0 requires grad even with all
  of layers 0–17 frozen and the input-embedding hook removed, and backward then runs through all
  36 layers. Detaching the hidden state below the first trained layer crashes for the same reason:
  `No operator found for memory_efficient_attention_backward ... does not support BMGHK format`.
- **Fix: cut the graph but keep the flag.** `cut_backward_below` replaces the input of the first
  trained layer with `detach().requires_grad_(True)`, so Unsloth keeps its trainable layout and
  backward stops there. `train` applies it whenever `--lora-layers` starts above 0. It is exact:
  nothing below the cut trains.
- The forward pass is unchanged: every layer still runs, frozen adapters included.
- Cutting at 27 saves less memory than at 18: the 27 layers below still build a graph during the
  forward pass (Unsloth's forced flag) until it is dropped at the cut.

A training run with the cut (`cut-layers18-35-s1`) took the same 4.6 minutes as without it. The
reason is in section 8.

## 8. Gradient checkpointing was back on during training (Oct 6)

`--time-steps` on 60-step runs showed backward at 575 ms per step of 32 questions, 18 ms per question,
where the profiler measured 9.6 with checkpointing off and about 20 with it on. `score_dev` calls
Unsloth's `FastLanguageModel.for_training`, which turns gradient checkpointing back on, and it runs at
step 0 before any update. Every training run since section 2 therefore trained with checkpointing
on. Results are unaffected (checkpointing is exact); time was not. `score_dev` now switches it off
again.

| run (32 questions per step) | forward | loss | backward | optimizer | ms per step |
|---|---:|---:|---:|---:|---:|
| all layers, checkpointing on (all earlier runs) | 245 | 20 | 575 | 5 | 845 |
| all layers, fixed | 252 | 15 | 339 | 5 | **611** |
| layers 18–35 with the cut, fixed | 248 | 14 | 177 | 3 | **442** |

Mean over steps 11–60 (`runs/fast/timing-*`). With checkpointing on, the cut of section 7 saved
nothing measurable; with it off, training only the last half is 28% cheaper per step than all
layers.

**Training is not bitwise reproducible.** Step 1 matches to every printed digit across runs with
the same seed; from step 2 on, runs drift by similar amounts whether they differ by the cut, by
checkpointing, or by nothing, which points at nondeterministic GPU kernels in the backward pass
amplified at lr 3e-4. Seed-to-seed spread is the yardstick, as in the tables above.

## 9. How little has to train (Oct 6)

Building on section 6: the o projection in fewer layers, and adapters with no low-rank matrices
at all. `--gate-bias` trains a vector added to every MLP gate's pre-activation,
down(silu(gate(x) + b) · up(x)) (36 × 11,008 = 396k values; those MLPs get a plain forward, since
Unsloth's fused MLP kernel ignores biases). `--residual-bias A-B` adds a trained vector to each MLP
output and `--attn-bias A-B` to each attention output, that is to the residual stream, in layers
A–B (2,048 values per layer). All start at zero, so training starts from the base model;
`--lora-modules none` freezes every LoRA adapter. The backward pass stops at the first layer where
anything trains (section 7). 2 seeds per arm, mean ± sd.

| what trains | layers | params | lr | Brier @150 | AUROC @150 | Brier @300 | ECE @300 | AUROC @300 | time |
|---|---|---:|---|---|---|---|---|---|---:|
| all 7 LoRA (baseline, 3 seeds) | all | 15.0M | 3e-4 | 0.116 ± 0.006 | 0.907 ± 0.007 | 0.120 ± 0.004 | 0.040 ± 0.014 | 0.915 ± 0.005 | 4.7 min |
| o LoRA | all | 1.2M | 3e-4 | 0.145 ± 0.000 | 0.890 ± 0.005 | 0.115 ± 0.001 | 0.032 ± 0.019 | 0.912 ± 0.001 | 4.7 min |
| o LoRA | 18–35 | 590k | 3e-4 | 0.153 ± 0.011 | 0.866 ± 0.001 | 0.118 ± 0.004 | 0.044 ± 0.010 | 0.905 ± 0.001 | 2.6 min |
| o LoRA | 27–35 | 295k | 3e-4 | 0.206 ± 0.044 | 0.828 ± 0.008 | 0.166 ± 0.008 | 0.076 ± 0.039 | 0.836 ± 0.001 | 2.2 min |
| gate biases | all | 396k | 3e-4 | 0.172 ± 0.005 | 0.811 ± 0.000 | 0.151 ± 0.003 | 0.043 ± 0.002 | 0.851 ± 0.009 | 4.0 min |
| gate biases | all | 396k | 3e-3 | 0.126 ± 0.005 | 0.896 ± 0.005 | **0.114 ± 0.001** | 0.036 ± 0.009 | **0.913 ± 0.001** | 4.0 min |
| gate biases | all | 396k | 3e-2 | 0.134 ± 0.015 | 0.892 ± 0.013 | 0.161 ± 0.005 | 0.078 ± 0.006 | 0.847 ± 0.003 | 4.0 min |
| MLP-output bias | 18–35 | 36.9k | 3e-3 | 0.161 ± 0.020 | 0.866 ± 0.001 | 0.135 ± 0.007 | 0.055 ± 0.000 | 0.889 ± 0.003 | 2.5 min |
| MLP-output bias | 18–35 | 36.9k | 1e-2 | 0.148 ± 0.030 | 0.878 ± 0.023 | 0.130 ± 0.018 | 0.065 ± 0.044 | 0.902 ± 0.005 | 2.7 min |
| MLP-output bias | 18–35 | 36.9k | 3e-2 | 0.145 ± 0.028 | 0.887 ± 0.006 | 0.138 ± 0.007 | 0.077 ± 0.038 | 0.908 ± 0.007 | 2.7 min |
| attention-output bias | 18–35 | 36.9k | 1e-2 | 0.147 ± 0.034 | 0.891 ± 0.008 | **0.114 ± 0.002** | 0.050 ± 0.022 | **0.908 ± 0.001** | 2.7 min |
| attention + MLP bias | 18–35 | 73.7k | 1e-2 | 0.144 ± 0.033 | 0.899 ± 0.004 | 0.128 ± 0.007 | 0.074 ± 0.034 | 0.908 ± 0.008 | 2.7 min |
| MLP-output bias | 27–35 | 18.4k | 1e-2 | 0.203 ± 0.045 | 0.819 ± 0.006 | 0.203 ± 0.044 | 0.125 ± 0.049 | 0.827 ± 0.000 | 2.2 min |
| attention-output bias | 27–35 | 18.4k | 1e-2 | 0.194 ± 0.023 | 0.828 ± 0.002 | 0.169 ± 0.007 | 0.062 ± 0.060 | 0.836 ± 0.004 | 2.2 min |
| attention + MLP bias | 27–35 | 36.9k | 1e-2 | 0.216 ± 0.034 | 0.828 ± 0.002 | 0.182 ± 0.029 | 0.088 ± 0.078 | 0.833 ± 0.001 | 2.2 min |

- **A constant vector per layer is nearly enough.** One trained vector on each attention output
  in layers 18–35 (36,864 values, 0.25% of the LoRA parameters) matches all-layer LoRA on Brier at
  step 300 (0.114 against 0.120), with AUROC 0.908 against 0.915. Gate biases (396k) match it on
  everything.
- **A fixed shift improves ranking, not only calibration.** The same vector is added for every
  question, yet AUROC rises from 0.787 to about 0.91: the shift acts through the later layers,
  whose response depends on the input.
- **Attention output beats MLP output in the same layers** (0.114 ± 0.002 against 0.130 ± 0.018
  Brier), as o was the best single LoRA projection in section 6. Both together did not help at
  lr 1e-2 (not tuned).
- **The last quarter fails for every kind of adapter:** layers 27–35 reach AUROC 0.83–0.86
  whatever trains there. What has to change lies earlier.
- Seed 1 spikes at step 200 in most arms (Brier up to 0.17–0.22, then back): batch order depends
  only on the seed, so it is a hard batch rather than any one method's instability.

## 10. The fast loop's adapters break the answers (Oct 6)

The fast loop scores frozen cached answers, so it never looked at what training does to the answers
themselves, although every adapter acts at every token. A 3,000-step attention-bias run of the released
pipeline with policy-generated answers dropped to 0.315 dev accuracy by step 500 (about 0.40 for LoRA runs at
lr 1e-5). Two measurements and a penalty, all from the forward pass that scores the confidence:

- **answer_kl** (every dev evaluation): KL(policy ‖ reference) over the vocabulary at each token of the
  cached dev answers, summed per answer, with the reference = the same model under `disable_adapter()`
  (LoRA and attention biases off; `rewarding_doubt.attn_bias`, which the fast loop now uses for
  `--attn-bias`). 0 at step 0.
- **regen_accuracy** (`--regen-every N`): the 507 dev questions answered again by the policy (released
  sampling, seeded per batch) and graded by F1; regen_malformed counts answers that never reach
  " Confidence" or do not parse. 0.396 at step 0, against 0.391 for the cached answers.
- **`--answer-kl W`**: W × answer_kl of each training row added to its loss (one extra reference forward
  without grad per minibatch).

Default schedule (Adam, 32 questions per update, 300 steps), 2 seeds:

| arm | Brier @300 | ECE @300 | AUROC @300 | answer KL @300 | regen accuracy @0 / 100 / 200 / 300 | malformed @300 |
|---|---|---|---|---|---|---|
| attention bias 18–35, lr 1e-2, W=0 | 0.116 ± 0.002 | 0.047 ± 0.007 | 0.904 ± 0.003 | 14.9 | 0.396 / 0.229 / 0.194 / 0.139 | 25% |
| W=0.1 | 0.135 ± 0.006 | 0.081 ± 0.002 | 0.895 ± 0.004 | 1.35 | 0.396 / 0.403 / 0.400 / 0.380 | 1.3% |
| W=1 | 0.135 ± 0.009 | 0.061 ± 0.026 | 0.888 ± 0.009 | 1.05 | 0.396 / 0.392 / 0.400 / 0.398 | 0.9% |
| W=10 | 0.174 ± 0.010 | 0.094 ± 0.033 | 0.827 ± 0.006 | 0.65 | 0.396 / 0.388 / 0.393 / 0.396 | 0.5% |
| all-layer LoRA, lr 3e-4, W=0 | 0.154 ± 0.001 | 0.090 ± 0.013 | 0.886 ± 0.009 | 22.0 | 0.396 / 0.319 / 0.256 / 0.239 | 28% |
| all-layer LoRA, W=1 | 0.135 ± 0.000 | 0.078 ± 0.018 | 0.903 ± 0.005 | 0.26 | 0.396 / 0.406 / 0.403 / 0.403 | 0.2% |

- **Unpenalized, both adapters destroy the answers** at the fast loop's learning rates: the attention bias's
  regenerated accuracy falls from 0.396 to 0.139 with a quarter of the answers malformed; all-layer LoRA's
  to 0.239. The calibration numbers of sections 3–9 are of the stated confidence on the base model's
  answers, not of what these policies would answer.
- **W = 1 keeps the answers** (0.398 for the bias, 0.403 for LoRA at step 300) at a cost in calibration:
  Brier 0.135 against 0.116 for the unpenalized bias. All-layer LoRA with W = 1 keeps AUROC 0.903, the
  attention bias 0.888. W = 10 holds the answers no better and costs much more calibration.
- This is still the fast loop: training uses the cached answers, so it shows the side effect on answers
  but not the compounding of training on one's own drifted answers.

## Defaults now in `fast_loop.py train`

`--batchsize 32 --passes 1 --minibatch 32 --lr 3e-4 --steps 300 --eval-every 50 --bucket-window 64`,
Adam, gradient checkpointing off (since section 8). A step takes 0.61 s, so about 3.5 minutes on an
L40S including 7 dev evaluations; runs before section 8 took 4.7 minutes with checkpointing on. The
released schedule is `--batchsize 8 --passes 4 --minibatch 4 --lr 1e-5`.
`--lora-modules` and `--lora-layers` restrict which adapters train (section 6).
`--gate-bias`, `--residual-bias` and `--attn-bias` train bias vectors instead (section 9).
`--answer-kl W` penalizes drift of the answers and `--regen-every N` measures it by regenerating them
(section 10); without the penalty these adapters degrade the answers.

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
