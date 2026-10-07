# A minimal trainer for the exact objective

October 6, 2026 (PST). `modal_repro/minimal_trainer.py` trains Qwen-2.5-3B to state a calibrated confidence with
the exact objective, on the fast loop's cached answers (`docs/fast_loop_log.md`), using only Hugging Face
`transformers`, PEFT and PyTorch. Each run is on one L40S on Modal; outputs and configs are in `runs/minimal/`.

**Summary.** It matches the fast loop's dev metrics within seed noise and takes 489 ms per step instead of 611
(all-layer LoRA), or 302 instead of 442 (layers 18–35). Most of the gain is from running the 132-token prompt
prefix, shared by every row, once per batch. None of 9 seeds collapsed to a constant confidence.

## How it works

- **Model**: Qwen-2.5-3B-Instruct in bf16, PyTorch SDPA attention, no gradient checkpointing.
- **Adapters**: PEFT LoRA (rank 8, alpha 8) only on the modules and layers that train (`--lora-modules`,
  `--lora-layers`), or zero-initialized vectors added to attention or MLP outputs by forward hooks
  (`--attn-bias`, `--residual-bias`). Inputs never require grad, so backward stops at the first trained layer.
- **Scoring**: one forward pass gives the 11 confidence levels, with logits only at the positions they read.
  The prompt prefix runs once per batch with grad; its key/value cache is expanded over the batch and only
  the answers run.
- **Same as the fast loop**: the objective (`baseline_matched_objective`, fixed KL weight 0.05), Adam at 3e-4,
  batches of 32 length-bucketed questions, 300 steps, dev metrics every 50 steps.

## Checks

**Reference levels.** The cache's reference log-probabilities came from the fast loop's 4-bit model and differ
from bf16 by up to 10 nats, so `minimal_trainer.py ref` recomputed them (26 s). Untrained bf16 model on the 507
dev answers: Brier 0.516, ECE 0.539, AUROC 0.816 (the 4-bit model: 0.566, 0.573, 0.787).

**Shared prefix.** In fp32 it matches a full forward to 0.003 nats, the same as a batched full forward against
one row at a time. In bf16 both comparisons differ by up to 2 nats on some rows: bf16 scoring has that much
noise however it is batched, so a bf16 bar of 0.01 cannot be met. A CPU test (`tests/test_minimal_trainer.py`,
tiny random Qwen2 with LoRA and both biases) checks levels and gradients in fp32.

## Results against the fast loop

Dev metrics of the expected confidence at step 300, mean ± sd over seeds:

| run | seeds | Brier | ECE | AUROC |
|---|---:|---|---|---|
| all-layer LoRA, minimal | 9 | 0.116 ± 0.017 | 0.048 ± 0.017 | 0.917 ± 0.010 |
| all-layer LoRA, fast loop | 3 | 0.120 ± 0.004 | 0.040 ± 0.014 | 0.915 ± 0.005 |
| attention bias 18–35 (lr 1e-2), minimal | 2 | 0.122 ± 0.008 | 0.065 ± 0.034 | 0.914 ± 0.006 |
| attention bias 18–35 (lr 1e-2), fast loop | 2 | 0.114 ± 0.002 | 0.050 ± 0.022 | 0.908 ± 0.001 |

A run takes 2.9 minutes for all-layer LoRA and 1.1 for the attention bias (from the first update, including
6 dev evaluations), against the fast loop's 3.5 and 2.7.

**Stability.** No seed collapsed to one confidence for every answer (Brier 0.32, AUROC near 0.5). The worst
dev point after step 50 is seed 3 at step 150 (Brier 0.176, AUROC 0.865), recovered by step 200. The fast
loop's seed 3 dips at step 200 too; batch order depends only on the seed, so seed 3 meets a hard stretch of
batches. An earlier 4-bit version of this trainer (commit bfbe499, since removed) collapsed on that stretch
(seed 3, steps 100–200) and on seed 2, 2 of 6 seeds; 0 of 9 here is fewer, but not conclusively (Fisher
exact p ≈ 0.14).

## Speed and memory

Milliseconds per step of 32 questions (mean over steps 11–60 of 60-step runs, `--time-steps`):

| what trains | forward | loss | backward | total | fast loop |
|---|---:|---:|---:|---:|---:|
| all-layer LoRA | 189 | 29 | 264 | **489** | 611 |
| LoRA, layers 18–35 | 136 | 24 | 138 | **302** | 442 |
| LoRA, layers 18–35, no shared prefix | 334 | 14 | 326 | 676 | |
| attention bias, layers 18–35 | 85 | 20 | 65 | **172** | |

Peak memory for one batch of 32 rows, all-layer LoRA: 14.1 GB with the shared prefix, 39.3 GB without
(weights 5.9 GB). Over a run the peak is about 33 GB, set by the longest batches. Activations take about
200 KB per token per layer, nearly half of it PEFT's fp32 copies of the LoRA inputs; the fast loop's Unsloth
kernels used less. All-layer LoRA without the shared prefix does not fit in 48 GB.

## Answer drift and `--answer-kl` (Oct 6, afternoon)

Every adapter acts at every token, so training the confidence also moves the answers, which nothing in the
objective protects (`docs/fast_loop_log.md` section 10). The scoring pass now also returns the log-softmax at
each cached answer token (`score_rows`), and `adapters_off` gives the reference (PEFT LoRA layers disabled,
bias hooks passing through):

- `answer_kl` in the dev metrics: KL(policy ‖ reference) over the vocabulary, summed per dev answer.
- `--regen-every N --gt /vol/fast/gt.json`: the dev questions answered again with `generate` (released
  sampling and answer pattern, seeded per batch) and graded by F1; `gt.json` (gold answers of every cached
  question) comes from `fast_loop.py gt` (the runs below used its dev-only predecessor, `--dev-gt`).
- `--answer-kl W`: W × the answer KL of each row added to its loss; one extra reference pass without grad
  (an attention-bias step goes from 172 to 256 ms).

Default schedule, 2 seeds, step 300 (`runs/minimal/akl-*`). The regenerated accuracy of the bf16 base model is
0.426:

| arm | Brier | ECE | AUROC | answer KL | regen accuracy @0 / 100 / 200 / 300 | malformed @300 |
|---|---|---|---|---|---|---|
| attention bias 18–35, lr 1e-2, W=0 | 0.114 ± 0.007 | 0.060 ± 0.024 | 0.912 ± 0.001 | 11.5 | 0.426 / 0.288 / 0.262 / 0.217 | 18% |
| W=0.1 | 0.115 ± 0.005 | 0.045 ± 0.040 | 0.908 ± 0.011 | 1.19 | 0.426 / 0.427 / 0.427 / 0.423 | 0.2% |
| W=1 | 0.128 ± 0.018 | 0.064 ± 0.037 | 0.900 ± 0.013 | 0.78 | 0.426 / 0.425 / 0.444 / 0.439 | 0% |
| W=10 | 0.156 ± 0.009 | 0.060 ± 0.003 | 0.843 ± 0.004 | 0.89 | 0.426 / 0.432 / 0.428 / 0.430 | 0.7% |
| all-layer LoRA, lr 3e-4, W=0 | 0.112 ± 0.011 | 0.051 ± 0.005 | 0.917 ± 0.015 | 5.66 | 0.426 / 0.275 / 0.223 / 0.404 | 7.1% |
| all-layer LoRA, W=1 | 0.112 ± 0.002 | 0.031 ± 0.011 | 0.912 ± 0.004 | 0.28 | 0.426 / 0.443 / 0.423 / 0.439 | 0.6% |

- Unpenalized, the answers degrade as in the fast loop (attention bias: 0.426 → 0.217; LoRA dips to 0.223 at
  step 200 and is back to 0.404 at 300).
- W = 0.1 (bias) and W = 1 (LoRA) keep the answers at or above the base model's and cost no calibration within
  seed noise; all-layer LoRA with W = 1 has the best ECE of any arm (0.031). In the fast loop (4-bit) the same
  penalties cost about 0.02 Brier; with 2 seeds the difference may be noise.
- Training still uses the cached base-model answers, so this measures the side effect, not the compounding of
  training on one's own answers.

## Search with the cached top-k reference, then online trials (Oct 6, evening)

**Top-k reference.** `minimal_trainer.py answer-ref` stores the bf16 reference's top-64 log-probs at every
cached answer token (99.3% of the mass on average, 83–86% at the 1st percentile); `--answer-kl` then needs no
reference pass. The coarse-grained KL (those 64 tokens plus one bucket) is 0.000 at step 0 in every run and
within about 10% of the exact KL (`answer_kl_topk` beside `answer_kl` in the dev metrics). `--answer-kl-target
T` adapts the weight multiplicatively toward an answer KL of T.

**Offline search** (cached answers, top-k penalty, 2 seeds, step 300; `runs/minimal/hs-*`). Safe = answer KL
at most 2, no regeneration more than 1.5 points below step 0, malformed at most 2%:

| arm | Brier | ECE | AUROC | answer KL | Δ accuracy @300 | safe |
|---|---|---|---|---|---|---|
| all-layer LoRA, lr 3e-4, W=1 | 0.108 ± 0.002 | 0.037 | 0.920 | 0.29 | +0.015 | yes |
| attention bias, lr 3e-2, W=0.3 | 0.113 ± 0.008 | 0.041 | 0.914 | 2.57 | −0.007 | no (KL) |
| all-layer LoRA, lr 3e-4, W=0.3 | 0.120 ± 0.012 | 0.043 | 0.905 | 0.61 | +0.012 | yes |
| attention bias, lr 1e-2, target 1 | 0.120 ± 0.010 | 0.061 | 0.910 | 1.19 | +0.012 | yes |
| attention bias, lr 3e-3, W=0.03 | 0.121 ± 0.004 | 0.058 | 0.907 | 0.69 | +0.001 | yes |
| all-layer LoRA, lr 3e-4, target 1 | 0.221 ± 0.143 | 0.161 | 0.730 | 0.72 | +0.028 | one seed collapsed |
| all-layer LoRA, lr 1e-3, W=0.3 | 0.246 ± 0.157 | 0.209 | 0.697 | 2.86 | +0.021 | no |

All 22 arms are in `runs/minimal/hs-*` (summary rows: Brier at step 300, mean of 2 seeds). For LoRA the penalty
also steadies the confidence: with too little of it, or at lr 1e-3, a seed can collapse to one confidence.
The adaptive target does not suit LoRA, whose answer KL (about 0.3) sits below the target, so the
controller weakens the penalty.

**Online trials** (`--online`: each batch's questions answered by the policy, graded, and trained on, with a
live reference; 300 steps, 2 seeds; `runs/minimal/on-*`):

| arm | Brier @300 | AUROC @300 | mean Brier, steps 150–300 | answer KL | worst Δ regen accuracy | malformed (training) | answer tokens |
|---|---|---|---|---|---|---|---|
| attention bias, lr 1e-2, target 1 | 0.110 ± 0.005 | 0.915 | 0.125 | 1.11 | −0.002 | 0.5% | 7.3 |
| attention bias, lr 3e-3, W=0.03 | 0.136 ± 0.015 | 0.896 | 0.128 | 0.83 | −0.006 | 0.8% | 7.4 |
| all-layer LoRA, lr 3e-4, W=1 | 0.159 ± 0.041 | 0.851 | 0.141 | 0.33 | −0.004 | 0.5% | 7.3 |
| all-layer LoRA, lr 3e-4, no penalty | 0.112 ± 0.000 | 0.921 | 0.113 | 3.99 | −0.148 | 4% | 7–9 |
| attention bias, lr 1e-2, no penalty | 0.127 ± 0.008 | 0.899 | 0.132 | 15.4 | −0.323 | 33–42% | 26–35 |

- **Without the penalty, training on its own answers compounds the drift.** The unpenalized attention bias's
  answers fall apart from the first steps: training-time accuracy 0.20–0.33 against about 0.41, a third or
  more malformed, answers growing from 7 to 26–35 tokens, and regenerated dev accuracy down by up to 32
  points. Unpenalized LoRA's training-time accuracy dips to 0.30 around steps 150–200.
- **With it, the answers hold online too**: every penalized arm stays within half a point of its starting
  regenerated accuracy, and training-time accuracy stays at 0.39–0.44.
- **The attention bias with an adaptive 1-nat target is the best online arm** (Brier 0.110, AUROC 0.915),
  better than the same arm offline (0.120). LoRA with W=1, best offline (0.108), is unsteady online: its
  dev calibration swings between evaluations (ECE 0.02 to 0.16) and one seed ends at Brier 0.188.
- Step times (L40S): online attention bias about 1.2 s per step of 32 questions (1.0 s generating),
  all-layer LoRA about 2.8 s (PEFT applies unmerged adapters at every decoding step).

## Stock parameters instead of hooks (Oct 6, evening)

The attention-output bias needs a hook (and, in Unsloth, a patched decoding loop). Qwen-2.5 has a stock parameter
with the same effect: `v_proj`'s bias. Attention weights sum to 1, so a change δ in every value adds the same
vector O·δ to the attention output at every position (test: identical shift at every position to 1e-5). With
2 key/value heads of 128 dimensions it reaches a subspace of at most 256 of the 2,048 residual dimensions per
layer. `--v-bias A-B` trains it, `--norm-gain A-B` trains the RMSNorm weights before attention and MLP; both are
bf16 weights, so Adam updates fp32 master copies that are copied in after each step, and `adapters_off`
restores the originals. Llama-3 has no projection biases; `attention_bias=True` adds zero-initialized ones,
whose `o_proj` bias is exactly the hooked attention-output bias (not tried).

Adaptive 1-nat answer-KL target, 300 steps, 2 seeds (`runs/minimal/stock-*`):

| arm | params | mode | Brier | ECE | AUROC | answer KL | worst Δ regen accuracy |
|---|---:|---|---|---|---|---|---|
| hooked attention bias 18–35, lr 1e-2 (reference) | 36.9k | online | 0.110 ± 0.005 | 0.057 | 0.915 | 1.11 | −0.002 |
| `v_proj` bias 18–35, lr 1e-2 | 4.6k | online | 0.116 ± 0.004 | 0.049 | 0.910 | 1.27 | −0.006 |
| RMSNorm gains 18–35, lr 3e-3 | 73.7k | online | 0.115 ± 0.009 | 0.061 | 0.913 | 0.86 | −0.020 |
| `v_proj` bias 18–35, lr 3e-2 | 4.6k | online | 0.134 ± 0.000 | 0.080 | 0.892 | 1.28 | −0.018 |
| hooked attention bias 18–35 (reference) | 36.9k | offline | 0.120 ± 0.010 | 0.061 | 0.910 | 1.19 | −0.010 |
| `v_proj` bias 18–35, lr 1e-2 | 4.6k | offline | 0.125 ± 0.006 | 0.071 | 0.902 | 1.35 | −0.037 |
| `v_proj` bias 18–35, lr 3e-2 | 4.6k | offline | 0.135 ± 0.004 | 0.080 | 0.898 | 1.33 | −0.006 |
| `v_proj` bias 18–35, lr 3e-3 | 4.6k | offline | 0.144 ± 0.010 | 0.063 | 0.879 | 1.08 | −0.041 |
| RMSNorm gains 18–35, lr 1e-3 | 73.7k | offline | 0.136 ± 0.010 | 0.054 | 0.886 | 1.25 | −0.034 |
| RMSNorm gains 18–35, lr 3e-3 | 73.7k | offline | 0.186 ± 0.093 | 0.134 | 0.876 | 1.24 | −0.037 |
| `v_proj` bias, all 36 layers, lr 1e-2 | 9.2k | offline | 0.219 ± 0.083 | 0.207 | 0.849 | 3.91 | −0.043 |

- **Online, the stock `v_proj` bias nearly matches the hooked bias** with an eighth of the parameters and no
  hooks; it is an ordinary weight of the checkpoint.
- **Online beats offline** for these small adapters, as for the hooked bias.
- **An answer KL near 1 nat is not safe for every adapter:** offline, the `v_proj` bias at lr 1e-2 or 3e-3 and the
  norm gains lost 3–4 points of accuracy at 1.1–1.35 nats. The cost of a nat depends on the directions moved;
  regenerated accuracy has to be measured, not inferred from the KL.
- On all 36 layers the controller did not hold the target (KL 3.9) and a seed degraded.

## 3,000-step online runs on RunPod (Oct 6, evening)

Online, layers 18–35, lr 1e-2, adaptive answer-KL target, seed 1, one RTX 4090 each (`runs/runpod/online-*`).
The first launch (hooked attention bias, targets 0.5, 1 and 2) exposed an unbounded controller: with targets 1
and 0.5 the answer KL stayed above target (Adam's step does not shrink as the weight grows, so at this lr the
KL has a floor of a few nats) and the weight grew to 1e22–1e33, wrecking those three runs by step ~1,800; they
were stopped. The weight is now bounded to [1e-3, 10] and follows a moving average of the KL
(`runs/minimal/controller-cap-check`: with an unreachable target it rises to 10 and stays). Target 2 ran to the
end. Regenerated dev accuracy at step 0 is 0.440 on these GPUs (0.426 on Modal's L40S).

| run (target 2) | params | Brier, mean of steps 1k–3k | ECE | AUROC | answer KL | regenerated accuracy | worst malformed | 3,000 steps |
|---|---:|---|---|---|---|---|---|---|
| hooked attention bias | 36.9k | 0.122 | 0.050 | 0.909 | 2.69 | 0.422–0.458 | 3.0% | 49 min |
| `v_proj` bias (stock, no hooks) | 4.6k | 0.108 | 0.044 | 0.919 | 2.54 | 0.404–0.428 | 1.8% | 55 min |

- Both are stable for 3,000 steps; the bounded controller holds the answer KL near 2 (weight 0.005–0.21).
- The stock `v_proj` bias is better calibrated (final Brier 0.109, ECE 0.041, AUROC 0.913) at a small accuracy
  cost (1–3.6 points below the start), while the hooked bias stayed within about 2 points; late in the run the
  `v_proj` controller's weight fell to near its floor. A lower target (1–1.5) or a higher minimum weight would
  likely close the gap. One seed each, proxy metrics.

## Recovering the `v_proj` run's answers (Oct 6, evening)

`--init` starts a run from another run's trained tensors (fresh optimizer; the reference stays the base model).
From the 3,000-step online `v_proj` run (step 0 here: Brier 0.108, AUROC 0.914, answer KL 2.06, regenerated dev
accuracy 0.418 against 0.426 for the base model on Modal), offline training with a fixed answer-KL weight, 300
steps, 2 seeds (`runs/minimal/recover-*`):

| lr, W | regenerated accuracy @0 / 50 / 100 / 150 / 200 / 250 / 300 | Brier @300 | ECE | AUROC | answer KL |
|---|---|---|---|---|---|
| 1e-2, 0 | 0.418 / 0.424 / 0.414 / 0.413 / 0.409 / 0.403 / 0.416 | 0.106 | 0.051 | 0.919 | 3.13 |
| 1e-2, 0.3 | 0.418 / 0.429 / 0.430 / 0.436 / 0.416 / 0.427 / 0.412 | 0.106 | 0.043 | 0.921 | 1.46 |
| 1e-2, 1 | 0.418 / 0.444 / 0.434 / 0.440 / 0.440 / 0.440 / 0.433 | 0.116 | 0.050 | 0.912 | 0.97 |
| 1e-2, 3 | 0.418 / 0.441 / 0.434 / 0.443 / 0.438 / 0.438 / 0.435 | 0.121 | 0.035 | 0.903 | 0.91 |
| 1e-2, 10 | 0.418 / 0.437 / 0.435 / 0.435 / 0.440 / 0.448 / 0.440 | 0.124 | 0.032 | 0.895 | 0.83 |
| 1e-2, 30 | 0.418 / 0.441 / 0.435 / 0.445 / 0.442 / 0.436 / 0.445 | 0.122 | 0.029 | 0.895 | 0.81 |
| 3e-3, 3 | 0.418 / 0.437 / 0.436 / 0.433 / 0.429 / 0.436 / 0.435 | 0.108 | 0.039 | 0.911 | 1.31 |
| 3e-3, 10 | 0.418 / 0.434 / 0.434 / 0.428 / 0.433 / 0.435 / 0.434 | 0.107 | 0.042 | 0.910 | 1.28 |

- Unpenalized, the drift continues (accuracy 0.403–0.416, answer KL 3.1).
- With W ≥ 1 the answers are back at base level within 50 steps; beyond that W barely matters.
- At lr 1e-2 the recovery costs calibration (Brier 0.116–0.124, AUROC 0.895–0.912); at lr 3e-3 with W = 3–10 it is
  close to free (Brier 0.107–0.108, AUROC 0.910–0.911, accuracy 0.434–0.435), settling near 1.3 nats.
- So only part of the 2 nats of drift hurts the answers: a short, gentle penalized phase after online training
  (about 50–100 steps at lr 3e-3, W ≈ 3) keeps the calibration and restores the answers. Offline, 2 seeds, proxy
  metrics; not yet tried online.

## A floor on the adaptive weight (Oct 6, night)

In the 3,000-step `v_proj` run the adaptive weight sank to about 0.001–0.06 late in training, and the answers
drifted (`docs/answer_kl_vs_accuracy.png`: accuracy falls past about 2 nats per answer). `--answer-kl-min` sets a
floor. Stock `v_proj` bias 18–35, lr 1e-2, offline, 1,000 steps, 2 seeds; regenerated accuracy 0.426 at step 0
(`runs/minimal/floor-*`):

| target, floor | regen accuracy @0 / 500 / 1,000 | Brier, steps 500–1,000 | ECE | AUROC | answer KL @1,000 | late weight |
|---|---|---|---|---|---|---|
| 1.5, 0.001 | 0.426 / 0.417 / 0.420 | 0.113 | 0.045 | 0.916 | 1.68 | 0.037 |
| 2, 0.001 | 0.426 / 0.388 / 0.420 | 0.113 | 0.044 | 0.916 | 2.26 | 0.013 |
| 1.5, 0.1 | 0.426 / 0.428 / 0.430 | 0.118 | 0.044 | 0.910 | 1.05 | 0.100 |
| 2, 0.1 | 0.426 / 0.425 / 0.427 | 0.114 | 0.034 | 0.911 | 1.08 | 0.100 |
| 1.5, 0.3 | 0.426 / 0.436 / 0.432 | 0.120 | 0.041 | 0.904 | 0.65 | 0.300 |
| 2, 0.3 | 0.426 / 0.439 / 0.441 | 0.123 | 0.044 | 0.903 | 0.61 | 0.300 |
| 1.5, 1 | 0.426 / 0.444 / 0.439 | 0.133 | 0.050 | 0.889 | 0.33 | 1.000 |
| 2, 1 | 0.426 / 0.444 / 0.436 | 0.131 | 0.050 | 0.892 | 0.34 | 1.000 |
| 1.5, 3 | 0.426 / 0.417 / 0.435 | 0.144 | 0.044 | 0.867 | 0.20 | 3.000 |
| 2, 3 | 0.426 / 0.426 / 0.427 | 0.144 | 0.050 | 0.868 | 0.22 | 3.000 |

- From a floor of 0.1 up, the weight sits on the floor (the controller wants less), so the target stops
  mattering and the floor sets the trade-off.
- Floor 0.1 keeps accuracy at base (0.427–0.430) with the KL near 1 nat, below the knee, for about 0.003 Brier;
  higher floors buy a little accuracy for up to 0.03 Brier. One training phase, no polish step. Offline; an online
  check and a 3,000-step run are next.

## Differences from the fast loop, and what is not done

- bf16 weights instead of 4-bit; fp32 LoRA weights where the fast loop's were bf16 (TRL casts them).
- Not carried over: gate biases and building the cache (`fast_loop.py cache` still does it).
- The loss is 32 separate calls of the objective (20–30 ms per step); not batched yet.
- All numbers are the fast loop's dev proxy, not the released evaluation.

## Reproducing

```bash
modal run modal_repro/app.py::minimal_ref         # bf16 reference cache and untrained metrics
modal run modal_repro/app.py::minimal_diagnose    # shared prefix vs full forward
modal run modal_repro/app.py::minimal_memory      # peak memory
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/validation_configs.json
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/stability_bf16_configs.json
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/timing_configs.json
modal run modal_repro/app.py::fast_gt             # gold answers of every cached question, for --regen-every / --online
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/answerkl_configs.json
modal run modal_repro/app.py::minimal_tests       # tests/test_minimal_trainer.py in the Modal image
```
