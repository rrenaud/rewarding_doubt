# Pilot: 1,024 train / 512 eval, two epochs, seed 1

Qwen/Qwen3-8B, rank-16 LoRA, learning rate 1e-5, batch size 8, paper reward, F1 grading,
Tinker SDK 0.31.0. Three adapters trained in parallel from the same cached answers, then
evaluated on the same 474 valid held-out answers. 38 of 512 held-out answers and 67 of
1,024 training answers were malformed and excluded. 240 updates per arm (957 / 8 × 2 epochs),
against 10 in the [smoke test](../smoke-20260930T222803Z/report.md). Fixed-answer accuracy is 49.4%.

## Headline

| Checkpoint | Readout | ECE ↓ | Brier ↓ | NLL ↓ | AUROC ↑ | Valid mass | Invalid sampled |
|---|---|---:|---:|---:|---:|---:|---:|
| base | fractional | 0.479 | 0.466 | 2.737 | 0.848 | 0.996 | 0.4% |
| discrete-ppo | fractional | 0.474 | 0.458 | 2.581 | 0.848 | 0.996 | 0.4% |
| **discrete-exact** | fractional | 0.173 | 0.185 | 0.581 | **0.863** | **0.999** | **0%** |
| discrete-exact | argmax | 0.174 | 0.186 | 0.584 | 0.858 | | |
| discrete-exact | sampled | 0.174 | 0.189 | 0.610 | 0.850 | | |
| fractional | fractional | **0.132** | **0.174** | **0.531** | 0.857 | **0.026** | **97.9%** |
| fractional | argmax | 0.138 | 0.182 | 0.556 | 0.825 | | |

1. **discrete-exact is the clear practical winner.** ECE fell from 0.48 to 0.17 and NLL from
   2.7 to 0.58. All three readouts agree, so the integer the model actually prints is
   calibrated. The format is intact: valid mass 0.999 and zero invalid samples.
2. **fractional has the best conditional numbers but broke generation.** Given that it emits
   one of the eleven strings, its mean is the best calibrated (ECE 0.13). But only 2.6% of
   its probability mass is on those strings. It prints a digit and then keeps writing
   ("3\n\nConfidence is low because…") instead of stopping. The sampled readout covers just
   10 of 474 examples and is meaningless. As a model you would deploy, this checkpoint
   fails.
3. **discrete-ppo did essentially nothing.** It finished within noise of the base model on
   every metric, even after 240 updates. That fits signal vanishing at saturation: the base
   model puts its top choice on "10" for 92% of answers, so the 8 samples in a group usually agree and every
   advantage is zero.

## Fractional format collapse

Valid confidence mass averaged over each training batch (`valid_confidence_mass_mean`):

| Step | 20 | 60 | 100 | 140 | 160 | 180 | 200 | 220 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fractional | 0.994 | 0.998 | 0.984 | 0.896 | 0.966 | 0.824 | 0.499 | 0.121 | **0.009** |
| discrete-exact | 0.997 | 0.999 | 1.000 | 0.999 | 1.000 | 0.996 | 0.999 | 1.000 | 0.999 |

Both exact objectives renormalize over the eleven candidates, so they are invariant to total
valid mass. Nothing anchors the EOS after the digit. This seed shows *that* the fractional
arm drifted and discrete-exact did not; it doesn't show *why*. A plausible mechanism:
fractional tolerates spread-out distributions, so lowering some branches' probability by
lowering EOS after their digit is a cheap way to move the mean. That is unverified. Its
calibration was already good by step ~60–100, before the collapse.

Remedies for the next run, in order of simplicity:

- Add a format-anchoring term to both exact arms, $-\lambda \log\sum_k e^{\ell_k}$ (maximum
  likelihood on emitting *some* valid string). This leaves the within-candidate objective
  unchanged.
- Alternatively, select checkpoints on a dev split using valid mass and invalid rate (save
  periodically; currently only `final` is saved).
- Add a "valid mass ≥ 0.95" gate to the evaluation report.

## Readout without training

On the base model the fractional readout again raises AUROC from 0.57 (argmax) to 0.85
without changing calibration, the same as in the smoke test. After discrete-exact training
that gap mostly closes (0.863 vs 0.858): the printed integer now carries the ranking
information itself.

## Reliability (fractional readout: bin → count, accuracy, mean confidence)

- base: 440 of 474 answers in bin 10 (accuracy 0.53, confidence 1.00). Still almost
  entirely "certain".
- discrete-exact: spread across bins 0–9. Bins 0–2 track closely (0.09/0.02, 0.23/0.11,
  0.40/0.21) but are **underconfident**, as are bins 3 and 6–8 (e.g. 0.92 accuracy at 0.69
  confidence). Most of the remaining ECE is underconfidence; it overshot after starting
  overconfident.
- fractional: similar shape, slightly tighter (e.g. bin 8: 0.96 at 0.78).

## Caveats

- One seed and one held-out set; the differences between exact and fractional
  (ECE 0.17 vs 0.13) are conditional on the fractional model's broken format and should not
  be read as a ranking.
- The held-out split was not used for any choice, but there was also no dev split; the
  learning rate and the two-epoch length are untuned.
- Tinker's PPO metrics (`ppo_mean_ratio` ≈ 0.5, `ppo_clipped_fraction` ≈ 0.58, constant from
  step 1) are averaged over all positions, including prompt positions padded with
  old log-prob 0 and advantage 0. They are a reporting artifact, not evidence of an
  off-policy mismatch. This is inferred from their being constant before the first weight
  change; it was not checked directly.
- Cost was not measured. The README pre-launch estimate for this configuration was about
  \$6.80; the smoke test came in at roughly 60% of its estimate. Check Tinker billing.

Metrics: [summary.json](summary.json). Per-example outputs: `*-eval/predictions.jsonl`.
Adapter paths: `*/checkpoint.json`.
