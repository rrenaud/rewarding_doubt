# Hinge rerun: valid-mass penalty on the exact arms, seed 1

Same as the [pilot](../pilot-20261001T002945Z/report.md): same cached answers (copied with
`--data-from`), 474 valid held-out answers, Qwen3-8B, rank-16 LoRA, lr 1e-5, ≈240 updates,
seed 1. The one change: both exact arms minimize

$$
\mathcal L' = \mathcal L + 1.0\cdot\operatorname{relu}\big(\log 0.95 - \log M\big),
\qquad M = \sum_{k=0}^{10} e^{\ell_k},
$$

where $M$ is the total probability of the eleven valid confidence strings. Sampler weights were saved
every 40 steps (`*/checkpoint.json` → `snapshots`). The base model and PPO were not rerun;
base numbers are from the pilot.

## Result: the collapse is fixed, and the two exact arms are now tied

| Checkpoint | Readout | ECE ↓ | Brier ↓ | NLL ↓ | AUROC ↑ | Valid mass | Invalid sampled |
|---|---|---:|---:|---:|---:|---:|---:|
| base (pilot) | fractional | 0.479 | 0.466 | 2.737 | 0.848 | 0.996 | 0.4% |
| fractional, no hinge (pilot) | fractional | 0.132 | 0.174 | 0.531 | 0.857 | 0.026 | 97.9% |
| discrete-exact, no hinge (pilot) | fractional | 0.173 | 0.185 | 0.581 | 0.863 | 0.999 | 0% |
| **fractional + hinge** | fractional | 0.129 | **0.169** | **0.512** | **0.866** | **1.000** | **0%** |
| fractional + hinge | argmax | 0.134 | 0.173 | 0.528 | 0.848 | | |
| fractional + hinge | sampled | **0.120** | 0.172 | 0.520 | 0.847 | | |
| **discrete-exact + hinge** | fractional | 0.123 | 0.171 | 0.538 | 0.865 | 1.000 | 0% |
| discrete-exact + hinge | argmax | 0.122 | 0.172 | 0.541 | 0.858 | | |
| discrete-exact + hinge | sampled | 0.123 | 0.172 | 0.543 | 0.857 | | |

- **Fractional now produces a usable model.** Valid mass is 1.000 and every one of the 474
  sampled replies is a clean integer. The printed integer is as well calibrated as the
  conditional mean (sampled ECE 0.120). Its held-out metrics match or beat the collapsed
  pilot run, so the hinge did not cost calibration.
- **fractional and discrete-exact are statistically indistinguishable here.** Fractional
  leads slightly on NLL, Brier and fractional-readout AUROC; discrete-exact is slightly
  better on argmax/sampled AUROC. All gaps are ≤ 0.03 NLL and ≤ 0.02 AUROC.
- **Run-to-run noise is about as large as these gaps.** discrete-exact's ECE moved from
  0.173 to 0.123 between the pilot and this rerun. Same seed, same data, and its penalty
  was active on only 7 steps for one example each. Tinker training is evidently not
  bit-reproducible, so a single run can't rank the arms; repeated runs or seeds are
  needed.

## What the hinge actually did

It was active on only 7 steps per arm, all between steps 4 and 33. Each time it fired on
one example in eight ($\texttt{format\_hinge\_active} = 0.12 = 1/8$), and the examples were the same in
both arms (steps 4, 8, 9, 11, 17, 19). These appear to be a few training questions where
even the base model puts noticeable mass on non-standard continuations. The lowest batch
mean mass over the run was 0.945.

Without the hinge, the pilot's fractional arm first dipped below 0.95 at those same steps
(17, 26). It then dipped more and more often (12 of its first 86 steps) and collapsed
during epoch 2. A small correction early on appears to prevent a drift that otherwise
compounds. This is inferred from one pair of runs.

## Reliability (fractional readout: bin → count, accuracy vs confidence)

Both arms use the full range. Most of the residual ECE is **underconfidence in the middle
and upper bins**: fractional bin 7 has 0.88 accuracy at 0.68 confidence and bin 8 has 0.97
at 0.78; discrete-exact bin 7 has 0.89 at 0.69 and bin 8 has 0.98 at 0.79. The bottom
bins track well (fractional bin 1: 0.20 at 0.13). Training overshot from the base
model's 0.99 confidence. A lower learning rate, fewer epochs, or checkpoint selection on a
dev split may reduce this.

## Cost and artifacts

Wall time was 34 minutes for training and 4 minutes for evaluation, running both arms in
parallel. Six intermediate sampler checkpoints per arm (steps 40–240) added roughly 1% to
training time. Cost was not measured; it is roughly two-thirds of the pilot's training
cost. Metrics: [summary.json](summary.json).

## Next steps

1. Pick a checkpoint on a dev split: training rows 1024+ (not yet cached), using the saved
   snapshots to address the late-training underconfidence.
2. Repeat both hinge arms 2–3 times (or with seeds) to measure the noise floor before
   claiming any ranking.
3. Optionally rerun PPO with a larger group or higher temperature; at G=8 it still has no
   learning signal from this base model.
