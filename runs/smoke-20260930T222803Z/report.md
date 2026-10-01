# Completed Tinker smoke test

Qwen/Qwen3-8B, rank-16 LoRA, learning rate 1e-5, seed 2. Three independent adapters, ten updates each, batch size eight, paper reward and F1 grading. SDK 0.31.0.

Cached 128 training and 128 held-out questions, with no question-ID overlap. Excluded 9 malformed training responses and 13 malformed evaluation responses. All methods used the same 80 training examples and the same 115 valid held-out answers. Fixed-answer accuracy is 39.1%.

## Training comparison using fractional confidence for every checkpoint

| Training method | ECE ↓ | Brier ↓ | NLL ↓ | AUROC ↑ |
|---|---:|---:|---:|---:|
| base | 0.5817 | 0.5668 | 3.3853 | 0.8727 |
| discrete-ppo | 0.5793 | 0.5631 | 3.2969 | 0.8708 |
| discrete-exact | 0.5606 | 0.5325 | 2.6527 | 0.8613 |
| fractional | 0.5593 | 0.5303 | 2.6016 | 0.8610 |

The fractional objective slightly outperformed exact discrete reward on calibration in this run; their difference is small. Both improved more than sampled PPO over ten updates. Calibration remains poor. This single-seed smoke test verifies the pipeline and suggests directions; it does not establish which objective is better.

## Readout comparison without training

| Base-model confidence readout | ECE ↓ | Brier ↓ | NLL ↓ | AUROC ↑ |
|---|---:|---:|---:|---:|
| argmax | 0.5852 | 0.5749 | 3.8171 | 0.5571 |
| sampled | 0.5809 | 0.5673 | 3.7214 | 0.5714 |
| fractional | 0.5817 | 0.5668 | 3.3853 | 0.8727 |

The fractional readout retains ordering information lost when confidences are rounded to one level. This substantially improves AUROC here, while leaving overconfidence largely intact.

## Validation and artifacts

- All three methods completed ten finite-loss updates, saved training and sampling checkpoints, and successfully reloaded for held-out evaluation.
- All evaluations used identical question IDs and correctness labels. Sampled confidence format error was zero for all four checkpoints.
- Fifteen local tests pass against SDK 0.31.0. The initial SDK 0.18.2 attempt was rejected before generation; that failed attempt is in a separate run directory.
- PPO had only one update with a substantial nonzero surrogate loss; low within-group confidence variation limited its signal.
- Complete metrics: [summary.json](summary.json). Per-example outputs are in each evaluation directory. Adapter paths are in each training directory’s checkpoint.json. Package versions: [environment.json](environment.json).
- Estimated token usage cost: **\$0.25**, excluding storage and assuming no cache discounts. This is a calculation from sequence lengths, not a verified bill; see [estimated_cost.json](estimated_cost.json).

For the next experiment, use more updates, multiple seeds and a larger held-out set. Keep exact-discrete and fractional arms paired; separately examine the low-variation PPO baseline and format exclusions.
