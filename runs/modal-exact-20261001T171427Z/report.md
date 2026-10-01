# Exact objectives vs the released PPO, on the released code's Llama setup

`modal_repro/exact_llama.py` reproduces the [released-code run](../modal-stage2-20261001T082216Z/report.md)
in everything except the update rule:

- **Same model and adapter.** 4-bit Unsloth Llama-3-8B-Instruct, with LoRA r=8/α=8, seed 3407,
  and bf16 LoRA weights as their loader sets.
- **Same data and prompt.** Their prompt, and the same 1,024 training questions; batch 8,
  2 epochs, torch Adam at lr 1e-5.
- **Same answer sampling.** Each step samples `Answer: …, Confidence` from the current policy
  exactly as `Train.py` does, and grades it with their F1 code.
- **The one change: the update rule.** PPO (value head, KL penalty, 4 PPO epochs on one sampled
  confidence) is replaced by **teacher-forcing all 11 `: k<eot>` continuations** with our
  `core.objective`. That is the same function as the Tinker runs, with paper reward and the
  hinge (λ=1, τ=0.95). There is no value head and no KL penalty.
- **Same evaluation.** Their protocol on the same 512 held-out questions (identical question
  order).

## Result

| Llama-3-8B, epoch 2 | ECE ↓ | AUROC ↑ | Brier ↓ | Accuracy | Training time |
|---|---:|---:|---:|---:|---:|
| base | 0.333 | 0.560 | 0.337 | 63.3% | |
| **released PPO** | 0.074 | **0.771** | **0.188** | 62.8% | 56 min |
| discrete-exact + hinge | 0.076 | 0.718 | 0.199 | 63.7% | **34 min** |
| fractional + hinge | **0.062** | 0.701 | 0.203 | 63.9% | **34 min** |

Epoch 1 for comparison: PPO 0.144, discrete-exact 0.140, fractional 0.150 ECE.

Paired bootstrap over questions (400 resamples), difference from released PPO:

| | ECE | AUROC | Brier |
|---|---|---|---|
| discrete-exact − PPO | +0.002 [−0.041, +0.027] | **−0.053 [−0.093, −0.012]** | +0.011 [−0.002, +0.024] |
| fractional − PPO | −0.012 [−0.050, +0.024] | **−0.069 [−0.104, −0.030]** | +0.014 [+0.002, +0.027] |

- **Calibration (ECE) is a tie.** All three methods remove about 78–81% of the base model's ECE,
  and the differences are well inside the evaluation noise.
- **The released PPO discriminates better.** Its AUROC is 0.05–0.07 higher, and the intervals
  exclude zero; its Brier is slightly better. The exact objectives make the model's confidence
  right *on average* (mean training confidence falls 0.88 → 0.62, close to the 64% accuracy),
  but they separate right from wrong answers less well.
- **The exact objectives are faster.** Training took 34 min against 56 min, with one forward and
  backward over 11 short branches per question instead of PPO's sampling, reference pass and
  8 minibatch updates.
- **The format held.** Valid-confidence mass stayed at 0.96–1.00, the hinge was active on
  1–4% of rows, and 11–13 of 2,048 sampled answers never reached ` Confidence` and were skipped.

## Important confound: optimizer steps

TRL's PPO makes **8 optimizer steps per batch** (4 PPO epochs × 2 minibatches of 4): 2,048 Adam
steps over the run. The exact objectives made **1 step per batch**: 256 steps. Both used lr 1e-5.
So PPO took 8× as many parameter updates from the same data. The exact runs also still looked
under-trained at the end: confidence was still falling in the last quarter.

The exact loss is a deterministic function of (question, answer, label). It needs no importance
weights, so reusing each batch is valid. The compute-matched comparison would let the exact
objectives make the same number of updates: 4 passes × 2 minibatches per batch, or equivalently
more epochs or a higher learning rate. Until that runs, "PPO has better AUROC" may reflect update
count rather than the objective.

## How this revises the Qwen/Tinker conclusion

On Qwen via Tinker, the exact objectives beat our `paper-ppo` reimplementation by a wide margin
(ECE 0.117 vs 0.300). On Llama with the **released** PPO, they tie on ECE and lose on AUROC.
So the Tinker gap was mostly about **our reimplementation**, not about PPO itself. Likely causes:
no value head (here the value loss drops 15 → 2.6 within the first quarter of training), and
possibly the model. It was not about the objectives being inherently better.

## Caveats

- One seed per method and one 512-question evaluation set; intervals cover evaluation noise
  only, not training noise. On Tinker, two same-seed runs differed by 0.05 ECE.
- Answers are sampled during training, so each method saw different answers (same questions).
  Accuracy is flat in all runs (63–64%).
- 1,024 training questions, about 1.2% of the paper's data. The ranking might change with scale.

## Next steps

1. **Compute-matched exact runs:** 8 optimizer steps per batch (4 passes × 2 minibatches), same
   data. This is the cleanest test of the confound above. It costs about \$2–3 per method.
2. **2–3 seeds** of the best configurations, to measure training noise on Llama.
3. A value head (or a per-question baseline) for the Tinker PPO reimplementation, to confirm the
   cause of its gap.

Files: `*/metrics.jsonl` holds the per-step training metrics, `*/eval_*.json` the per-question
outputs from their inference code, `summary.json` the metrics, and `*/train.log` the logs.
Adapters are on the Modal Volume `rewarding-doubt-repro` under `outputs/exact-20261001T171427Z/`.
