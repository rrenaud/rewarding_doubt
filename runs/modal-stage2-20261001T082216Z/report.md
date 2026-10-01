# Released Rewarding Doubt code at our data scale (Modal, Llama-3-8B)

The released `SingleAnswerSetting/Train.py` (commit `416c9e0`) ran on Modal with its exact
pinned environment: torch 2.5.1, transformers 4.48.0, TRL 0.8.6, peft 0.14.0 and Unsloth
`d6982c1`. It used 4-bit `unsloth/llama-3-8b-Instruct-bnb-4bit`, LoRA r=8, lr 1e-5, 2 epochs
and **batch size 8**. Batch size 8 is inferred: the paper's 7-day runtime fits it, and the
stage 1 timing at the default of 2 does not.

The run used **our 1,024 training questions** and was evaluated on **our 512 held-out
questions**, matched by TriviaQA question ID in the `unfiltered` config their code loads.
Training and inference code run unmodified. `modal_repro/subset.py` only narrows the dataset
their `load_prepared_dataset` returns, and keeps a copy of each epoch's checkpoint. Evaluation
is their `InferenceDatasetSplit.inference_dataset` plus their `EvaluationMetrics` functions, with
the notebook's settings: F1 > 0.5, max confidence 10, 11-bin torchmetrics ECE.

## Sanity check: the base model matches the paper

| Llama-3-8B base ("Verbalize") | ECE | AUROC | Accuracy |
|---|---:|---:|---:|
| Paper, Table 1 (full validation, 11,313 questions) | 0.346 | 0.586 | 63.1% |
| This run (our 512 questions) | 0.333 [0.29, 0.38] | 0.560 | 63.3% |

The bracket is a 95% bootstrap interval over questions.

## Result: their method works at 1.2% of the paper's data

| Llama-3-8B | ECE ↓ | AUROC ↑ | Brier ↓ | Accuracy | Wrong format |
|---|---:|---:|---:|---:|---:|
| base | 0.333 | 0.560 | 0.337 | 63.3% | 0.6% |
| after epoch 1 | 0.144 | 0.602 | 0.238 | 64.5% | 0.0% |
| **after epoch 2** (their best-training-reward pick) | **0.074** [0.05, 0.12] | **0.771** | **0.188** | 62.8% | 0.2% |
| *Paper, full data (175k samples)* | *0.023* | *0.859* | | *63.1%* | |

- **Calibration improves steadily.** ECE falls from 0.33 to 0.14 to 0.07, and accuracy is
  unchanged. With 256 PPO steps instead of about 21,900, it gets roughly 85% of the way from
  the base model to the paper's ECE. AUROC improves more slowly (0.56 → 0.77, against 0.86 in
  the paper).
- **Training is smooth.** By quarters of training:
  - mean reward rises 6.9 → 8.8 → 8.9 → 9.1
  - KL rises 2.4 → 6.7 nats, reaching TRL's target of 6
  - the clip fraction stays at 1–2%
  - the value head's loss falls 14.9 → 2.6, so it learns quickly
- **Port check.** Our port of their metrics (`paper_ppo.evaluation_metrics` plus
  `is_correct_f1`) reproduces all five numbers above from their saved outputs, to 1e-6.

## Comparison with our runs, all under the released evaluation protocol

All rows below are scored on the same 512 held-out questions with the same metric code
(see [paper-protocol run](../paper-protocol-20261001T082301Z/summary.json)).

| Model | Method | ECE ↓ | AUROC ↑ | Brier ↓ | Accuracy |
|---|---|---:|---:|---:|---:|
| Llama-3-8B | base | 0.333 | 0.560 | 0.337 | 63.3% |
| Llama-3-8B | **released code** (TRL PPO, value head) | **0.074** | 0.771 | 0.188 | 62.8% |
| Qwen3-8B | base, paper prompt | 0.474 | 0.640 | 0.460 | 49.8% |
| Qwen3-8B | our Tinker reimplementation, `paper-ppo` (no value head) | 0.300 | 0.812 | 0.274 | 49.9% |
| Qwen3-8B | discrete-exact + hinge | 0.117 | **0.853** | 0.172 | 48.1% |
| Qwen3-8B | fractional + hinge | 0.120 | 0.846 | 0.173 | 48.5% |

**What can and cannot be concluded:**

1. **The released code clearly beats our Tinker reimplementation.** It removed 78% of the base
   model's ECE (0.333 → 0.074); our `paper-ppo` removed 37% (0.474 → 0.300), with the same data,
   step count, batch size and learning rate. The candidate causes are confounded:
   - The **model** differs. Qwen3-8B starts more overconfident (ECE 0.47 vs 0.33) and less
     accurate (49% vs 63%), so it has more to fix.
   - The **value head** is absent on Tinker. Here the value head learns within the first
     quarter of training, and it gives each sample a per-prompt baseline. Our V=0 version relies
     on batch whitening alone, which matches the much noisier learning curve we saw on Tinker.
   - Smaller differences: 4-bit quantization, LoRA parameterization, and training-time grading.
     Our Tinker runs graded against value + aliases + normalized aliases from `rc.nocontext`;
     the released code grades against `unfiltered` normalized aliases only. Evaluation uses the
     latter for both.
2. **Cross-model rankings are not valid.** Released-code Llama has the lowest ECE (0.074), and
   our exact objectives on Qwen have the highest AUROC (0.853), but different base models start
   from different calibration and accuracy. ECE in particular depends on the accuracy level.
   So "exact beats PPO" is established only within Qwen (0.117 vs 0.300); "exact beats the
   released code" is **not established**.

## Cost and timing

Training took 56 min for 256 steps on an L40S (about 13 s/step at batch 8). Evaluation took
about 2 min per checkpoint. Stage 2 cost about \$2.50 of L40S time, plus about \$0.85 for the
stage 1 smoke test. At this speed a full reproduction (87.6k questions × 2 epochs = 21.9k steps)
would take about 79 L40S-hours, **about \$155**. That is half the earlier estimate, and fits the
paper's 7 days on a slower A40.

## Next steps

The decisive experiment is **our exact objectives on Llama-3-8B, on Modal**: same model,
quantization, LoRA, data and evaluation as the released code, changing only the objective.
That needs a short HF/peft training loop (11-branch scoring plus the hinge); it does not need
Tinker. It would cost roughly the same as this run.

A cheaper partial test of the value-head hypothesis: rerun the released code with the value
loss disabled (`vf_coef=0`). This still leaves the value head's predictions in the advantage
calculation, so it is weaker; zeroing the values is the clean ablation.

Files: `result.json` holds the metrics and timings, `eval_*.json` the per-question outputs from
their inference code, `scalars.json` the TRL TensorBoard scalars, and `train.log` / `eval_*.log`
the logs. Checkpoints are on the Modal Volume `rewarding-doubt-repro` under
`outputs/stage2-20261001T082216Z/`.
