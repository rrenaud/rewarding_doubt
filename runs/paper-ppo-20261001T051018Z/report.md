# Paper-faithful PPO baseline (`paper-ppo`), seed 1

A reimplementation of the released Rewarding Doubt training loop (`SingleAnswerSetting/Train.py`,
TRL 0.8.6 `PPOTrainer`) on Tinker with Qwen3-8B. The setup:

- **Answers on-policy, confidence trained.** Each step samples a fresh answer from the current
  adapter (T=0.6, top-p 0.9, stopping at ` Confidence`). Only the `: k<eos>` continuation is trained.
- **Reward.** The released parser, F1 > 0.5 grading, and the code's reward scale (×10, +2.5 for
  a correct answer, −30 for bad format).
- **KL penalty to the base model.** Per token, with adaptive β starting at 0.05.
- **TRL PPO settings.** GAE with γ=1, λ=0.95, and whitened advantages; 4 PPO epochs over
  minibatches of 4; clip 0.2; Adam lr 1e-5 with β₂=0.999 and eps=1e-8.
- **Prompt and adapter.** The released TriviaQA prompt and LoRA rank 8.
- **Remaining differences.** There is no value head (V=0), which Tinker cannot provide. The model
  is Qwen, not Llama. Training used our 1,024-question subset for 2 epochs (256 steps of 8),
  not all ~87k questions.

Every piece of PPO math is checked against TRL 0.8.6's own functions on padded batches
(`tests/test_paper_ppo.py::test_matches_trl_0_8_6_on_padded_batches`). The test catches
planted bugs.

Evaluation is the same fixed-answer protocol as every other run: 474 cached base-model answers,
all 11 confidence strings scored. It uses the paper prompt, so the base model is re-evaluated
under that prompt.

## Result

| Checkpoint | Readout | ECE ↓ | Brier ↓ | NLL ↓ | AUROC ↑ | Valid mass |
|---|---|---:|---:|---:|---:|---:|
| base, paper prompt | fractional | 0.473 | 0.456 | 2.400 | 0.817 | 1.000 |
| base, paper prompt | sampled | 0.474 | 0.461 | 2.753 | 0.625 | |
| **paper-ppo, final (step 256)** | fractional | 0.308 | 0.277 | 1.036 | 0.856 | 1.000 |
| paper-ppo, final | sampled | 0.313 | 0.282 | 1.113 | 0.823 | |
| *for comparison:* discrete-exact + hinge ([hinge run](../hinge-20261001T023959Z/report.md)) | fractional | 0.123 | 0.171 | 0.538 | 0.865 | 1.000 |
| *for comparison:* fractional + hinge | fractional | 0.129 | 0.169 | 0.512 | 0.866 | 1.000 |

The final checkpoint is what the paper's own procedure would report. `Train.py` keeps the epoch
with the highest mean *training* reward, and epoch 1 scored 7.53 against 5.30 for epoch 0.

- **The reimplementation works, unlike our simplified PPO.** ECE fell from 0.47 to 0.31 and NLL
  from 2.40 to 1.04. The pilot's `discrete-ppo` moved essentially nothing (0.479 → 0.474). The
  per-token KL term, whitened advantages and four PPO epochs per rollout give it real signal,
  where a group of 8 identical samples gave none.
- **It is still well behind the exact objectives.** Its final ECE is about 2.5× theirs (0.31 vs
  0.12–0.13) and NLL about 2× (1.04 vs 0.51–0.54). This held-out gap is much larger than the
  ~0.05 ECE run-to-run noise measured for discrete-exact.
- **It remains overconfident.** 208 of 474 answers sit in the top bin, at 79% accuracy and 0.99
  confidence. Bin 9 has 46% accuracy at 0.89 confidence.

## Learning curve (held-out, fractional readout)

| Step | 40 | 80 | 120 | 160 | 200 | 240 | 256 (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ECE | 0.302 | 0.323 | 0.405 | 0.205 | **0.117** | 0.172 | 0.308 |
| NLL | 0.828 | 0.879 | 1.880 | 0.668 | **0.545** | 0.575 | 1.036 |
| AUROC | 0.838 | 0.842 | 0.845 | 0.852 | 0.843 | 0.849 | 0.856 |

The curve is **non-monotonic and noisy**. ECE swings by up to 0.2 between snapshots 40 steps
apart, and by 0.14 between the last two, only 16 steps apart. Step 200 (ECE 0.117) matches the
exact objectives. But picking it uses the *held-out* set, so it is not a fair number; a dev split
would be needed to select it honestly. The comparison stands on the final checkpoint: paper-ppo
*can* reach that region, but does not stay there.

This fits the variance argument in [the objectives doc](../../docs/objectives.md). PPO sees one
sampled confidence per question and a batch-mean baseline, whereas the exact objectives use the
full 11-way distribution on every step.

## Training dynamics

| Epoch | Mean training score | On-policy answer accuracy | Invalid format | Mean sampled confidence | Mean KL (nats) |
|---|---:|---:|---:|---:|---:|
| 0 | 5.30 | 42.3% | 3.5% | 0.83 | 3.6 |
| 1 | 7.53 | 41.9% | 1.7% | 0.63 | 8.4 |

- **Answers are not damaged.** On-policy answer accuracy is flat (42.3% → 41.9%). It is lower than
  the cached answers' 49.4% because of the different prompt and sampling, and because their
  parser fails on long answers.
- **The KL penalty barely moved.** KL rose to the TRL target of 6 nats and above, but with horizon
  10,000 and 8 samples per step, β only went from 0.0500 to 0.0498.
- **Format failures are real ones.** Invalid rollouts are mostly answers that ramble past 256 tokens
  or end in `. Confidence:` instead of `, Confidence:`. These get −30 under the released parser.
  Every rollout is in `paper-ppo-s1/rollouts.jsonl`.

## Cost and time

Training took 1 h 55 min for 256 steps (~27 s/step), against 34 min for 240 steps of each exact
arm. Each step makes two sampling rounds, a reference-model pass, and 8 pipelined
forward/backward + optimizer calls. Token cost was not measured, but is likely below the exact arms':
about 5.8k trained tokens per step (4 PPO epochs × 8 × ~180) against about 13k for 11-branch scoring. The snapshot evaluations are in `paper-ppo-s1-snapshots/`, with
the curve data in `curve.json`.

## Caveats and next steps

- One seed. Given how much the curve swings, the final checkpoint's value is itself noisy:
  ±0.1 ECE between neighbouring snapshots.
- Selection on a dev split, for all arms, is the next step. Plausibly *both* paper-ppo and the
  exact arms improve at a selected checkpoint, so the gap should be compared at matched
  selection.
- Evaluation uses fixed base-model answers. The paper evaluates on answers freshly generated by
  the trained model; that protocol is not implemented.
- Not yet tried: a learned value head (e.g. a separate small model), the paper's full dataset
  size, and multiple seeds.
