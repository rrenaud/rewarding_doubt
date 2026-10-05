# Phase 1: Rewarding Doubt with Tinker (Qwen3-8B)

> This was the README during the first phase of the project, which used the Tinker API. The current overview is the top-level [README](../README.md).

An executable **single-answer adaptation**, plus an experiment that retains the
confidence distribution instead of decoding one integer. A live 128/128 smoke test
completed all three ten-update training arms and checkpoint evaluations using
Tinker SDK 0.31.0. See the [smoke report](../runs/smoke-20260930T222803Z/report.md)
for results and limitations; this is not a full reproduction of the paper.

## Experiment

For a fixed question and generated answer, score every complete confidence string
`0` through `10`, followed by EOS, from the same prefix. Let `ell[k]` be the sum of
its token log probabilities (including EOS). Define

```
q[k] = exp(ell[k] - logsumexp(ell))
p_fractional = sum(q[k] * k/10 for k in 0..10)
R(p,y) = y*log(clip(p)) + (1-y)*log(1-clip(p))
```

This is conditional on emitting one of the eleven complete confidence strings.
It is not answer-sequence likelihood. We report their total probability mass
separately, and score all eleven candidates rather than relying on top-k.
Qwen splits `10` into two tokens: scoring only the first token would confuse it
with `1`. Including EOS makes the candidate events disjoint. The fixed prefix
ends with a space; confidence strings are appended as explicit token sequences.

Three independent training arms share cached answers, labels and prompts:

| Mode | Objective | Purpose |
|---|---|---|
| `discrete-ppo` | Sample confidence continuations; PPO with leave-one-out group advantages | Practical discrete RL baseline |
| `discrete-exact` | Maximize `sum(q[k] * R(k/10,y))` | Remove sampling noise while retaining discrete rewards |
| `fractional` | Maximize `R(sum(q[k]*k/10),y)` | Train the continuous confidence directly |

The last two objectives differ: reward of the mean is not mean reward. Tinker's
custom-loss callback differentiates through the normalization and weighted mean.
Detaching the fractional confidence and treating it as a constant REINFORCE reward
would omit this gradient. Only the confidence suffix (including EOS) has loss; the answer
and question tokens are conditioning context. Shared adapter updates can still
change answer generation, so fixed answers are essential to the controlled study.

Evaluate **every checkpoint**, including the base model, with fractional, argmax,
and sampled confidence readouts. This separates a readout-only improvement from
an improvement that requires retraining. Report accuracy, ECE (11 equal-width
bins), AUROC, Brier score, clipped NLL, invalid-output rate and valid-confidence mass.
Sampled-confidence metrics exclude invalid confidences and report that exclusion
rate; their sample population may therefore differ from other readouts.

See [docs/objectives.md](objectives.md) for the derivations, optima and
trade-offs of each objective and readout.

## Paper versus released implementation

The [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/bf8065446507b0d3842838564ac4f1f3-Paper-Conference.pdf)
and [released single-answer reward helper](https://github.com/pasta99/RewardingDoubt/blob/main/SingleAnswerSetting/util/RLHelper.py)
differ. Both use logarithmic scoring with epsilon 0.001.

* The paper describes rewards normalized to `[-1,1]` and invalid-format reward -3.
  `--reward paper` uses an affine map of the clipped log-score extrema to that range.
* Released code uses `10 * ((log_score - log(.001)/2) /
  (log(.999) - log(.001)/2) + .25*y)`, and invalid-format reward -30.
  `--reward released` implements this formula exactly. Its range is approximately
  `[-10,10]` for wrong answers and `[-7.5,12.5]` for correct answers.
* The paper says correctness is maximum word-overlap F1 > .5. The released
  `QAResult_to_reward` defaults to exact match, and `Train.py` does not override it.
  Use `--grading f1` (default) or `--grading exact` independently of reward choice.
* Appendix C's printed incorrect-answer clipping expression allows log(0).
  We follow the main text and executable code by clipping p to `[.001,.999]`.

Reward scaling changes its strength relative to other loss terms and the optimizer.
The correctness bonus is constant across confidences for a fixed answer, so it
does not change the optimal confidence. Our exact objectives and per-answer PPO
centering eliminate its confidence gradient.

## Paper-faithful PPO baseline (`--mode paper-ppo`)

`src/rewarding_doubt/paper_ppo.py` and `paper_ppo_step` in `cli.py` reimplement the released
`SingleAnswerSetting/Train.py` loop (TRL 0.8.6 `PPOTrainer`) on Tinker:

* Each step samples the answer **on-policy** from the current adapter (T=0.6, top-p 0.9,
  stopping at the ` Confidence` token, as in `Train.py`), then treats prompt + answer as the
  PPO query. Only the confidence continuation (`: 7<eos>`, T=1, up to 500 tokens) is trained.
* Rewards use the released parser (regex search) and grading on the generated answer; use
  `--reward released` for the code's ×10 scale, +2.5 correctness bonus and −30 format penalty.
* Per-token reward −β·(log π − log π_ref) against the base model, with the score on the last
  token. β starts at 0.05 with TRL's adaptive controller (target 6, horizon 10,000).
* GAE (γ=1, λ=0.95), advantages whitened over the batch, 4 PPO epochs over minibatches of
  batch/2, clip 0.2, token-mean loss, Adam(β₂=0.999, eps=1e-8).
* `--prompt paper` uses the released TriviaQA system prompt; `--rank 8` matches its LoRA rank.

Remaining differences: Qwen3-8B instead of Llama-3-8B-Instruct (Tinker has no Llama), no value
head (Tinker trains only the LM, so V=0 and whitening acts as a batch-mean baseline), no
`ratio_threshold` batch skip, our 1,024-question subset instead of all ~87k TriviaQA
`unfiltered` training questions, and F1 grading as the paper states (the code defaults to exact
match). Every rollout is logged to `rollouts.jsonl`.

## Reproduction limits

This is not an exact reproduction of the paper's reported numbers. The paper uses
Llama-3-8B-Instruct with Unsloth quantization, TRL PPO with a value head and KL
regularization, and on-policy answer generation. This implementation uses Tinker
LoRA, cached base-model answers, a paraphrased prompt, and group-relative PPO
without a value head or reference KL. PPO currently makes one update per rollout;
clipping is consequently mostly inactive on the initial on-policy update.
The exact arms condition on eleven canonical confidence strings, whereas PPO samples unrestricted
continuations and penalizes invalid formats. `discrete-exact` is the closer
controlled comparison for `fractional`.

The provisional model is Qwen3-8B; select a supported model explicitly for a live
run. We request non-thinking chat rendering. The dataset preparation command uses
TriviaQA's `rc.nocontext` configuration, whereas the released code names
`unfiltered`. Preparation caches a full answer/confidence response and keeps the
answer; invalid responses are retained in the cache but excluded from training
and evaluation with counts. QAMPARI, the paper's other baselines, out-of-domain
dataset adapters, confidence intervals and a full hyperparameter reproduction are
not implemented. Arbitrary fixed-answer JSONL datasets can be supplied directly.

## Run

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev]'
source .venv/bin/activate
python -m pytest -q
# Configure TINKER_API_KEY or Tinker CLI authentication before live commands.

rewarding-doubt prepare --split train --limit 128 --output data/train.jsonl
rewarding-doubt prepare --split validation --limit 128 --output data/eval.jsonl

rewarding-doubt evaluate --data data/eval.jsonl --output runs/base

rewarding-doubt train --data data/train.jsonl --mode discrete-ppo --output runs/ppo --max-steps 10
rewarding-doubt train --data data/train.jsonl --mode discrete-exact --output runs/exact --max-steps 10
rewarding-doubt train --data data/train.jsonl --mode fractional --output runs/fractional --max-steps 10

# Use sampler_path from the corresponding run's checkpoint.json:
rewarding-doubt evaluate --data data/eval.jsonl --checkpoint tinker://YOUR_SAMPLER_PATH --output runs/fractional-eval
```

These commands incur Tinker usage. There is no automatic live run. Exact training
uses eleven teacher-forced branches per answer; evaluation also uses eleven
log-probability requests and one generation per answer. Start with small limits.
Do not select checkpoints or tune settings on the final held-out test set. For
larger experiments, use separate development/test splits and multiple seeds.

Both exact modes add a hinge penalty `--format-weight * relu(log(--format-threshold) - log M)`
on the total valid confidence mass `M` (defaults 1.0 and 0.95). Renormalizing over the
eleven candidates otherwise leaves `M` unconstrained, and the unpenalized fractional arm
collapsed to `M = 0.03` in the pilot. Training saves sampler weights every `--save-every`
steps (default 40) and lists them under `snapshots` in `checkpoint.json`.

Each training run writes its arguments, SDK version, input hash, per-step metrics,
and final training/sampling checkpoint paths. Cached data is JSONL with `question`,
`answer`, and `references` (list of aliases); optional fields include `id`,
`multiple_choice`, and `answer_format_valid`. Keep train/evaluation questions
disjoint. Cache outputs and run directories are never silently overwritten.

To run a complete experiment using a raw key or `TINKER_API_KEY=...` assignment
stored in `~/.tinker_api_key`:

```bash
# Pilot: 1,024 train / 512 eval, two epochs, seeds 1-3, all arms in parallel.
.venv/bin/python scripts/run_experiment.py
# The original smoke test:
.venv/bin/python scripts/run_experiment.py --name smoke --train-limit 128 --eval-limit 128 \
    --seeds 2 --max-steps 10 --parallel 1
```

This creates a timestamped directory under `runs/` with data, stage logs, status,
checkpoints, `summary.json` (every evaluation), and `aggregate.json` (mean and std
across seeds). Answers are cached once and shared by every seed and arm. The key is
loaded into subprocess environments and is not included in command arguments or run configuration.
SDK 0.18.2 is rejected by the live service; this project now uses SDK 0.31.x.

Tinker references: [custom losses](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/),
[loss and masking conventions](https://tinker-docs.thinkingmachines.ai/tinker/losses/).

## Pilot cost and launch checklist

Checked 2026-09-30: [Qwen3-8B prices](https://tinker-docs.thinkingmachines.ai/tinker/models/)
per million tokens are \$0.195 prefill/forward, \$0.60 output, \$0.44 training.
Cached prefill is \$0.039; the estimates below conservatively assume no caching.
Checkpoint storage is additional (\$0.10/GB/month). These are usage estimates,
not enforced spending limits.

Assumptions: 150-token average confidence prefix, 32 generated tokens per cached
answer, at most 8 confidence output tokens, PPO group size 8, eleven branches in
each exact arm. Include the extra forward pass in the SDK's custom-loss method,
full context tokens for every branch, and evaluation of the base plus three adapters.

| Component | Smoke: 128 train / 128 eval, 10 steps per arm | Pilot: 1,024 train / 512 eval, 2 epochs per arm |
|---|---:|---:|
| Generate cached answers | \$0.01 | \$0.07 |
| Discrete PPO | \$0.07 | \$1.69 |
| Exact discrete reward | \$0.08 | \$2.16 |
| Fractional confidence | \$0.08 | \$2.16 |
| Four evaluations | \$0.18 | \$0.74 |
| **Estimated token total** | **\$0.43** | **\$6.82** |

Actual cost scales with sequence length and retries; allow about \$1 for the smoke
test or \$10 for this pilot, and inspect billing after the first step. A \$20 planning
budget leaves room for debugging; this code does not enforce a dollar cap. The
pilot can reveal promising trends but is too small for a strong scientific claim.

Before launch:

1. Enable Tinker billing and configure `TINKER_API_KEY` in the shell that runs the
   experiment, or use the file-based launcher above. Do not put the key in source files.
2. Install the local package using the command above. CPU-only local execution is
   sufficient: Tinker hosts training and inference.
3. Run a live smoke test with a few cached answers and one update of each arm.
   Local tests, Qwen tokenizer checks, and the first complete live smoke run pass.
   For a different model or SDK, inspect parsing, finite losses, checkpoint
   save/load and billing again.
4. Run the 128/128 smoke commands above, or prepare 1,024/512 examples and omit
   `--max-steps 10` to train for the default two epochs. Evaluate all three adapters
   on the same cached validation answers. Use paper reward and F1 grading initially.
