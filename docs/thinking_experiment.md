# Experiment design: unscored thinking before the confidence

**Question.** Can the model calibrate better if it may write a short, unscored check of its answer
before stating its confidence, and how much regularization does that free text need?

**Hypothesis.** A few sentences of self-verification ("The capital of Australia is Canberra, not
Sydney…") give the model a way to use knowledge it does not apply when it must emit a number
immediately. That should mostly improve ranking (AUROC), and calibration where the base model's
first instinct is wrong. Without a constraint, the free text will drift into whatever maximizes
reward: degenerate strings, compressed codes, or empty text. A KL penalty toward the base model on
the thinking tokens is the natural guard.

## Format

The answer is generated and fixed first, exactly as now. The thinking comes after it, so it can
inspect the answer but cannot change what is graded.

```
Answer: <answer>
Check: <free text, at most T tokens, never scored>
Confidence: <0-10>
```

- Stage 1, answer: sampled at T=0.6 until `\nCheck:` (as now, with the stop string changed). Graded by F1 > 0.5.
- Stage 2, thinking: sampled at T=1 until `\nConfidence:` or T tokens (T = 96 in the main runs).
- Stage 3, confidence: scored exactly, as in discrete-exact. One forward pass over
  prompt + answer + check + `\nConfidence: ` + sampled k* gives π over the 11 levels and the stop
  check, with the usual hinge.

Llama-3's single-token numbers 0–10 keep the single-pass scoring unchanged.

## Objective

For a question q with answer a and label y, sample G thinking strings t₁…t_G from the current
policy (G = 4). Let J(t) be the exact expected calibration reward given that thinking:
J(t) = Σₖ π_θ(k | q, a, t) R(k/10, y). This is the same quantity the current discrete-exact arm
maximizes, so it involves no sampling in k.

```
loss = − mean_i J(t_i)                                         # confidence tokens, exact gradient
       − mean_i (J(t_i) − b_i) · log π_θ(t_i | q, a)           # thinking tokens, score function
       + β · mean_i KL_tokens(t_i)                             # thinking tokens, KL to base
       + hinge(number mass) + hinge(stop after k*)             # confidence format, as now
```

- **Confidence tokens** get the exact discrete gradient, conditioned on each sampled check.
- **Thinking tokens** get a policy gradient with reward J(t_i). The baseline b_i is the
  leave-one-out mean of J over the other G − 1 checks for the same question, so the signal is
  "this check led to better calibration than the others". The reward is already low-variance
  because J is exact in k; the remaining variance comes only from the thinking sample.
- **KL on thinking tokens only**, as a per-token estimate log π_θ(t) − log π_ref(t), with π_ref the
  base model (adapter disabled, one no-grad pass, which the KL arm already does). No KL on the
  confidence tokens: that is what hurt calibration in the 2×2.
- **The reward must not see the label.** J depends on y, but the thinking is generated before
  grading and only influences π(k), so the only way to raise J is to make the confidence track
  correctness.

## Arms (3 seeds each, tuned exact + hinge settings: lr 4e-5, 2 updates per batch)

| Arm | Thinking at train time | Thinking trained? | β (KL on thinking) | What it isolates |
|---|---|---|---|---|
| A. No thinking | – | – | – | current best (control) |
| B. Frozen thinking | sampled from the base model | no (gradient only on the confidence) | – | does the base model's own check help, if the confidence learns to use it? |
| C. Trained thinking, KL | sampled from the policy | yes | 0.05 | main arm |
| D. Trained thinking, weak KL | sampled from the policy | yes | 0.005 | how much KL matters, and what drift looks like |
| E. Filler | fixed filler of the same length ("Let me check." repeated) | – | – | is any gain from the content, or just extra compute and positions? |

If budget allows, β ∈ {0, 0.005, 0.05, 0.2} on the small training set (512 questions, 2 seeds)
before the main runs maps the KL trade-off; β = 0 is expected to drift.

## Measurements

- **Calibration and ranking**, as now: ECE, AUROC, Brier (sampled and from π), accuracy, format
  failures; dev for selection, test once.
- **Does the content matter?** At evaluation, swap each question's check with another question's,
  and re-score π. The drop in AUROC is how much the confidence relies on the check's content.
- **Thinking health.** Mean length; KL to the base model per token; perplexity of the check under
  the base model; the fraction of checks that restate or contradict the answer (a cheap keyword
  heuristic, plus 30 hand-read samples per arm); for arm D, examples of drift.
- **Cost.** Seconds per step and tokens generated.

## Evaluation protocol

The released evaluation parses `Answer: X, Confidence: k` with one regex, which a check in between
would break. All arms, including A, are therefore evaluated with one shared protocol for this
experiment:
- generate the answer, then the check (arms B–E), then the confidence
- parse each field separately
- grade with the same F1 > 0.5 code
- compute the same torchmetrics ECE

Arm A also gets the released evaluation, to tie this experiment to every earlier number.

## Stages and cost

| Stage | What | Runs | Estimated cost |
|---|---|---|---|
| 0. Feasibility | Base model on the new format (dev, no training): format compliance, check length, and zero-shot calibration with and without a check | 2 evals | under $1 |
| 1. Pilot | Arm C, 1 seed, 64 steps; inspect checks, KL, step time | 1 | ~$1 |
| 2. β map (optional) | β ∈ {0, 0.005, 0.05, 0.2}, 512 questions, 2 seeds | 8 | ~$6 |
| 3. Main | Arms A–E, 3 seeds, full training set, 256 steps | 15 | ~$20–30 |

Generating G = 4 checks of up to 96 tokens per question makes a step roughly 3–4× the current
3.6 s. The training pass grows from 1 to 4 sequences per question, about 260 tokens each.

## Risks and how the design handles them

- **The base model ignores the format** (no `Check:` line, or the number inside the check). Stage 0
  measures this; the prompt gets one worked example if compliance is below about 95%.
- **Thinking that changes the answer.** The graded answer is fixed before the check, so a check
  that concludes "actually it's Y" can only lower the confidence. That is the desired behavior.
- **Reward hacking through the check.** For example, emitting a fixed token pattern that steers π
  toward a constant, well-calibrated-on-average confidence. The content-swap ablation and AUROC
  catch this: a constant confidence has AUROC 0.5.
- **High learning rates damaging answers.** Unchanged from the search: the eligibility rule
  (accuracy within 2 points of base, format failures ≤ 2%) applies to selection.
- **Cost of KL.** The reference pass grows with the check's length. It stays a no-grad pass, about
  20–30% of a step.
