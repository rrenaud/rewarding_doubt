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

## Results (2026-10-02, dev only, reduced scale)

**Scale actually run** (to fit a $15 budget): 512 training questions, 128 steps (2 epochs), tuned
exact + hinge settings (lr 4e-5, 2 updates per batch, hinge 1.01 at 0.95), G = 4 checks per
question for the trained and frozen arms, checks up to 96 tokens (they average ~25), evaluation
on the 512 dev questions. Test was not touched. Code: `modal_repro/thinking_llama.py`; runs:
`runs/modal-thinking-{stage0,main,main-none,seeds34}`; summary `runs/thinking/summary.json`.
Spend: about $12.8 on Modal L40S.

**Stage 0 (base model, no training).** The base model follows the Answer/Check/Confidence format
on 100% of dev questions, checks average 25 tokens, and essentially all next-token mass after
"Confidence: " is on the 11 numbers. Its zero-shot check *hurts* ranking: AUROC of E_π[k] is 0.621
with its own check, 0.669 with another question's check, 0.692 with no check. Base checks are
mostly confirmations ("This is a well-documented event…") that push it toward 10. Answer accuracy
is 61.9% with this prompt (66.0% with the released prompt).

| Arm (dev, sampled confidence) | Seeds | Brier ↓ | ECE ↓ | AUROC ↑ | AUROC of E_π[k]: own / swapped / no check | Accuracy |
|---|---:|---:|---:|---:|---|---:|
| none (no check line) | 4 | 0.176 ± 0.003 | **0.058 ± 0.007** | 0.783 ± 0.008 | 0.796 / – / – | 62.8% |
| filler (fixed 36-token text) | 2 | 0.181 | 0.082 | 0.782 | 0.801 / – / 0.792 | 63.2% |
| frozen (base-model check) | 4 | **0.167 ± 0.005** | 0.067 ± 0.010 | **0.823 ± 0.012** | 0.839 / 0.832 / 0.828 | 58.8% |
| frozen, excluding seed 3 | 3 | 0.169 | 0.059 | 0.813 | | 62.4% |
| trained check, β = 0.05 | 4 | 0.177 ± 0.007 | 0.080 ± 0.024 | 0.792 ± 0.010 | 0.810 / 0.807 / 0.797 | 62.5% |
| trained check, β = 0.005 | 2 | 0.177 | 0.099 | 0.815 | 0.833 / 0.815 / 0.828 | 62.4% |

± is the standard error over seeds. Per-seed ECE: none 0.046 / 0.079 / 0.054 / 0.053;
frozen 0.075 / 0.050 / 0.092 / 0.053; trained β = 0.05 0.064 / 0.150 / 0.050 / 0.055.

- **Frozen seed 3 damaged its answers** (accuracy 47.9%) and is ineligible under the search's rule.
  Its AUROC (0.854) is inflated by the damage. Without it, the frozen arm still ranks better than no
  check: AUROC 0.808, 0.831 and 0.801, against 0.804, 0.767, 0.783 and 0.779. Brier is slightly
  better and ECE is the same.
- **Training the check did not help.**
  - With β = 0.05, AUROC, ECE and Brier are no better than no check, and one of four seeds is
    badly miscalibrated (ECE 0.150).
  - The policy-gradient signal is weak: the advantage |r − b| averages 0.02–0.06, because four
    checks for the same answer lead to almost the same confidence.
  - KL to the base model stays small (0.1–0.3 nats per check at β = 0.05; it grows to about
    3 nats by the end at β = 0.005).
  - Checks stay fluent and on-topic in both arms. No drift or codes appeared in 128 steps.
- **The check's content matters little, even where the arm helps.**
  - For the frozen arm, re-scoring the same adapter with another question's check costs 0.007 AUROC.
    Dropping the check line costs 0.011.
  - The frozen arm's adapter scores 0.828 AUROC *without* a check, against 0.796 for the no-check
    arm. Its advantage therefore seems to come from training on four varied contexts per answer,
    not from reading the check at test time.
  - Confound: the frozen and trained arms average the confidence loss over four sequences per
    question, while none and filler use one.
- **A fixed filler line is no better than nothing.** That rules out "more positions or compute" as
  the source of any gain.

**Verdict at this scale.** Letting the model write an unscored check before the confidence did not
improve calibration (ECE). A frozen, base-model check gave a small, fairly consistent ranking gain
(about +0.03 AUROC), but most of it survives removing the check at test time, so it is probably a
training-data effect. Training the check with REINFORCE + KL gave nothing measurable in 128 steps.

**What would make this conclusive.**
- The frozen arm with G = 1 (one check per answer). That tests the augmentation explanation
  directly.
- More steps for the trained arm, so its weak signal can accumulate.
- A stronger check-level signal, e.g. G = 8, or a per-token baseline.
- Full training set and 3+ seeds, then test.
