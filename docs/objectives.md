# Confidence objectives: formulas and trade-offs

This note explains the three training objectives (`discrete-ppo`, `discrete-exact`,
`fractional`) and the three confidence readouts (`sampled`, `argmax`, `fractional`)
implemented in `src/rewarding_doubt/core.py` and `cli.py`, and when each is the
better choice.

## Setup

For each question $x$ the base model has already produced an answer $a$ (cached, never
regenerated during training). The answer is graded once: $y \in \{0,1\}$ is 1 if the
maximum word-overlap F1 against any reference alias exceeds 0.5. Only the confidence
that follows `Answer: a, Confidence: ` is trained.

There are eleven valid confidence strings, $k \in \{0,\dots,10\}$, each meaning
confidence $c_k = k/10$. Each one is scored as a complete token sequence ending in EOS.
This matters because Qwen tokenizes `10` as `1`,`0`, so scoring only its first token
would confuse it with `1`. Let $\ell_k$ be the summed log-probability of string $k$ and
renormalize over the eleven:

$$
\pi_k = \frac{e^{\ell_k}}{\sum_{j=0}^{10} e^{\ell_j}}
\qquad\text{(conditional on emitting a valid confidence)}
$$

The total valid mass $\sum_j e^{\ell_j}$ is reported separately as `valid_confidence_mass`.
It stays above 0.99 in practice.

### Reward: the clipped log score

$$
S(p, y) = y \log \tilde p + (1-y)\log(1-\tilde p), \qquad \tilde p = \operatorname{clip}(p, 0.001, 0.999)
$$

This is the negative of binary cross-entropy. It is a **strictly proper scoring rule**:
if the true probability that the answer is correct is $p^*$, then the expected score

$$
\mathbb{E}_{y\sim p^*}[S(p,y)] = p^*\log p + (1-p^*)\log(1-p)
$$

is uniquely maximized at $p = p^*$. In other words, reporting your honest belief maximizes
the expected reward.

`--reward paper` rescales this affinely to $[-1, 1]$:

$$
R(p,y) = 2\,\frac{S(p,y) - \log 0.001}{\log 0.999 - \log 0.001} - 1
$$

An affine rescaling does not change the optimum; it only changes gradient magnitude
relative to the learning rate. The `released` variant adds a constant correctness
bonus $2.5y$. Within a fixed answer that bonus is the same for every confidence, so it
does not change which confidence is best.

## The three training objectives

### 1. `discrete-ppo`: sample, then reinforce

This is the paper's approach. For each prompt, draw $G=8$ unrestricted continuations
$k_1,\dots,k_G \sim \pi_\theta(\cdot \mid x, a)$ and give each one a reward. A continuation
that doesn't parse as a valid confidence gets $-3$.

$$
r_i = \begin{cases} R(c_{k_i}, y) & \text{valid} \\ -3 & \text{invalid}\end{cases}
\qquad
A_i = \frac{G}{G-1}\Big(r_i - \tfrac1G\textstyle\sum_j r_j\Big) = r_i - \tfrac{1}{G-1}\textstyle\sum_{j\ne i} r_j
$$

The advantage uses a leave-one-out baseline: each sample is compared with the mean of
the *other* samples. The baseline doesn't depend on sample $i$, so the gradient stays
unbiased. The PPO loss is

$$
\mathcal L = -\frac{1}{G}\sum_i \min\!\big(\rho_i A_i,\ \operatorname{clip}(\rho_i, 1\pm\epsilon) A_i\big),
\qquad \rho_i = \frac{\pi_\theta(k_i)}{\pi_{\text{old}}(k_i)}
$$

We take exactly one update per rollout, so $\rho_i = 1$ and the clipping never activates.
The update reduces to REINFORCE with a baseline:

$$
\nabla J \approx \frac1G\sum_i A_i\,\nabla\log\pi_\theta(k_i)
$$

This is a Monte Carlo estimate of the gradient of the **expected reward**
$\mathbb E_{k\sim\pi}[R(c_k,y)]$.

### 2. `discrete-exact`: the same objective, with the expectation computed exactly

There are only eleven possible outcomes, so the expectation can be computed exactly
instead of sampled:

$$
J_{\text{exact}} = \sum_{k=0}^{10} \pi_k\, R(c_k, y)
$$

Its gradient is the exact version of what PPO estimates:

$$
\nabla J_{\text{exact}} = \sum_k \pi_k\,\big(R(c_k,y) - J_{\text{exact}}\big)\,\nabla \log \pi_k
$$

All eleven branches are teacher-forced on every step, so sampling contributes no noise.
This approximates PPO's objective with two small differences: $\pi$ is renormalized over
valid strings, so there is no invalid-format penalty, and the gradient is exact rather
than estimated.

**What it converges to.** Average over questions whose true correctness probability is
$p^*$. The objective is then *linear* in $\pi$:

$$
\mathbb E_y[J_{\text{exact}}] = \sum_k \pi_k \big[p^*\log c_k + (1-p^*)\log(1-c_k)\big]
$$

A linear function over the probability simplex is maximized at a corner. The optimal
policy therefore puts all its mass on the single grid level $c_k$ that scores best for
$p^*$. The result is a **deterministic, quantized** confidence. PPO has the same
optimum.

### 3. `fractional`: score the mean confidence

Define the fractional confidence as the expected level under $\pi$:

$$
\bar p = \sum_{k=0}^{10} \pi_k\, c_k
\qquad\qquad
J_{\text{frac}} = R(\bar p, y)
$$

This is **reward of the mean**, not mean of the reward. Its gradient goes through the
normalization and the weighted mean, using Tinker's custom-loss callback:

$$
\nabla J_{\text{frac}} = R'(\bar p, y)\sum_k \pi_k\,(c_k - \bar p)\,\nabla\log\pi_k,
\qquad
R'(\bar p, y) \propto \frac{y}{\bar p} - \frac{1-y}{1-\bar p}
$$

The structure is simple: one scalar ("raise $\bar p$" or "lower $\bar p$") multiplies each
level's distance from the mean. A common shortcut is to detach $\bar p$ and use
$R(\bar p,y)$ as a REINFORCE reward. That is *not* this gradient; it would drop the
$(c_k-\bar p)$ structure.

**What it converges to.** Because $S$ is strictly proper,
$\mathbb E_{y\sim p^*}[R(\bar p,y)]$ is maximized at $\bar p = p^*$ exactly. That optimum
is continuous, not snapped to the 0.1 grid. Many distributions $\pi$ have the right mean,
so $\pi$ is free to stay spread out.

### How the two exact objectives differ

The log score is concave in $p$, so by Jensen's inequality, for every $\pi$ and $y$:

$$
R(\bar p, y) \;\ge\; \sum_k \pi_k R(c_k, y)
\quad\text{i.e.}\quad J_{\text{frac}} \ge J_{\text{exact}}
$$

The gap is the **penalty `discrete-exact` places on spread**. `discrete-exact` rewards
the model for committing to one level; `fractional` only cares that the mean is right.
For the same $\pi$, both gradients push mass toward levels that score well. But
`discrete-exact` weights each level by *its own* reward, while `fractional` weights it by
its distance from the mean times a single slope.

## The three readouts

Any checkpoint, including the untrained base model, can be read out three ways:

| Readout | Formula | Needs |
|---|---|---|
| `sampled` | $c_k$ with $k\sim\pi_\theta$ (temperature 1; invalid strings excluded and counted) | ordinary generation |
| `argmax` | $c_{\arg\max_k \pi_k}$ | log-probs of all 11 strings |
| `fractional` | $\bar p = \sum_k \pi_k c_k$ | log-probs of all 11 strings |

Evaluating every checkpoint with every readout separates two kinds of gain: improvement
that comes only from reading the confidence out differently (no training needed), and
improvement that requires training. In the smoke test the base model's **fractional
readout alone** lifted AUROC from 0.56 (argmax) to 0.87. The model's preference between
"10" and "9" carries real information about correctness, and rounding throws it away.
Calibration stayed poor, though (ECE ≈ 0.58): the mean was still about 0.99 while
accuracy was 39%.

Metrics:

- **ECE**: count-weighted mean of |accuracy − confidence| over 11 equal-width bins.
- **Brier**: $\frac1N\sum (p-y)^2$.
- **NLL**: $-\frac1N\sum S(p,y)$.
- **AUROC**: ranking quality only; it ignores calibration.

## Which to prefer

| | `discrete-ppo` | `discrete-exact` | `fractional` |
|---|---|---|---|
| Objective | $\mathbb E_\pi[R(c_k,y)]$, sampled | $\mathbb E_\pi[R(c_k,y)]$, exact | $R(\mathbb E_\pi[c_k], y)$ |
| Optimum | point mass on best grid level | point mass on best grid level | $\bar p = p^*$, continuous |
| Gradient noise | high (Monte Carlo) | none | none |
| Signal when model is saturated at "10" | none if all $G$ samples agree | small but nonzero | small but nonzero |
| Calibrates the *emitted* number | yes | yes | not necessarily |
| Inference requirement | sample text | sample text (or argmax) | 11 log-prob queries |
| Training cost (this repo, per step) | ~17 s: sample, then train | ~9 s: 11 teacher-forced branches | ~9 s: 11 teacher-forced branches |
| Handles open-ended outputs | yes | no (needs enumerable outputs) | no (needs enumerable, numeric outputs) |

**Prefer `discrete-ppo`** when the output space can't be enumerated: free-text
confidence, the answer and the confidence generated together on-policy (as in the paper),
multi-answer settings, or format compliance that must itself be learned. It is the
general tool. Its weakness is variance, and **vanishing signal at saturation**. If the
model puts probability $\pi_{10}$ on "10", all $G$ samples agree with probability
$\pi_{10}^G$; every advantage is then zero and that prompt contributes nothing. In the
smoke test the base model had $\pi_{10}\approx 0.99$ and PPO barely moved
(one substantive update out of ten).

**Prefer `discrete-exact`** when the outputs can be enumerated, as here, and you will
deploy by *reading the emitted integer*, for example a chat user who sees
"Confidence: 7". It optimizes the same thing PPO does, without sampling noise, and its
optimum is a sharp policy whose sampled and argmax outputs agree with its calibrated
level. It is the clean controlled comparison for `fractional`, since the only change is
where the expectation sits relative to the score.

**Prefer `fractional`** when you can afford eleven log-prob queries at inference and want
the best probability estimate. Its optimum is the true $p^*$ instead of the nearest grid
point, and it keeps the within-distribution ordering information that gave the large
AUROC gain above. The cost is that the emitted token is no longer what's calibrated.
A 50/50 mix of "0" and "8" has $\bar p = 0.4$ and is perfectly fine for this objective,
but a sampled reply would print 0 or 8. Check its `sampled` and `argmax` metrics before
trusting the text output. Because it never penalizes spread, it may also learn hedged
distributions that look unusual when sampled.

**Practical rule:** if the consumer is a program that can query log-probs, train and read
out `fractional`. If the consumer is a person reading generated text, train
`discrete-exact` (or PPO when outputs can't be enumerated) and use `sampled`/`argmax`.
In either case, report all three readouts so readout gains aren't mistaken for training
gains.

## Evidence so far

- **Smoke test** (128/128, 10 updates, one seed): `fractional` ≈ `discrete-exact` >
  `discrete-ppo` ≈ base on ECE/Brier/NLL (fractional readout ECE 0.559 / 0.561 / 0.579 /
  0.582). The differences are small and come from a single seed. See
  [the smoke report](../runs/smoke-20260930T222803Z/report.md).
- **Pilot** (1,024/512, ≈240 updates, seed 1;
  [report](../runs/pilot-20261001T002945Z/report.md)). Held-out ECE, fractional readout:
  base 0.479, `discrete-ppo` 0.474, `discrete-exact` 0.173, `fractional` 0.132. However,
  the `fractional` checkpoint's valid confidence mass collapsed to 0.026: it prints a digit
  and keeps writing instead of emitting EOS, so 98% of its sampled replies are invalid.
  `discrete-exact` kept mass at 0.999 and calibrated all three readouts. PPO stayed within
  noise of the base model, as the saturation argument predicts.

- **Hinge rerun** (same data and seed, `1.0·relu(log 0.95 − log M)` on both exact arms;
  [report](../runs/hinge-20261001T023959Z/report.md)). Fractional keeps valid mass at 1.000
  with no invalid samples. Held-out ECE is fractional 0.129 (sampled 0.120) vs discrete-exact
  0.123; NLL is 0.512 vs 0.538. The arms are tied within run-to-run noise: discrete-exact
  alone moved 0.173 → 0.123 between two same-seed runs. The hinge fired on only 7 early steps.

- **Paper-faithful PPO** (TRL 0.8.6 loop with on-policy answers and KL to base, verified against
  TRL's code; [report](../runs/paper-ppo-20261001T051018Z/report.md)). It does learn: ECE fell
  from 0.473 to 0.308 and NLL from 2.40 to 1.04 at the final checkpoint, which the paper's
  best-training-reward rule selects. That is still about 2.5× the exact arms' ECE. Its held-out
  learning curve swings between 0.12 and 0.41 ECE across snapshots, so it can reach the exact
  arms' level (step 200) but does not stay there. Training took 1 h 55 min, against 34 min.

- **Released code on Modal** (Llama-3-8B, exact pinned environment, our 1,024/512 questions;
  [report](../runs/modal-stage2-20261001T082216Z/report.md)). Its base model reproduces the
  paper's Table 1 (ECE 0.333 vs 0.346, accuracy 63.3% vs 63.1%). After 256 steps, ECE falls
  smoothly from 0.333 to 0.074 (AUROC 0.56 → 0.77), much better than our Tinker `paper-ppo`
  (0.474 → 0.300 on Qwen). The model and the missing value head are confounded. Under the same
  released evaluation protocol, our exact objectives reach 0.117/0.120 ECE on Qwen, but
  cross-model comparisons are not valid. The decisive test is the exact objectives on Llama.

## Caveat both exact objectives share: nothing anchors the format

$\pi$ is renormalized over the eleven strings, so both exact objectives are invariant to
the total valid mass $\sum_k e^{\ell_k}$. A model can lower every $\ell_k$ (for example by
not emitting EOS after the digit) at no cost to the objective. In the pilot this drift hit
`fractional` but not `discrete-exact`. One seed can't establish why; a plausible guess is
that `fractional` tolerates spread, so suppressing some branches is a cheap way to move
$\bar p$. The fix, now the default for both exact modes, is a hinge on the valid mass:

$$
\mathcal L' = \mathcal L + \lambda\,\operatorname{relu}\!\Big(\log\tau - \log\sum_k e^{\ell_k}\Big),
\qquad \lambda = 1,\ \tau = 0.95
$$

It is zero while the model emits a valid string with probability at least $\tau$, so it
leaves the calibration objective alone in normal operation. An always-on $-\log M$ term
would instead keep nudging $\pi$ toward its current mode, since $\partial \log M/\partial\ell_k = \pi_k$. Always report valid mass and invalid sampling rate alongside calibration.
PPO does not have this problem, because invalid samples are penalized directly.
