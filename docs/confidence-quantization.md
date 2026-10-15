# How much does an eleven-level confidence output lose?

An emitted confidence label carries at most **ln(11) = 2.397895 nats
(3.459432 bits)** about underlying confidence. This is a communication bound,
not a 2.398-nat penalty in prediction NLL. The NLL cost depends on where the true
probabilities lie and how labels are decoded. There is no universal positive NLL
penalty: the true probabilities could already lie on the grid.

This repository uses eleven complete strings `0` through `10`, not necessarily
eleven tokenizer tokens (`10` uses two tokens for the current model). Its existing
fractional objective already avoids the discrete readout restriction by reading
the distribution over all eleven strings.

## Communication and predictive loss are different quantities

Let P be the underlying probability of correctness given the available information,
Y the binary correctness outcome, and K the single emitted confidence label.
Assume Y | P is Bernoulli(P), and K is computed from P, possibly with independent
randomness. For an eleven-valued K,

    I(P; K) <= H(K) <= ln(11).

For a deterministic encoder, I(P; K) = H(K); the capacity is attained only with
equally frequent labels. This applies to the confidence channel alone. Extra text,
repeated draws, or access to logits provides a different channel. Conditional on
any shared context C, the corresponding bound I(P; K | C) <= ln(11) still holds.
See the [Stanford information theory notes](https://web.stanford.edu/~kedart/ee376a_winter1617/Lecture_3.pdf)
for the finite-alphabet entropy bound.

If a decoder can assign each bin its calibrated value m(K) = E[P | K], then its
minimum expected log loss is H(Y | K), whereas knowing P gives H(Y | P). Thus

    H(Y | K) - H(Y | P)
      = E[ KL(Bern(P) || Bern(m(K))) ]
      = I(Y; P | K).

That is the predictive information lost by binning, measured in nats per answer.
It is distribution-dependent and at most ln(2), since Y is binary. Reporting a
fixed grid value q(K) instead adds a calibration cost:

    E[KL(Bern(P) || Bern(q(K)))]
      = I(Y; P | K) + E[KL(Bern(m(K)) || Bern(q(K)))].

These statements use ordinary, unclipped log loss with appropriate endpoint
limits. They do not subtract differential entropy from ln(11); continuous
confidence has no distribution-free finite number of information nats.

## The relevant grid penalty in NLL

For a true probability p and reported probability q,

    L(p,q) = -p ln(q) - (1-p) ln(1-q)
    L(p,q) - h(p) = p ln(p/q) + (1-p) ln((1-p)/(1-q)).

This is Bernoulli KL divergence. Log score and its associated KL divergence are
described in [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
For a fixed grid Q, the ideal grid-only forecaster has excess loss

    D_Q = E_P[ min_{q in Q} KL(Bern(P) || Bern(q)) ].

Consequently its expected NLL is at least E[h(P)] + D_Q. This is the population
oracle floor for that representation, not a floor one can calculate from binary
labels alone. Calibration, estimation error, and imperfect training add other
loss. A finite empirical test score can fluctuate below its population expectation.

The log-loss-optimal switch between adjacent positive levels a < b is

    t(a,b) = ln((1-a)/(1-b)) / [ln(b/a) + ln((1-a)/(1-b))].

For .6 and .7, t = .6511107902. Hence .61 and .69 are distinguishable as .6 and .7;
**.61 and .64** are a better example of two values mapped to the same label.
The model is not explicitly programmed to round: these are oracle choices under
expected logarithmic reward, without other optimization constraints.

| True p | Best grid q | Excess NLL, nats/answer |
|---:|---:|---:|
| .61 | .6 | .000208929 |
| .64 | .6 | .003374868 |
| .65 | .6 | .005291773 |
| .69 | .7 | .000236616 |

Locally, KL(Bern(p) || Bern(p+e)) is approximately e²/[2p(1-p)].
With spacing delta and approximately uniform rounding error within an interior
bin, the average is approximately delta²/[24p(1-p)]. Near p=.65 with delta=.1,
this is .00183 nats/answer. The approximation fails near the endpoints.

## The repository clips scores at epsilon = .001

`core.py` clips probabilities to [.001,.999]. For NLL the effective grid is

    Q_epsilon = {.001, .1, .2, ..., .9, .999}.

The fair representation-only comparison uses the same clipping for both arms:

    d_epsilon(p) = min_q L(p,q) - L(p, clip(p)).

The accompanying script computes these oracle quantities:

* Uniform P on [0,1]: E[d_epsilon(P)] = **.00488137 nats/answer**.
  This is an illustrative prior, not an assumption established by the experiment.
* Worst p: d_epsilon(p) = **.04772167 nats/answer**, at p=.02215933 or
  p=.97784067, the transitions into the endpoint bins.
* The minimum is zero. There is **no nonzero distribution-free loss floor**.

The worst value follows by checking all adjacent-level transitions and endpoints:
within each grid region the regret is convex for p in [.001,.999], so a maximum
occurs at a region boundary. Outside this interval the endpoint grid and clipped
continuous oracle coincide. The worst-case value bounds the extra loss of the
*optimal* grid forecaster, not the NLL of an arbitrary trained model.

Clipping also means strict propriety only holds in the interior: probabilities
below .001 or above .999 cannot be distinguished by this score. Neither readout
beats that shared clipping restriction. Report raw NLL for these comparisons;
the repository's affine-scaled rewards are not themselves nats.

## Why the fractional readout escapes this restriction

The implementation computes pi[k] from normalized complete-string probabilities,
then reports p_bar = sum_k pi[k] k/10. A probability vector is not one draw from
that vector, so its mean need not lie on the grid. For example:

    .61 = .9 * .6 + .1 * .7
    .69 = .1 * .6 + .9 * .7.

These distributions can encode different means even though a single sampled
label remains grid-valued. In ideal real arithmetic every interior mean is
representable; finite precision and model capacity still apply in practice.
Accessing all eleven likelihoods is additional information and inference work.

Expected discrete reward is linear in pi, so its unconstrained population optimum
puts mass on a best-scoring grid level (or tied levels). Fractional reward instead
optimizes the mean, whose ideal clipped value equals clip(p). Thus fractional
training *can* beat the optimal discrete representation whenever true probabilities
fall between levels. It is not guaranteed to learn them or generalize better.

## What the saved experiments support

Offline analysis of the hinge run, seed 1, on 474 fixed held-out answers:

| Training arm | Fractional NLL | Argmax NLL | Grid projection of fractional NLL |
|---|---:|---:|---:|
| fractional | .511602 | .527510 | .514550 |
| discrete-exact | .537617 | .541251 | .541566 |

Projection selects q minimizing KL(Bern(p_bar) || Bern(q)) without looking at the
label. It isolates converting the same scalar prediction to eleven levels more
directly than argmax, which need not be near the mean.

For the fractional-trained checkpoint, argmax minus fractional NLL is .015908
(paired row-bootstrap 95% interval [.004938, .027413]). Grid projection minus
fractional is only **.002948** (interval **[-.003279, .009247]**). These intervals
describe held-out-row variation conditional on this checkpoint and assume
independent rows; they do not capture training-seed variation or checkpoint
selection. The latter difference does not yet establish a population benefit
over this grid projection, much less over every possible discrete forecaster.

Substituting model confidences for true p gives a plug-in oracle regret of
.002741 nats/answer for the fractional arm. This is a diagnostic, **not a proven
floor**, because model confidences are not known true probabilities.

## Training recommendation

Run a longer **controlled comparison**, rather than assuming more steps help.
The saved hinge report shows upper-bin underconfidence after about 240 updates,
so longer training on the same 1,024-question cache could worsen generalization.
The current fractional-vs-discrete fractional-readout gap (.0260 NLL) cannot be
attributed solely to quantization: both of those readouts are continuous.

1. Use a separate development split to evaluate existing snapshots first. Keep
   the already-inspected held-out results exploratory and reserve a fresh final test.
2. Compare fractional and discrete-exact with matched data, compute accounting,
   several seeds, the valid-mass hinge, and checkpoint selection by dev NLL.
   If extending training, include more unique examples and a lower learning-rate
   condition, since extra passes alone can overfit.
3. Evaluate fractional, argmax, and grid-projected fractional readouts for every
   checkpoint. Report paired NLL differences, valid mass, and seed variation.
4. Expect a representation-only effect on the order of thousandths of a nat
   under the illustrative calculations above. Use pilot paired-loss variance to
   size a larger test set for a prespecified effect; 474 answers do not resolve
   the observed projection gap.

No new paid training was launched for this analysis.

## Reproduce locally

From this repository's root (standard library only):

```bash
python scripts/quantization_analysis.py \
  runs/hinge-20261001T023959Z/fractional-s1-eval/predictions.jsonl \
  runs/hinge-20261001T023959Z/discrete-exact-s1-eval/predictions.jsonl
```

Numerical results are saved in [quantization-results.json](quantization-results.json).
Uniform-prior integration uses 100,000 midpoint samples; bootstrap intervals use
5,000 resamples with a fixed RNG seed. The repo copy includes the source working
tree and Git history as copied, including local run artifacts. It is not synced
to subsequent updates of the original repo.
