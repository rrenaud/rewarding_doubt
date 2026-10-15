# Turning confidence precision into betting value

There is no dollar EV attached to a millinat without specifying odds, stakes,
and the betting policy. Two useful games give exact interpretations. These are
toy calculations with no fees, spread, or stake constraints beyond those stated.

## Fixed-size bets against uniformly distributed prices

Let Y be correctness, p its true probability, q our reported confidence, and
r an independent market price uniform on [0,1]. Buy exactly one YES contract
if q > r, otherwise one NO contract. A YES costs r dollars and pays $1 if Y=1;
a NO costs 1-r dollars and pays $1 if Y=0. This is one $1-payout contract per
opportunity, not a fixed $1 cash stake.

Integrating over r, profit for an observed binary outcome y is

    integral_0^q (y-r) dr + integral_q^1 (r-y) dr
      = 0.5 - (q-y)^2 dollars.

Consequently expected profit is 0.5 minus Brier score. Compared with knowing p,
the dollar EV penalty is exactly (p-q)^2 per opportunity. This makes Brier score
an economic decision metric for uniformly distributed prices.

On the 474 saved fractional + hinge evaluation answers:

| Readout | Brier score | Mean profit per opportunity |
|---|---:|---:|
| Full distribution's fractional mean | 0.168786 | 33.1214 cents |
| Argmax label | 0.173249 | 32.6751 cents |
| Uniform over eleven labels, read out as mean q=.5 | 0.250000 | 25.0000 cents |

Argmax loses **0.4463 cents per opportunity**, or **$4.46 per 1,000** opportunities
of this size. A 5,000-resample paired bootstrap over answers gives a 95% interval
of **$0.46–$8.50 per 1,000**. This is 5.50% of the fractional readout's incremental
profit over the q=.5 baseline, or 1.35% of its total profit. The positive profit
of the uninformative baseline comes from the assumed random market prices.
Uniform labels mean q=.5 here; sampling a random label would be a different policy.

These figures integrate over hypothetical prices using actual held-out outcomes;
they are not profits from observed market offers. The argmax penalty includes
readout distortion and is not an estimate of the optimal grid's irreducible penalty.
Intervals condition on this checkpoint and assume independent evaluation rows.

If true probabilities p are also uniform, nearest-grid reporting on
{0,.1,...,1} has E[(p-q)^2] = 1/1200. Its ideal fixed-size betting cost is
**0.08333 cents per opportunity**, or **$0.8333 per 1,000**. Here nearest-grid
rounding is optimal for squared error; the earlier log-loss-optimal grid has
different boundaries. The full-information policy earns 1/3 dollar per opportunity.

## Kelly betting: millinats are expected log-wealth penalties

Suppose instead we can allocate wealth across both outcomes at fair, complementary
prices r and 1-r. Using confidence q to allocate fractions q and 1-q gives wealth
multipliers q/r on YES and (1-q)/(1-r) on NO. This allocation is log-optimal if q
is the true probability. Expected log growth under the true p is

    g(p,q;r) = p ln(q/r) + (1-p) ln((1-q)/(1-r)).

Compared with the oracle allocation q=p, the shortfall is exactly

    g(p,p;r) - g(p,q;r) = KL(Bern(p) || Bern(q)).

Thus **one millinat of excess NLL is .001 less expected log wealth per bet**.
The price r cancels from the comparison. This is the information/growth connection
in [Kelly (1956)](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/Kelly1956-1uwz47o.pdf).
It concerns log-optimal growth, not arithmetic expected dollars.

For uniformly distributed true probabilities, our earlier clipped grid comparison
costs **4.881 millinats per bet**, relative to the equally clipped continuous
oracle. The corresponding ratio of geometric-mean wealth is

    exp(-N * 0.004881373).

That is about **0.487% lower geometric growth factor per bet**, and **61.4% as much
geometric-mean wealth after 100 bets**. These are ratios of geometric means, not
expected dollar wealth ratios or a promise about a finite realized trajectory.
At even-money prices, an unclipped uniform-p oracle has 193.147 millinats of
expected log growth per bet over q=.5; the grid penalty is about 2.53% of that
growth edge. Shared .001 clipping makes a negligible additional difference here.

On the saved fractional checkpoint, the empirical argmax-minus-fractional NLL
gap is **15.908 millinats per answer**. Under the same clipped allocation rule,
that is also the average log-wealth advantage of the fractional readout on those
outcomes. Extrapolating a population growth rate from that finite sample would
require further validation.

## A smart bettor may recalibrate first

All comparisons above act on the stated q directly. A bettor who knows the true
conditional correctness frequency for each argmax label should instead use
E[Y | label]. This can remove calibration error while leaving information lost
within a label unresolved. Likewise a fractional prediction can be recalibrated.
To measure the value to such a bettor, fit both recalibration maps on a separate
development split, then compare betting returns on fresh held-out data. Our
current raw-readout gaps do not establish the gap between those optimized policies.
