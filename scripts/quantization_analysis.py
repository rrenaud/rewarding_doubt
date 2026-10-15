"""Offline quantization bounds and paired readout comparisons; no service calls.

Run from the repository root:
  python scripts/quantization_analysis.py [path/to/predictions.jsonl ...]
All logarithms and NLL differences are in nats. Model predictions are not truth.
"""
import argparse
import json
import math
from pathlib import Path
import random


EPS = 0.001  # src/rewarding_doubt/core.py
GRID = tuple(max(EPS, min(1 - EPS, k / 10)) for k in range(11))


def clip(p):
    return max(EPS, min(1 - EPS, p))


def cross_entropy(p, q):
    return -p * math.log(q) - (1 - p) * math.log1p(-q)


def entropy(p):
    return -sum(t * math.log(t) for t in (p, 1 - p) if t > 0)


def best_grid(p):
    return min(GRID, key=lambda q: cross_entropy(p, q))


def regret(p):
    """Oracle grid NLL minus oracle continuous NLL, with identical clipping."""
    return cross_entropy(p, best_grid(p)) - cross_entropy(p, clip(p))


def boundaries():
    return [math.log((1 - a) / (1 - b)) /
            (math.log(b / a) + math.log((1 - a) / (1 - b)))
            for a, b in zip(GRID, GRID[1:])]


def paired_comparison(rows, samples=5000):
    """Positive differences favor fractional; percentile paired bootstrap over rows."""
    rng = random.Random(0)
    p = [float(r['fractional']) for r in rows]
    y = [int(r['correct']) for r in rows]
    n = len(rows)
    if not n or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p):
        raise ValueError('Need nonempty, finite probability predictions')
    fractional = [cross_entropy(label, clip(x)) for label, x in zip(y, p)]
    result = {'n': n, 'fractional_nll': sum(fractional) / n,
              'plugin_grid_regret_not_a_true_bound': sum(map(regret, p)) / n}
    for name, qs in [('argmax', [float(r['argmax']) for r in rows]),
                     ('grid_projection', [best_grid(x) for x in p])]:
        losses = [cross_entropy(label, clip(q)) for label, q in zip(y, qs)]
        delta = [a - b for a, b in zip(losses, fractional)]
        boot = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n
                      for _ in range(samples))
        result[name] = {'nll': sum(losses) / n,
                        'nll_minus_fractional': sum(delta) / n,
                        'paired_bootstrap_95_percent':
                        [boot[int(.025 * samples)], boot[int(.975 * samples)]]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='*', type=Path)
    args = parser.parse_args()
    bs = boundaries()
    candidates = [0., EPS, *bs, 1 - EPS, 1.]
    worst = max(candidates, key=regret)
    # Composite midpoint integration; illustrative uniform prior, not empirical truth.
    count = 100000
    uniform = sum(regret((i + .5) / count) for i in range(count)) / count
    examples = []
    for p in (.61, .64, .65, .69):
        q = best_grid(p)
        examples.append({'p': p, 'optimal_grid_q': q,
                         'excess_nll': cross_entropy(p, q) - entropy(p)})
    result = {'capacity_nats': math.log(11), 'capacity_bits': math.log2(11),
              'eps': EPS, 'optimal_grid_boundaries': bs,
              'worst_oracle_grid_excess_nll': regret(worst), 'worst_at_p': worst,
              'uniform_prior_mean_oracle_grid_excess_nll': uniform,
              'examples': examples, 'evaluations': {}}
    for path in args.predictions:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result['evaluations'][str(path)] = paired_comparison(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
