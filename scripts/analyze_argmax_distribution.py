"""Analyze saved full confidence distributions, without inference or training.

Run from any directory: python scripts/analyze_argmax_distribution.py
Requires numpy, scipy, matplotlib, markdown (same as plot_quantization.py).
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_quantization import ROOT, FIGURES, BLUE, ORANGE, build_html, save


LEVELS = np.arange(11) / 10
MAIN = {
    'Fractional + hinge': 'runs/hinge-20261001T023959Z/fractional-s1-eval/predictions.jsonl',
    'Discrete-exact + hinge': 'runs/hinge-20261001T023959Z/discrete-exact-s1-eval/predictions.jsonl',
}


def interval(values):
    rng = np.random.default_rng(20261001)
    draws = rng.integers(len(values), size=(5000, len(values)))
    return np.quantile(np.mean(values[draws], axis=1), [.025, .975]).tolist()


def analyze(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    logs = np.array([r['candidate_logprobs'] for r in rows])
    assert logs.shape == (len(rows), 11) and np.isfinite(logs).all()
    shifted = logs - logs.max(axis=1, keepdims=True)
    logpi = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    pi = np.exp(logpi)
    top = pi.max(axis=1)
    argmax = LEVELS[pi.argmax(axis=1)]
    mean = pi @ LEVELS
    entropy = -(pi * logpi).sum(axis=1)
    sd = np.sqrt((pi * (LEVELS - mean[:, None]) ** 2).sum(axis=1))
    displacement = (pi * abs(LEVELS - argmax[:, None])).sum(axis=1)
    valid = np.exp(logs.max(axis=1) + np.log(np.exp(shifted).sum(axis=1)))
    y = np.array([r['correct'] for r in rows], dtype=float)
    assert np.isin(y, [0, 1]).all()
    assert np.allclose(mean, [r['fractional'] for r in rows], atol=1e-7)
    assert np.allclose(argmax, [r['argmax'] for r in rows], atol=1e-7)
    assert np.allclose(valid, [r['valid_confidence_mass'] for r in rows], atol=1e-7)
    def nll(p):
        p = np.clip(p, .001, .999)
        return -1000 * (y * np.log(p) + (1 - y) * np.log1p(-p))
    delta = nll(argmax) - nll(mean)
    summary = dict(n=len(rows), mean_top_probability=float(top.mean()),
                   median_top_probability=float(np.median(top)),
                   fraction_top_below_half=float((top < .5).mean()),
                   mean_discarded_mass=float((1 - top).mean()),
                   mean_entropy_mnat=float(1000 * entropy.mean()),
                   mean_effective_labels=float(np.exp(entropy).mean()),
                   mean_abs_argmax_mean_gap_pp=float(100 * abs(mean - argmax).mean()),
                   mean_distribution_sd_pp=float(100 * sd.mean()),
                   mean_expected_displacement_pp=float(100 * displacement.mean()),
                   mean_valid_mass=float(valid.mean()),
                   mean_unconditional_top_probability=float((top * valid).mean()),
                   argmax_minus_fractional_nll_mnat=float(delta.mean()),
                   paired_bootstrap_95_mnat=interval(delta), strata=[])
    for lo, hi in [(0, .5), (.5, .75), (.75, 1.000001)]:
        mask = (top >= lo) & (top < hi)
        if mask.any():
            summary['strata'].append(dict(top_probability_range=[lo, min(hi, 1)],
                n=int(mask.sum()), mean_abs_gap_pp=float(100 * abs(mean[mask] - argmax[mask]).mean()),
                nll_difference_mnat=float(delta[mask].mean()), ci95_mnat=interval(delta[mask])))
    detail = [dict(id=r.get('id', str(i)), correct=bool(y[i]),
                   top_probability=float(top[i]), valid_mass=float(valid[i]),
                   entropy_mnat=float(1000 * entropy[i]), stddev_pp=float(100 * sd[i]),
                   argmax=float(argmax[i]), fractional=float(mean[i]),
                   abs_gap_pp=float(100 * abs(mean[i] - argmax[i])),
                   nll_difference_mnat=float(delta[i])) for i, r in enumerate(rows)]
    return summary, detail, pi


def figures(main):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for (name, (_, details, _)), color in zip(main.items(), [ORANGE, BLUE]):
        top = np.sort([r['top_probability'] for r in details])
        axs[0].step(top, 100 * np.arange(1, len(top) + 1) / len(top), where='post',
                    label=name, color=color)
        axs[1].scatter([r['top_probability'] for r in details], [r['abs_gap_pp'] for r in details],
                       alpha=.35, s=15, label=name, color=color)
    axs[0].axvline(.5, color='#777', ls='--')
    axs[0].set(xlabel='Probability of most likely confidence label', ylabel='Examples at or below this probability (%)',
               title='How often does the top label lack a majority?', xlim=(0, 1), ylim=(0, 100))
    axs[0].legend(fontsize=9)
    axs[1].set(xlabel='Probability of most likely confidence label', ylabel='|Argmax − fractional mean| (percentage points)',
               title='Diffuse outputs can conceal larger shifts', xlim=(0, 1))
    for ax in axs:
        ax.grid(alpha=.15)
    save(fig, '05-empirical-argmax-suppression')

    name = 'Fractional + hinge'
    _, details, pi = main[name]
    # Label-free selection: most diffuse, largest shift, then median entropy.
    order = np.argsort([r['entropy_mnat'] for r in details])
    selected = [int(np.argmin([r['top_probability'] for r in details])),
                int(np.argmax([r['abs_gap_pp'] for r in details])), int(order[len(order) // 2])]
    titles = ['Lowest top-label probability', 'Largest argmax-to-mean shift', 'Median-entropy example']
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for ax, idx, title in zip(axs, selected, titles):
        r = details[idx]
        ax.bar(LEVELS, pi[idx], width=.065, color=BLUE)
        ax.axvline(r['argmax'], color='#333', ls=':', lw=2, label=f"Argmax {r['argmax']:.2f}")
        ax.axvline(r['fractional'], color=ORANGE, ls='--', lw=2, label=f"Mean {r['fractional']:.3f}")
        ax.set(xlim=(-.06, 1.06), ylim=(0, 1), xlabel='Confidence level',
               title=f"{title}\n{r['id']}: top mass {r['top_probability']:.1%}")
        ax.legend(fontsize=9, loc='upper left')
    axs[0].set_ylabel('Conditional probability of confidence label')
    fig.supxlabel('Actual saved distributions from fractional + hinge; selected without correctness labels.', fontsize=10)
    save(fig, '06-observed-confidence-distributions')
    return [dict(selection=t, **details[i], probabilities=pi[i].tolist()) for t, i in zip(titles, selected)]


def report(all_results, main):
    lines = ['# What argmax discards in the saved confidence distributions', '',
        'We can measure this directly from the saved probabilities of all eleven complete confidence strings. '
        'The main comparison uses the two hinge-trained checkpoints on 474 held-out answers each. '
        '**A low top-label probability means argmax hides substantial probability mass on other confidence levels.** '
        'Whether that lost detail matters for correctness is a separate, testable question.', '',
        '## Measurements', '',
        'For normalized probabilities pi[k], levels c[k] = k/10, and k* = argmax pi[k]:', '',
        '```text\nTop-label probability:       m = pi[k*]\nMass outside the top label:  1 - m\nEntropy (millinats):         -1,000 sum_k pi[k] ln pi[k]\nEffective label count:      exp(entropy / 1,000)\nArgmax confidence:          a = c[k*]\nFractional confidence:      mu = sum_k pi[k] c[k]\nSpread:                     sqrt(sum_k pi[k] (c[k] - mu)^2)\nMean displacement:          sum_k pi[k] |c[k] - a|\n```', '',
        'The mass outside the top label is also the total variation distance between pi and '
        'a point mass at argmax. Entropy describes variability in the emitted label, while spread '
        'and displacement account for how far apart the confidence levels are. '
        'The entropy here is **not** a measured mutual-information loss about true correctness.', '',
        'Top-label probability is conditional on one of the eleven canonical strings being emitted. '
        'We normalize their saved sequence log probabilities with log-sum-exp and report valid mass '
        'separately. These are complete strings including EOS, not individual tokenizer tokens.', '',
        '| Checkpoint | Mean top probability | Top < 50% | Mass outside top | Entropy (mnat) | Mean absolute shift (pp) | Mean spread (pp) | Valid mass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, (s, _, _) in main.items():
        lines.append(f"| {name} | {s['mean_top_probability']:.1%} | {s['fraction_top_below_half']:.1%} | "
            f"{s['mean_discarded_mass']:.1%} | {s['mean_entropy_mnat']:.1f} | "
            f"{s['mean_abs_argmax_mean_gap_pp']:.2f} | {s['mean_distribution_sd_pp']:.2f} | {s['mean_valid_mass']:.4%} |")
    lines += ['', 'Here pp means percentage points on the 0–100% correctness-confidence scale.', '',
        '![Empirical top-label probabilities and argmax-to-mean shifts.](figures/confidence-quantization/05-empirical-argmax-suppression.png)', '',
        'The fractional-trained model retains substantially more variation among confidence labels. '
        'This is consistent with its objective caring about the mean, while the discrete objective '
        'rewards committing to a grid level. The figure describes these checkpoints; it does not '
        'isolate a causal training effect across repeated runs.', '',
        '## Does discarding the distribution hurt held-out predictions?', '',
        'For each answer we compute NLL(argmax) − NLL(fractional), with the same .001 clipping. '
        'Positive values favor the fractional mean. All values below are **millinats per answer**. '
        'The 95% intervals use 5,000 paired bootstrap resamples of examples with a fixed seed.', '',
        '| Checkpoint | Top-probability group | n | Mean absolute shift (pp) | NLL difference (mnat) | 95% interval |',
        '|---|---|---:|---:|---:|---|']
    for name, (s, _, _) in main.items():
        lo, hi = s['paired_bootstrap_95_mnat']
        lines.append(f"| {name} | All | {s['n']} | {s['mean_abs_argmax_mean_gap_pp']:.2f} | "
                     f"{s['argmax_minus_fractional_nll_mnat']:.2f} | [{lo:.2f}, {hi:.2f}] |")
        for g in s['strata']:
            a, b = g['top_probability_range']
            lo, hi = g['ci95_mnat']
            lines.append(f"| {name} | {a:.0%}–{b:.0%} | {g['n']} | {g['mean_abs_gap_pp']:.2f} | "
                         f"{g['nll_difference_mnat']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines += ['', 'Groups are lower-inclusive, upper-exclusive, except the last includes 100%. '
        'These exploratory intervals assume independent examples, condition on the trained checkpoint, '
        'and do not adjust for multiple comparisons or capture training-seed variation. '
        'Small groups can be especially unstable.', '',
        '## Actual example distributions', '',
        '![Three actual confidence distributions with argmax and fractional mean.](figures/confidence-quantization/06-observed-confidence-distributions.png)', '',
        'Examples are chosen using only distribution diagnostics: lowest top-label probability, '
        'largest argmax-to-mean shift, and median entropy. The first two deliberately show extremes; '
        'the third provides a central comparison. IDs and all eleven probabilities are in the JSON output.', '',
        '## What this does and does not establish', '',
        'Your proposed diagnostic detects real information discarded by argmax. It is useful to '
        'measure top-label mass **together with numerical spread and argmax-to-mean shift**. '
        'For example, probabilities (.25, .50, .25) on levels (.5, .6, .7) have only 50% mass '
        'on the top label, but both argmax and mean equal .6. There is spread, yet switching '
        'between these two scalar readouts changes no predicted correctness probability.', '',
        'A distribution over verbal confidence reports is also not automatically a calibrated '
        'posterior over the true probability of correctness. Language variation and the training '
        'objective can create spread. Fractional training constrains the mean and leaves many '
        'different shapes equally good. Consequently, high entropy alone does not prove '
        'useful epistemic uncertainty. The paired held-out NLL comparison tests the practical '
        'benefit of preserving the mean. The mean itself still discards the distribution’s shape.', '',
        '## All saved checkpoints', '',
        'These are descriptive summaries, not matched comparisons across different runs. '
        'In particular, the no-hinge fractional pilot has low valid mass: a diffuse distribution '
        'conditional on rare valid outputs can be misleading without that column.', '',
        '| Prediction file | n | Mean top probability | Top < 50% | Entropy (mnat) | Mean shift (pp) | Valid mass |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for path, s in all_results.items():
        lines.append(f"| `{path}` | {s['n']} | {s['mean_top_probability']:.1%} | "
            f"{s['fraction_top_below_half']:.1%} | {s['mean_entropy_mnat']:.1f} | "
            f"{s['mean_abs_argmax_mean_gap_pp']:.2f} | {s['mean_valid_mass']:.3%} |")
    lines += ['', '## Reproduce', '', '```bash\npython scripts/analyze_argmax_distribution.py\n```', '',
        'The script checks that reconstructed means, argmax values, and valid mass match saved '
        'predictions. Outputs: [summary JSON](argmax-distribution-results.json), '
        '[per-example CSV](argmax-distribution-rows.csv), PNG/SVG plots, and this '
        '[offline HTML report](argmax-distribution.html). No new model calls or training are needed.']
    source = ROOT / 'docs/argmax-distribution.md'
    source.write_text('\n'.join(lines) + '\n')
    build_html(source)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_results, arrays, csv_rows = {}, {}, []
    for path in sorted((ROOT / 'runs').glob('**/predictions.jsonl')):
        result = analyze(path)
        relative = str(path.relative_to(ROOT))
        all_results[relative] = result[0]
        arrays[relative] = result
        csv_rows.extend(dict(prediction_file=relative, **r) for r in result[1])
    selected = {name: arrays[path] for name, path in MAIN.items()}
    examples = figures(selected)
    (ROOT / 'docs/argmax-distribution-results.json').write_text(
        json.dumps({'checkpoints': all_results, 'selected_examples': examples}, indent=2) + '\n')
    with (ROOT / 'docs/argmax-distribution-rows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report(all_results, selected)
    for name, (summary, _, _) in selected.items():
        print(name, json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
