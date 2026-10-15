"""Build publication-quality figures and a self-contained illustrated HTML note.

Run: python scripts/plot_quantization.py
Requires numpy, scipy, matplotlib, and markdown. No network or service calls.
"""
import base64
from html import escape
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta

from quantization_analysis import EPS, GRID, boundaries, regret


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'docs/figures/confidence-quantization'
BLUE, ORANGE, GREEN = '#2563a6', '#d06521', '#168577'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white', 'svg.fonttype': 'none'})


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def ce(p, q):
    return -p * np.log(q) - (1 - p) * np.log1p(-q)


def quantizer_figure():
    p = np.unique(np.r_[np.linspace(0, 1, 10001), boundaries()])
    losses = ce(p[:, None], np.array(GRID)[None, :])
    q = np.array(GRID)[losses.argmin(axis=1)]
    d = 1000 * (losses.min(axis=1) - ce(p, np.clip(p, EPS, 1 - EPS)))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    axs[0].plot(p, np.clip(p, EPS, 1 - EPS), color=BLUE, label='Continuous oracle (clipped)')
    axs[0].plot(p, q, color=ORANGE, label='Best of 11 levels')
    axs[0].set(xlabel='Underlying correctness probability p', ylabel='Reported probability',
               title='A. Distinctions disappear within each step', xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(loc='upper left', fontsize=9)
    axs[1].plot(p, d, color=ORANGE)
    axs[1].fill_between(p, d, color=ORANGE, alpha=.12)
    axs[1].set(xlabel='Underlying correctness probability p', ylabel='Extra NLL (millinats / answer)',
               title='B. The cost depends strongly on p', xlim=(0, 1), ylim=(0, 55))
    axs[1].annotate('Worst case: 47.72 millinats', xy=(boundaries()[0], 1000 * regret(boundaries()[0])),
                    xytext=(.19, 46), arrowprops={'arrowstyle': '->', 'color': '#444'})
    axs[1].annotate('p = .65: 5.29 millinats', xy=(.65, 1000 * regret(.65)), xytext=(.36, 25),
                    arrowprops={'arrowstyle': '->', 'color': '#444'})
    for ax in axs:
        ax.grid(alpha=.15)
    save(fig, '01-grid-and-regret')


def population_figure():
    populations = [
        ('Uniform: all probabilities equally common', lambda p: 1., 'Uniform(0, 1)'),
        ('Middle-heavy: mostly uncertain examples', lambda p: beta.pdf(p, 5, 5), 'Beta(5, 5)'),
        ('Endpoint-heavy: many nearly certain examples', lambda p: beta.pdf(p, .5, .5), 'Beta(1/2, 1/2)'),
        ('Two groups: mostly hard or mostly easy',
         lambda p: .5 * beta.pdf(p, 5, 30) + .5 * beta.pdf(p, 30, 5),
         '0.5 Beta(5, 30) + 0.5 Beta(30, 5)'),
    ]
    cuts = [0, EPS, *boundaries(), 1 - EPS, 1]
    p = np.linspace(.001, .999, 3000)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    results = []
    for ax, (title, density, label) in zip(axs.flat, populations):
        value = sum(quad(lambda x: density(x) * regret(x), a, b,
                         epsabs=1e-11, epsrel=1e-9)[0] for a, b in zip(cuts, cuts[1:]))
        mass = quad(density, 0, 1, epsabs=1e-9)[0]
        assert abs(mass - 1) < 1e-7
        values = [density(x) for x in p]
        ax.plot(p, values, color=BLUE)
        ax.fill_between(p, values, color=BLUE, alpha=.13)
        ax.set(xlim=(0, 1), ylim=(0, 11 if 'Endpoint' in title else 4), title=title,
               xlabel='Underlying correctness probability p', ylabel='Probability density')
        ax.text(.5, .91, f'Mean grid penalty: {1000 * value:.3f} millinats / answer',
                transform=ax.transAxes, ha='center', va='top',
                bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
        ax.text(.5, .79, label, transform=ax.transAxes, ha='center', fontsize=9)
        ax.grid(alpha=.15)
        results.append({'distribution': label, 'mean_oracle_grid_regret_nats': value})
    save(fig, '02-example-populations')
    return results


def mixture_figure():
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, p in zip(axs.flat, [.61, .64, .69]):
        weight = (p - .6) / .1
        ax.bar([.6, .7], [1 - weight, weight], width=.025, color=[BLUE, GREEN])
        ax.axvline(p, color=ORANGE, lw=2, ls='--', label=f'Fractional mean = {p:.2f}')
        ax.set(xlim=(.55, .75), ylim=(0, 1.05), xticks=[.6, .7],
               xlabel='Confidence level', ylabel='Probability of emitting this level',
               title=f'Underlying p = {p:.2f}')
        for x, height in zip([.6, .7], [1 - weight, weight]):
            ax.text(x, height + .03, f'{height:.0%}', ha='center')
        ax.legend(loc='upper left' if p > .65 else 'upper right', fontsize=9)
    ax = axs[1, 1]
    p = .64
    weight = np.linspace(0, 1, 501)
    optimum = ce(p, p)
    discrete = 1000 * ((1 - weight) * ce(p, .6) + weight * ce(p, .7) - optimum)
    fractional = 1000 * (ce(p, .6 + .1 * weight) - optimum)
    ax.plot(weight, discrete, color=BLUE, label='Score a sampled label')
    ax.plot(weight, fractional, color=ORANGE, label='Score the fractional mean')
    ax.scatter([0], [discrete[0]], color=BLUE, zorder=3)
    ax.scatter([.4], [0], color=ORANGE, zorder=3)
    ax.axvline(.4, color=ORANGE, ls=':', alpha=.6)
    ax.set(xlabel='Probability mass on level 0.7 (rest on 0.6)',
           ylabel='Extra NLL over true p (millinats / answer)',
           title='At p = .64, the training objectives disagree', xlim=(0, 1), ylim=(-1, 12))
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    save(fig, '03-mixtures-and-objectives')


def empirical_figure():
    results = json.loads((ROOT / 'docs/quantization-results.json').read_text())
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    labels = []
    i = 0
    for path, entry in results['evaluations'].items():
        arm = 'Fractional training' if '/fractional-s1-eval/' in path else 'Discrete-exact training'
        for method, label, color in [('argmax', 'argmax', BLUE),
                                     ('grid_projection', 'grid projection', ORANGE)]:
            result = entry[method]
            value = 1000 * result['nll_minus_fractional']
            lo, hi = (1000 * x for x in result['paired_bootstrap_95_percent'])
            ax.errorbar(value, i, xerr=np.array([[value - lo], [hi - value]]),
                        fmt='o', color=color, capsize=5, markersize=7)
            labels.append(f'{arm}: {label}')
            i += 1
    ax.axvline(0, color='#555', ls='--', lw=1)
    ax.set(yticks=range(i), yticklabels=labels,
           xlabel='Discrete readout NLL − fractional readout NLL (millinats / answer)',
           title='Saved checkpoints: how much does the readout change NLL?')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    fig.supxlabel('Positive favors fractional; paired 95% row-bootstrap intervals, 474 answers.\n'
                  'Intervals do not include training-seed variation.', fontsize=10)
    save(fig, '04-observed-readout-gaps')


def build_html(source=None):
    source = source or ROOT / 'docs/confidence-quantization.md'
    body = markdown.markdown(source.read_text(), extensions=['tables', 'fenced_code'])

    def embed(match):
        path = source.parent / match.group(1)
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'

    body = re.sub(r'src="(figures/[^\"]+\.png)"', embed, body)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confidence quantization: information, loss, and fractional readouts</title>
<style>body{max-width:1000px;margin:48px auto;padding:0 24px;color:#172333;
font:17px/1.65 system-ui,sans-serif}h1,h2,h3{line-height:1.2}h2{margin-top:2.2em}
img{width:100%;height:auto}a{color:#2563a6}pre{background:#f3f5f8;padding:18px;
overflow:auto;font-size:14px}table{border-collapse:collapse;width:100%;font-size:15px}
th,td{padding:9px;text-align:left;border-bottom:1px solid #dbe1e8}
blockquote{border-left:4px solid #168577;margin-left:0;padding-left:20px}
@media print{body{font-size:11pt;margin:0;max-width:none}img,table{break-inside:avoid}}
</style><body>''' + body + '</body></html>'
    title = escape(source.read_text().splitlines()[0].lstrip('# '))
    html = re.sub(r'<title>.*?</title>', lambda _: f'<title>{title}</title>', html)
    (source.with_suffix('.html')).write_text(html)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    quantizer_figure()
    populations = population_figure()
    mixture_figure()
    empirical_figure()
    (FIGURES / 'population-results.json').write_text(json.dumps(populations, indent=2) + '\n')
    build_html()
    print(json.dumps(populations, indent=2))
    print('Wrote four PNG/SVG figures and docs/confidence-quantization.html')


if __name__ == '__main__':
    main()
