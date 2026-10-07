"""Answer KL against the change in regenerated dev accuracy, for every evaluation that measured both.

    python scripts/plot_answer_kl.py        # docs/answer_kl_vs_accuracy{.png,_zoom.png,.html}; prints a binned table
    python scripts/plot_answer_kl.py llama  # Llama-3-8B runs (runs/minimal/llama-*) -> docs/answer_kl_vs_accuracy_llama.png

answer_kl: KL(policy || base model) over the vocabulary, summed over each cached dev answer's tokens (nats per
answer, about 7-8 tokens). Δ accuracy: regenerated dev accuracy (the policy answers the 507 dev questions again,
seeded draws, F1 > 0.5) minus the base model's on the same hardware and trainer: 0.396 for the fast loop (4-bit,
Modal), 0.426 for the minimal trainer on Modal, 0.440 on RunPod. Malformed answers count as wrong.
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [  # label, globs, keep(run name), base accuracy, color, marker size
    ("fast loop, offline (300 steps)", ["runs/fast/akl-*"], lambda r: True, 0.396, "#9aa5b1", 14),
    ("minimal, offline (300 steps)", ["runs/minimal/akl-*", "runs/minimal/hs-*", "runs/minimal/stock-*"],
     lambda r: "online" not in r and "smoke" not in r, 0.426, "#4c78a8", 14),
    ("minimal, online (300 steps)", ["runs/minimal/on-*", "runs/minimal/stock-online-*"], lambda r: True, 0.426, "#f58518", 16),
    ("minimal, recovery from trained v_proj", ["runs/minimal/recover-*"], lambda r: True, 0.426, "#54a24b", 16),
    ("RunPod, online (3,000 steps)", ["runs/runpod/online-*"], lambda r: True, 0.440, "#e45756", 40),
]


def points(patterns, keep, base):
    out = []
    files = sorted({f for pattern in patterns for f in glob.glob(str(ROOT / pattern / "train.log"))})
    for f in files:
        if not keep(Path(f).parent.name):
            continue
        for line in open(f):
            if line.startswith('{"step"'):
                r = json.loads(line)
                if r["step"] > 0 and "regen_accuracy" in r and "answer_kl" in r:
                    out.append(dict(run=Path(f).parent.name, step=r["step"], kl=r["answer_kl"], dacc=r["regen_accuracy"] - base,
                                    malformed=r.get("regen_malformed", 0.0)))
    return out


def main():
    data = {label: points(patterns, keep, base) for label, patterns, keep, base, *_ in GROUPS}
    everything = [p for ps in data.values() for p in ps]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for (label, _, _, _, color, size) in GROUPS:
        ps = data[label]
        if not ps:
            continue
        for a, key in ((ax, "dacc"), (bx, "malformed")):
            a.scatter([p["kl"] for p in ps], [p[key] for p in ps], s=size, c=color, alpha=0.75, edgecolors="none",
                      label=f"{label} ({len(ps)})")
    long_runs = sorted({p["run"] for p in data["RunPod, online (3,000 steps)"]})
    for run in long_runs:  # trajectories of the long runs
        ps = sorted((p for p in data["RunPod, online (3,000 steps)"] if p["run"] == run), key=lambda p: p["step"])
        ax.plot([p["kl"] for p in ps], [p["dacc"] for p in ps], c="#e45756", lw=0.8, alpha=0.5)
        ax.annotate(run.replace("online-", ""), (ps[-1]["kl"], ps[-1]["dacc"]), fontsize=7, color="#a33", xytext=(3, 2),
                    textcoords="offset points")
    edges = [0.1, 0.5, 1, 1.5, 2, 3, 4, 6, 10, 40]
    centers, medians, table = [], [], []
    print(f"{'answer KL (nats/answer)':>24} {'n':>4} {'median Δacc':>12} {'mean Δacc':>10} {'10th pct Δacc':>14} {'mean malformed':>15}")
    for lo, hi in zip(edges, edges[1:]):
        b = sorted(p["dacc"] for p in everything if lo <= p["kl"] < hi)
        if len(b) >= 3:
            mal = st.fmean(p["malformed"] for p in everything if lo <= p["kl"] < hi)
            centers.append((lo * hi) ** 0.5)
            medians.append(st.median(b))
            print(f"{f'[{lo}, {hi})':>24} {len(b):>4} {st.median(b):>+12.3f} {st.fmean(b):>+10.3f} {b[len(b) // 10]:>+14.3f} {mal:>15.3f}")
            table.append((f"{lo}–{hi}", len(b), st.median(b), b[len(b) // 10], mal))
    ax.plot(centers, medians, c="black", lw=2, label="binned median")
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel("answer KL to the base model (nats per answer, ~7-8 tokens)")
        a.grid(alpha=0.3)
    ax.axhline(0, c="black", lw=0.8)
    ax.axhspan(-0.02, 0.02, color="#cccccc", alpha=0.35, lw=0, label="±2 points")
    ax.set_ylabel("Δ regenerated dev accuracy vs base model")
    bx.set_ylabel("malformed answers (fraction)")
    ax.set_title("Answer accuracy against answer KL")
    bx.set_title("Malformed answers against answer KL")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = ROOT / "docs/answer_kl_vs_accuracy.png"
    fig.savefig(out, dpi=130)
    print(f"{len(everything)} points -> {out.relative_to(ROOT)}")
    zoom(data, everything)
    write_html(table, len(everything))


def zoom(data, everything, lo=0.8, hi=3.0):
    """The knee: answer KL from lo to hi nats, linear axis. Points sized by training step (later = larger);
    the 3,000-step runs labeled with their steps; a binned median in 0.25-nat bins."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for (label, _, _, _, color, _) in GROUPS:
        ps = [p for p in data[label] if lo <= p["kl"] <= hi]
        if not ps:
            continue
        long = label.startswith("RunPod")
        ax.scatter([p["kl"] for p in ps], [p["dacc"] for p in ps], c=color, edgecolors="black" if long else "none",
                   linewidths=0.8 if long else 0, alpha=0.95 if long else 0.55, zorder=3 if long else 2,
                   s=[(140 if long else 8 + 0.12 * p["step"]) for p in ps], label=f"{label} ({len(ps)} here)")
        if long:
            for i, p in enumerate(sorted(ps, key=lambda p: p["kl"])):  # alternate above / below to limit overlaps
                ax.annotate(f"{p['run'].replace('online-', '')} @{p['step']}", (p["kl"], p["dacc"]), fontsize=7.5,
                            color="#8b1e1e", xytext=(6, 6 if i % 2 == 0 else -13), textcoords="offset points", zorder=4)
    edges = [lo + 0.25 * i for i in range(int((hi - lo) / 0.25) + 2)]
    centers, medians = [], []
    for a, b in zip(edges, edges[1:]):
        bin_ = [p["dacc"] for p in everything if a <= p["kl"] < b]
        if len(bin_) >= 3:
            centers.append((a + b) / 2)
            medians.append(st.median(bin_))
    ax.plot(centers, medians, c="black", lw=2, zorder=5, label="median, 0.25-nat bins")
    ax.axhline(0, c="black", lw=0.8)
    ax.axhspan(-0.02, 0.02, color="#cccccc", alpha=0.35, lw=0, label="±2 points")
    ax.axvline(2.0, c="#e45756", lw=1, ls="--", label="2 nats")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("answer KL to the base model (nats per answer, ~7-8 tokens)")
    ax.set_ylabel("Δ regenerated dev accuracy vs base model")
    ax.set_title(f"The knee, {lo}-{hi} nats (short runs: marker size grows with training step)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = ROOT / "docs/answer_kl_vs_accuracy_zoom.png"
    fig.savefig(out, dpi=130)
    print(f"zoom: {sum(lo <= p['kl'] <= hi for p in everything)} points -> {out.relative_to(ROOT)}")



def write_html(table, n):
    """docs/answer_kl_vs_accuracy.html: both plots (embedded) and the binned table, self-contained."""
    import base64
    img = lambda name: base64.b64encode((ROOT / "docs" / name).read_bytes()).decode()
    rows = "\n".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]:+.3f}</td><td>{r[3]:+.3f}</td><td>{100 * r[4]:.1f}%</td></tr>" for r in table)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answer KL vs accuracy</title>
<style>
:root {{ --fg:#1d2433; --muted:#5b6475; --bg:#ffffff; --line:#e3e6ec; --head:#f5f6f8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6e8ec; --muted:#a3abb9; --bg:#15181e; --line:#2c313b; --head:#1d2129; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 6px; }} h2 {{ font-size:17px; margin:28px 0 8px; }} p.sub {{ color:var(--muted); margin:0 0 14px; }}
img {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:6px; }}
table {{ border-collapse:collapse; margin:16px 0; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 14px; text-align:right; }}
th {{ background:var(--head); font-weight:600; }} th:first-child, td:first-child {{ text-align:left; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }} code {{ font-size:13px; }}
</style></head><body><main>
<h1>Answer KL against answer accuracy</h1>
<p class="sub">Qwen-2.5-3B, every evaluation that measured both ({n}). Answer KL: KL(policy ‖ base) over the vocabulary, summed per
cached dev answer (~7–8 tokens). Δ accuracy: regenerated dev accuracy (507 questions, seeded draws, F1 &gt; 0.5) minus the base
model's on the same hardware.</p>
<img src="data:image/png;base64,{img('answer_kl_vs_accuracy.png')}" alt="Answer KL against change in accuracy, all runs: flat below about 2 nats, falling steeply above; malformed answers rise past about 4 nats.">
<table><thead><tr><th>answer KL (nats/answer)</th><th>points</th><th>median Δ accuracy</th><th>10th pct Δ accuracy</th><th>malformed</th></tr></thead>
<tbody>
{rows}
</tbody></table>
<h2>The knee, 0.8–3 nats</h2>
<p class="sub">Short (300-step) runs: marker size grows with the training step. The 3,000-step RunPod runs are the large outlined
markers, labeled with their step; their accuracy was regenerated every 500 steps, when their answer KL was 2.07–3.0, so none falls
below 2 nats. The black line is the median in 0.25-nat bins.</p>
<img src="data:image/png;base64,{img('answer_kl_vs_accuracy_zoom.png')}" alt="Zoom on 0.8 to 3 nats: the median sits at 0 to +1 point up to 2 nats and near -1 point beyond; the long runs sit at 2.1 to 2.9 nats, mostly 1 to 2 points down, the v_proj run 3 to 3.6 points down at steps 1,500 and 2,000.">
<ul>
<li><b>Below ~2 nats per answer, accuracy is flat</b> (median 0 to +1 point; the worst tenth within about 2 points).</li>
<li><b>From 2 nats it falls</b>: about −1 point at 2–3, −2 at 3–4, −5 at 4–6, −13 at 6–10, −21 beyond.</li>
<li><b>The long runs sit just past the knee</b> (2.1–2.9 nats), 1–2 points down, the <code>v_proj</code> run 3–3.6 points at steps 1,500–2,000.</li>
<li><b>Malformed answers are a late signal</b>: under 1% until about 4 nats.</li>
</ul>
<p class="sub">Source: <code>scripts/plot_answer_kl.py</code> (rerun to update).</p>
</main></body></html>"""
    out = ROOT / "docs/answer_kl_vs_accuracy.html"
    out.write_text(html)
    print(f"-> {out.relative_to(ROOT)}")


def llama():
    """Llama-3-8B: every run starts from the base model, so Δ accuracy is against the run's own step 0. One color
    per adapter family; unpenalized drift runs (llama-drift-*) trace the curve, the others are penalized."""
    import collections
    families = collections.defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "runs/minimal/llama-*/train.log"))):
        name = Path(f).parent.name
        curve = [json.loads(l) for l in open(f) if l.startswith('{"step"')]
        base = next((r["regen_accuracy"] for r in curve if r["step"] == 0 and "regen_accuracy" in r), None)
        if base is None:
            continue
        family = ("o_proj bias" if "obias" in name or "hookbias" in name else "RMSNorm gains" if "norm" in name
                  else "LoRA o_proj" if "lora" in name else "other")
        for r in curve:
            if r["step"] > 0 and "regen_accuracy" in r:
                families[family].append(dict(run=name, step=r["step"], kl=r["answer_kl"], dacc=r["regen_accuracy"] - base,
                                             malformed=r.get("regen_malformed", 0.0), drift="drift" in name))
    everything = [p for ps in families.values() for p in ps]
    colors = {"o_proj bias": "#e45756", "RMSNorm gains": "#4c78a8", "LoRA o_proj": "#54a24b", "other": "#9aa5b1"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for family, ps in families.items():
        for drift in (True, False):
            sel = [p for p in ps if p["drift"] == drift]
            if sel:
                ax.scatter([max(p["kl"], 1e-3) for p in sel], [p["dacc"] for p in sel], c=colors[family], s=28 if drift else 14,
                           marker="o" if drift else "x", alpha=0.8, label=f"{family}, {'unpenalized drift' if drift else 'penalized / other'} ({len(sel)})")
    edges = [0.001, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 4, 6, 10, 40]
    centers, medians = [], []
    print(f"Llama-3-8B: {len(everything)} points")
    print(f"{'answer KL (nats/answer)':>24} {'n':>4} {'median Δacc':>12} {'10th pct Δacc':>14} {'mean malformed':>15}")
    for lo, hi in zip(edges, edges[1:]):
        b = sorted(p["dacc"] for p in everything if lo <= p["kl"] < hi)
        if len(b) >= 3:
            centers.append((lo * hi) ** 0.5)
            medians.append(st.median(b))
            mal = st.fmean(p["malformed"] for p in everything if lo <= p["kl"] < hi)
            print(f"{f'[{lo}, {hi})':>24} {len(b):>4} {st.median(b):>+12.3f} {b[len(b) // 10]:>+14.3f} {mal:>15.3f}")
    ax.plot(centers, medians, c="black", lw=2, label="binned median")
    ax.set_xscale("log")
    ax.axhline(0, c="black", lw=0.8)
    ax.axhspan(-0.02, 0.02, color="#cccccc", alpha=0.35, lw=0, label="±2 points")
    ax.set_xlabel("answer KL to the base model (nats per answer)")
    ax.set_ylabel("Δ regenerated dev accuracy vs the run's step 0")
    ax.set_title("Llama-3-8B: answer accuracy against answer KL")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = ROOT / "docs/answer_kl_vs_accuracy_llama.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    llama() if sys.argv[1:] == ["llama"] else main()
