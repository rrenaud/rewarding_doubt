"""Answer KL against the change in regenerated dev accuracy, for every evaluation that measured both.

    python scripts/plot_answer_kl.py        # writes docs/answer_kl_vs_accuracy.png and prints a binned table

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
    centers, medians = [], []
    print(f"{'answer KL (nats/answer)':>24} {'n':>4} {'median Δacc':>12} {'mean Δacc':>10} {'10th pct Δacc':>14} {'mean malformed':>15}")
    for lo, hi in zip(edges, edges[1:]):
        b = sorted(p["dacc"] for p in everything if lo <= p["kl"] < hi)
        if len(b) >= 3:
            mal = st.fmean(p["malformed"] for p in everything if lo <= p["kl"] < hi)
            centers.append((lo * hi) ** 0.5)
            medians.append(st.median(b))
            print(f"{f'[{lo}, {hi})':>24} {len(b):>4} {st.median(b):>+12.3f} {st.fmean(b):>+10.3f} {b[len(b) // 10]:>+14.3f} {mal:>15.3f}")
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


if __name__ == "__main__":
    main()
