"""Per-example losses of fitted models against the base model, from minimal_trainer.py --dump-examples files.

    python scripts/example_deltas.py RUN_DIR [RUN_DIR ...] [--split train|dev] [--top 15] [--vs RUN_DIR] [--merge-seeds]

Each RUN_DIR holds examples.jsonl (one line per cached example: label, confidence distributions and objective
terms under the fitted model and under the base model, delta_loss = fitted - base loss as trained).
Prints, per run: the mean delta by label, and the examples whose loss fell or rose most. With several runs: the
per-example delta of each run against the first (or --vs), i.e. which examples one experiment fits better than
another, and how correlated the runs' deltas are. Seeds of an arm (names ending -s<N>) are averaged per example
when --merge-seeds is given.
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path


def load(run_dir, split):
    rows = [json.loads(l) for l in open(Path(run_dir) / "examples.jsonl")]
    return {(r["split"], r["qid"]): r for r in rows if r["split"] == split}


def arm(run_dir):
    name = Path(run_dir).name
    return name.rsplit("-s", 1)[0] if name.rsplit("-s", 1)[-1].split("-")[0].isdigit() else name


def merge(runs):
    """{arm: {key: row with delta_loss/delta_brier/fitted conf averaged over seeds}}"""
    out = {}
    for a, members in runs.items():
        keys = set.intersection(*(set(m) for m in members))
        merged = {}
        for k in keys:
            r = dict(members[0][k])
            r["delta_loss"] = st.fmean(m[k]["delta_loss"] for m in members)
            r["delta_brier"] = st.fmean(m[k]["delta_brier"] for m in members)
            r["fitted"] = dict(r["fitted"], expected_conf=st.fmean(m[k]["fitted"]["expected_conf"] for m in members),
                               answer_kl=st.fmean(m[k]["fitted"]["answer_kl"] for m in members))
            merged[k] = r
        out[f"{a} ({len(members)} seeds)"] = merged
    return out


def line(r, extra=""):
    return (f"{r['delta_loss']:+7.3f}  label {r['label']}  conf {r['base']['expected_conf']:.2f}->{r['fitted']['expected_conf']:.2f}"
            f"  aKL {r['fitted']['answer_kl']:.2f}{extra}  | {r['question'][:70]!r} -> {str(r['answer'])[:30]!r}")


def summarize(name, rows, top):
    print(f"\n=== {name}: {len(rows)} examples")
    by_label = collections.defaultdict(list)
    for r in rows.values():
        by_label[r["label"]].append(r)
    for label, rs in sorted(by_label.items()):
        print(f"  label {label}: n {len(rs):5d}  mean delta loss {st.fmean(r['delta_loss'] for r in rs):+.3f}  "
              f"delta brier {st.fmean(r['delta_brier'] for r in rs):+.4f}  conf {st.fmean(r['base']['expected_conf'] for r in rs):.2f}"
              f"->{st.fmean(r['fitted']['expected_conf'] for r in rs):.2f}")
    ordered = sorted(rows.values(), key=lambda r: r["delta_loss"])
    print(f"  most improved (fitted loss - base loss):")
    for r in ordered[:top]:
        print("   ", line(r))
    print(f"  most worsened:")
    for r in ordered[-top:][::-1]:
        print("   ", line(r))


def compare(ref_name, ref, name, rows, top):
    keys = set(ref) & set(rows)
    d = {k: rows[k]["delta_loss"] - ref[k]["delta_loss"] for k in keys}
    xs, ys = [ref[k]["delta_loss"] for k in keys], [rows[k]["delta_loss"] for k in keys]
    corr = st.correlation(xs, ys) if len(keys) > 2 else float("nan")
    print(f"\n=== {name} vs {ref_name}: {len(keys)} shared examples, correlation of deltas {corr:.3f}, "
          f"mean loss difference {st.fmean(d.values()):+.4f} (negative: {name} fits better)")
    for title, ks in (("fits better than the reference", sorted(keys, key=d.get)[:top]),
                      ("fits worse than the reference", sorted(keys, key=d.get)[-top:][::-1])):
        print(f"  {title}:")
        for k in ks:
            r = rows[k]
            print(f"    {d[k]:+7.3f}  ({ref[k]['delta_loss']:+.3f} -> {r['delta_loss']:+.3f})  label {r['label']}  conf base "
                  f"{r['base']['expected_conf']:.2f}, {ref[k]['fitted']['expected_conf']:.2f} vs {r['fitted']['expected_conf']:.2f}"
                  f"  | {r['question'][:60]!r} -> {str(r['answer'])[:25]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--vs", default="", help="the reference run for comparisons (default: the first)")
    parser.add_argument("--merge-seeds", action="store_true")
    args = parser.parse_args()
    loaded = {Path(r).name: load(r, args.split) for r in args.runs}
    if args.merge_seeds:
        groups = collections.defaultdict(list)
        for r in args.runs:
            groups[arm(r)].append(loaded[Path(r).name])
        loaded = merge(groups)
    for name, rows in loaded.items():
        summarize(name, rows, args.top)
    names = list(loaded)
    ref_name = next((n for n in names if args.vs and Path(args.vs).name in n), names[0])
    for name in names:
        if name != ref_name:
            compare(ref_name, loaded[ref_name], name, loaded[name], args.top)


if __name__ == "__main__":
    main()
