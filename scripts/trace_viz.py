"""Build a page showing how individual dev questions' confidence distributions move during training.

    python scripts/trace_viz.py runs/fast/<label>/trace.jsonl OUT.html

Reads the trace.jsonl written by `fast_loop.py train --trace-every N`: a header line describing each
dev row, then one line per traced step with each row's 11 level log-probs. The page embeds every
row (probabilities quantized to 1/1000) and opens on a few picked questions: confident and
unsure base-model answers, right and wrong, plus the ones that moved most and least directly.
"""
import json
import math
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("trace_viz.html")


def expected(q):
    """Expected confidence (0-10) of level probabilities q, renormalized over the 11 levels."""
    total = sum(q)
    return sum(k * p for k, p in enumerate(q)) / total if total else 0.0


def pick(rows, curves):
    """Indices of a handful of instructive questions, each with the reason it was picked."""
    start = [c[0] for c in curves]
    end = [c[-1] for c in curves]
    path = [sum(abs(b - a) for a, b in zip(c, c[1:])) for c in curves]
    right = [i for i, r in enumerate(rows) if r["em"]]
    wrong = [i for i, r in enumerate(rows) if not r["em"]]
    picks, seen = [], set()

    def add(candidates, key, reason):
        for i in sorted(candidates, key=key):
            if i not in seen:
                seen.add(i)
                picks.append(dict(index=i, reason=reason))
                return

    add(wrong, lambda i: -start[i], "Wrong; the base model was most sure")
    add(right, lambda i: start[i], "Right; the base model was least sure")
    add(right, lambda i: -start[i], "Right; the base model was most sure")
    add(wrong, lambda i: start[i], "Wrong; the base model was least sure")
    add(wrong, lambda i: end[i] - start[i], "Wrong; confidence fell the most")
    add(right, lambda i: start[i] - end[i], "Right; confidence rose the most")
    add(range(len(rows)), lambda i: -(path[i] - abs(end[i] - start[i])), "Took the least direct path")
    add(wrong, lambda i: -end[i], "Wrong; still the most confident at the end")
    return picks


def main(trace_path, out_path):
    lines = Path(trace_path).read_text().splitlines()
    rows = json.loads(lines[0])["rows"]
    snaps = [json.loads(l) for l in lines[1:]]
    steps = [s["step"] for s in snaps]
    # probs[i][t] = 11 per-mille probabilities of row i at traced step t
    probs = [[[round(1000 * math.exp(x)) for x in s["levels"][i]] for s in snaps] for i in range(len(rows))]
    curves = [[expected(q) for q in p] for p in probs]
    data = dict(
        source=str(trace_path), steps=steps,
        rows=[dict(qid=r["qid"], text=r["text"].split("\nassistant\n")[0].strip(), answer=r["answer"], em=r["em"], f1=r["f1"]) for r in rows],
        probs=probs, picks=pick(rows, curves))
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    Path(out_path).write_text(html)
    print(json.dumps(dict(out=out_path, rows=len(rows), steps=len(steps), kb=round(len(html) / 1024),
                          picks=[(p["reason"], rows[p["index"]]["answer"]) for p in data["picks"]])))


if __name__ == "__main__":
    main(*sys.argv[1:])
