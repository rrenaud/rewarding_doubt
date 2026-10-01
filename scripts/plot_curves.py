"""Render runs/modal-curves-*/ into a self-contained interactive curves.html.

    python scripts/plot_curves.py runs/modal-curves-<stamp> [--base runs/modal-stage2-.../result.json]

Training time per step excludes model loading, snapshot saving and Train.py's end-of-epoch
evaluation. Exact runs log `seconds` per step (sampling + updates). For PPO, a step is the
interval between consecutive PPOTrainer.step exits minus snapshot-save time; the first step
and the epoch-boundary step (which includes Train.py's validation pass) use the median step.
"""
import argparse
import json
import statistics
from pathlib import Path

SERIES = {  # label: (display name, objective color slot, dashed?, optimizer updates per step)
    "ppo": ("Released PPO (8 updates/batch)", 1, False, 8),
    "discrete-exact-8x": ("Discrete-exact, 8 updates/batch", 2, False, 8),
    "fractional-8x": ("Fractional, 8 updates/batch", 3, False, 8),
    "discrete-exact-1x": ("Discrete-exact, 1 update/batch", 2, True, 1),
    "fractional-1x": ("Fractional, 1 update/batch", 3, True, 1),
}


def step_minutes(label_dir: Path, label: str) -> dict[int, float]:
    """Cumulative training minutes at the end of each step."""
    rows = [json.loads(line) for line in (label_dir / "timing.jsonl").read_text().splitlines() if line.strip()]
    if label == "ppo":
        durations = {}
        for prev, row in zip(rows, rows[1:]):
            durations[row["step"]] = row["exit"] - prev["exit"] - prev.get("save_seconds", 0.)
        median = statistics.median(durations.values())
        per_epoch = len(rows) // 2
        durations[1] = median
        durations[per_epoch + 1] = median  # gap includes Train.py's end-of-epoch evaluation and save
    else:
        durations = {row["step"]: row["seconds"] for row in rows}
    total, out = 0., {}
    for step in sorted(durations):
        total += durations[step]
        out[step] = total / 60
    return out


def collect(run: Path, base: dict) -> list[dict]:
    curve = json.loads((run / "curve.json").read_text())
    series = []
    for label, (name, slot, dashed, updates) in SERIES.items():
        if label not in curve:
            continue
        minutes = step_minutes(run / label, label)
        points = [dict(step=0, updates=0, minutes=0., **{k: base[k] for k in ["ece", "auroc", "brier", "accuracy"]})]
        for step, m in sorted(curve[label].items(), key=lambda kv: int(kv[0])):
            step = int(step)
            points.append(dict(step=step, updates=step * updates, minutes=round(minutes[step], 2),
                               **{k: m[k] for k in ["ece", "auroc", "brier", "accuracy"]}))
        series.append(dict(label=label, name=name, slot=slot, dashed=dashed, points=points))
    return series


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration Learning Curves</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; }
p, li { color: var(--text-secondary); max-width: 75ch; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 12px; align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 14px; border: 0; background: transparent; color: var(--text-secondary);
              padding: 6px 12px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1); }
.seg-label { font-size: 13px; color: var(--muted); margin-right: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 4px 0 8px; font-size: 13px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.chart-wrap { position: relative; }
#chart { display: block; width: 100%; height: auto; overflow: visible; }
.legend svg, .tooltip svg { flex: none; display: block; }
.tick { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.axis-title { fill: var(--text-secondary); font-size: 13px; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
           min-width: 220px; display: none; }
.tooltip .row { display: flex; align-items: center; gap: 8px; }
.tooltip strong { font-variant-numeric: tabular-nums; min-width: 44px; }
.tooltip .sub { color: var(--muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
th[colspan] { text-align: center; }
.scroll { overflow-x: auto; }
details summary { cursor: pointer; color: var(--text-secondary); margin: 8px 0; }
.note { font-size: 13px; color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>Calibration learning curves on the released Llama setup</h1>
<p>Llama-3-8B-Instruct (4-bit, LoRA r=8), 1,024 TriviaQA training questions, batch 8, 2 epochs, lr 1e-5.
Every snapshot is scored with the released evaluation on the same 512 held-out questions; step 0 is the base model.
Only the update rule differs between lines.</p>

<div class="controls" role="group" aria-label="Chart controls">
  <div><span class="seg-label">Metric</span>
    <span class="seg" id="metric">
      <button data-v="ece" aria-pressed="true">ECE ↓</button><button data-v="auroc" aria-pressed="false">AUROC ↑</button><button data-v="brier" aria-pressed="false">Brier ↓</button>
    </span></div>
  <div><span class="seg-label">X axis</span>
    <span class="seg" id="xaxis">
      <button data-v="step" aria-pressed="true">Training steps</button><button data-v="updates" aria-pressed="false">Optimizer updates</button><button data-v="minutes" aria-pressed="false">Training minutes</button>
    </span></div>
</div>

<div class="card">
  <div class="legend" id="legend"></div>
  <div class="chart-wrap" id="wrap">
    <svg id="chart" viewBox="0 0 960 420" role="img" aria-labelledby="chart-desc"><desc id="chart-desc"></desc></svg>
    <div class="tooltip" id="tip" role="status"></div>
  </div>
  <p class="note" id="xnote"></p>
</div>

<h2>Matched-budget comparison</h2>
<p>Each cell is the last snapshot at or before the budget. Snapshots are every 32 steps, so a cell can lag the budget by
up to 32 steps. Selecting snapshots this way uses no held-out information.</p>
<div class="card scroll"><table id="budget"></table></div>

__EXPLAIN__

<details><summary>All snapshot values (table view)</summary>
<div class="card scroll"><table id="all"></table></div>
</details>
</main>
<script>
const SERIES = __DATA__;
const COLORS = {1: "var(--series-1)", 2: "var(--series-2)", 3: "var(--series-3)"};
const XLABEL = {step: "Training steps (batches of 8 questions)", updates: "Optimizer updates", minutes: "Training minutes (excl. loading, saving, evaluation)"};
const MLABEL = {ece: "ECE (lower is better)", auroc: "AUROC (higher is better)", brier: "Brier (lower is better)"};
let metric = "ece", xkey = "step";
const svg = document.getElementById("chart"), tip = document.getElementById("tip"), wrap = document.getElementById("wrap");
const W = 960, H = 420, M = {l: 64, r: 24, t: 16, b: 56};
const NS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const start = Math.floor(lo / step) * step, out = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function fmt(v, key) { return key === "minutes" ? v.toFixed(0) : key === "step" || key === "updates" ? String(Math.round(v)) : v.toFixed(3); }

function legend() {
  const box = document.getElementById("legend"); box.textContent = "";
  for (const s of SERIES) {
    const item = document.createElement("span");
    const sw = document.createElementNS(NS, "svg"); sw.setAttribute("width", "24"); sw.setAttribute("height", "10");
    el("line", {x1: 0, y1: 5, x2: 24, y2: 5, stroke: COLORS[s.slot], "stroke-width": 2, "stroke-dasharray": s.dashed ? "5 4" : "none"}, sw);
    item.appendChild(sw); item.appendChild(document.createTextNode(s.name)); box.appendChild(item);
  }
}

function draw() {
  svg.querySelectorAll("g").forEach(g => g.remove());
  const all = SERIES.flatMap(s => s.points);
  const xmax = Math.max(...all.map(p => p[xkey]));
  const ys = all.map(p => p[metric]);
  let ylo = Math.min(...ys), yhi = Math.max(...ys);
  const pad = (yhi - ylo) * 0.08; ylo = Math.max(0, ylo - pad); yhi = yhi + pad;
  const yt = niceTicks(ylo, yhi, 5); ylo = Math.min(ylo, yt[0]); yhi = Math.max(yhi, yt[yt.length - 1]);
  const xt = niceTicks(0, xmax, 8).filter(t => t <= xmax + 1e-9);
  const X = v => M.l + (v / xmax) * (W - M.l - M.r);
  const Y = v => H - M.b - (v - ylo) / (yhi - ylo) * (H - M.t - M.b);
  const g = el("g", {}, svg);
  for (const t of yt) {
    el("line", {x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)", "stroke-width": 1}, g);
    el("text", {x: M.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick"}, g).textContent = t.toFixed(2);
  }
  el("line", {x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1}, g);
  for (const t of xt) el("text", {x: X(t), y: H - M.b + 18, "text-anchor": "middle", class: "tick"}, g).textContent = fmt(t, xkey);
  el("text", {x: (M.l + W - M.r) / 2, y: H - 10, "text-anchor": "middle", class: "axis-title"}, g).textContent = XLABEL[xkey];
  el("text", {x: 16, y: (M.t + H - M.b) / 2, transform: `rotate(-90 16 ${(M.t + H - M.b) / 2})`, "text-anchor": "middle", class: "axis-title"}, g).textContent = MLABEL[metric];
  for (const s of SERIES) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[xkey]).toFixed(1)},${Y(p[metric]).toFixed(1)}`).join("");
    el("path", {d, fill: "none", stroke: COLORS[s.slot], "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dashed ? "6 5" : "none"}, g);
    for (const p of s.points.slice(1))
      el("circle", {cx: X(p[xkey]), cy: Y(p[metric]), r: 4, fill: s.dashed ? "var(--surface-1)" : COLORS[s.slot], stroke: COLORS[s.slot], "stroke-width": 2}, g);
  }
  const base = SERIES[0].points[0];
  el("circle", {cx: X(0), cy: Y(base[metric]), r: 5, fill: "var(--text-secondary)", stroke: "var(--surface-1)", "stroke-width": 2}, g);
  el("text", {x: X(0) + 10, y: Y(base[metric]) + (metric === "auroc" ? 16 : -10), class: "tick"}, g).textContent = "base model";
  const cross = el("line", {y1: M.t, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden"}, g);
  const hit = el("rect", {x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent", tabindex: 0}, g);
  function show(px) {
    const xv = Math.max(0, Math.min(xmax, (px - M.l) / (W - M.l - M.r) * xmax));
    cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const head = document.createElement("div"); head.className = "sub";
    head.textContent = `Nearest snapshot to ${fmt(xv, xkey)} ${xkey === "minutes" ? "min" : xkey === "step" ? "steps" : "updates"}`;
    tip.appendChild(head);
    for (const s of SERIES) {
      const p = s.points.reduce((a, b) => Math.abs(b[xkey] - xv) < Math.abs(a[xkey] - xv) ? b : a);
      const row = document.createElement("div"); row.className = "row";
      const sw = document.createElementNS(NS, "svg"); sw.setAttribute("width", "16"); sw.setAttribute("height", "8");
      el("line", {x1: 0, y1: 4, x2: 16, y2: 4, stroke: COLORS[s.slot], "stroke-width": 2, "stroke-dasharray": s.dashed ? "4 3" : "none"}, sw);
      const v = document.createElement("strong"); v.textContent = p[metric].toFixed(3);
      const n = document.createElement("span"); n.textContent = `${s.name}`;
      const at = document.createElement("span"); at.className = "sub"; at.textContent = ` @ ${fmt(p[xkey], xkey)}`;
      row.append(sw, v, n, at); tip.appendChild(row);
    }
    const rect = svg.getBoundingClientRect(), scale = rect.width / W;
    tip.style.display = "block";
    const left = px * scale + 16, maxLeft = wrap.clientWidth - tip.offsetWidth - 4;
    tip.style.left = Math.max(0, Math.min(left, maxLeft)) + "px"; tip.style.top = (M.t * scale + 8) + "px";
  }
  hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(); show((e.clientX - r.left) * W / r.width); });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  hit.addEventListener("focus", () => show(W - M.r));
  hit.addEventListener("blur", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  document.getElementById("chart-desc").textContent = `${MLABEL[metric]} against ${XLABEL[xkey]} for ${SERIES.length} training configurations.`;
  document.getElementById("xnote").textContent = xkey === "minutes"
    ? "Wall-clock training time on one L40S: answer sampling plus updates. Model loading, snapshot saving and end-of-epoch evaluation are excluded."
    : xkey === "updates" ? "Released PPO and the 8-update runs make 8 Adam steps per batch (4 passes × 2 minibatches); the 1-update runs make one."
    : "Each step samples answers for 8 questions; all configurations see the same questions in the same order.";
}

function tables() {
  const budgets = [["Steps", "step", [128, 256]], ["Optimizer updates", "updates", [256, 1024, 2048]], ["Minutes", "minutes", [30, 60, 90]]];
  const t = document.getElementById("budget"); t.textContent = "";
  const h1 = t.insertRow(), h2 = t.insertRow();
  const c0 = document.createElement("th"); c0.rowSpan = 2; c0.textContent = "Configuration"; h1.appendChild(c0);
  for (const [name, , vals] of budgets) {
    const th = document.createElement("th"); th.colSpan = vals.length; th.textContent = `${name} ≤`; h1.appendChild(th);
    for (const v of vals) { const th2 = document.createElement("th"); th2.textContent = v; h2.appendChild(th2); }
  }
  for (const s of SERIES) {
    const r = t.insertRow(); r.insertCell().textContent = s.name;
    for (const [, key, vals] of budgets) for (const v of vals) {
      const ok = s.points.filter(p => p[key] <= v + 1e-9), p = ok[ok.length - 1];
      r.insertCell().textContent = p && p.step > 0 ? `${p.ece.toFixed(3)} / ${p.auroc.toFixed(3)}` : "—";
    }
  }
  const cap = t.createCaption(); cap.className = "note"; cap.style.captionSide = "bottom"; cap.style.textAlign = "left";
  cap.textContent = "Cells: ECE / AUROC. — means no snapshot fits within the budget.";
  const a = document.getElementById("all"); a.textContent = "";
  const hr = a.insertRow();
  for (const c of ["Configuration", "Step", "Updates", "Minutes", "ECE", "AUROC", "Brier", "Accuracy"]) { const th = document.createElement("th"); th.textContent = c; hr.appendChild(th); }
  for (const s of SERIES) for (const p of s.points) {
    const r = a.insertRow();
    for (const v of [s.name, p.step, p.updates, p.minutes.toFixed(1), p.ece.toFixed(3), p.auroc.toFixed(3), p.brier.toFixed(3), (100 * p.accuracy).toFixed(1) + "%"]) r.insertCell().textContent = v;
  }
}
for (const [id, set] of [["metric", v => metric = v], ["xaxis", v => xkey = v]])
  for (const b of document.querySelectorAll(`#${id} button`))
    b.addEventListener("click", () => { document.querySelectorAll(`#${id} button`).forEach(x => x.setAttribute("aria-pressed", x === b)); set(b.dataset.v); draw(); });
legend(); draw(); tables();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", type=Path, default=Path("runs/modal-stage2-20261001T082216Z/result.json"))
    parser.add_argument("--explain", type=Path, help="HTML fragment with the trade-off discussion")
    args = parser.parse_args()
    base = json.loads(args.base.read_text())["metrics"]["base"]
    series = collect(args.run, base)
    explain = args.explain.read_text() if args.explain else ""
    page = PAGE.replace("__DATA__", json.dumps(series).replace("</", "<\\/")).replace("__EXPLAIN__", explain)
    (args.run / "curves.html").write_text(page)
    (args.run / "curves_data.json").write_text(json.dumps(series, indent=1) + "\n")
    print(args.run / "curves.html")


if __name__ == "__main__":
    main()
