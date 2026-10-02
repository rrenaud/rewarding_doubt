"""Render a hyperparameter search directory into an interactive hparam_search.html.

    python scripts/plot_hparam_search.py runs/hparam-search-<stamp>
"""
import json
import sys
from pathlib import Path

STAGES = [("round1", "Round 1: random"), ("stage1", "Stage 1: learning rate"), ("stage2", "Stage 2: updates per batch"),
          ("stage3", "Stage 3: learning rate again"), ("stage4", "Stage 4: regularization")]
KEYS = ["brier", "ece", "auroc", "accuracy", "wrong_format", "ece_unsampled", "auroc_unsampled"]


def load(search: Path):
    base_acc = json.loads((search / "base_dev_metrics.json").read_text())["accuracy"]
    history = [json.loads(l) for l in (search / "history.jsonl").read_text().splitlines()]
    runs = []
    for stage, _ in STAGES:
        cfg_path = search / f"{stage}_configs.json"
        curve_path = search / stage / "curve.json"
        if not cfg_path.exists() or not curve_path.exists():
            continue
        configs = json.loads(cfg_path.read_text())
        curve = json.loads(curve_path.read_text())
        selected = next((h["current"] for h in history if h["stage"] == stage), None)
        for label, cfg in configs.items():
            row = dict(stage=stage, label=label, arm=cfg["arm"], params=cfg["params"])
            if label in curve:
                m = curve[label][max(curve[label], key=int)]
                row.update({k: m.get(k if k != "wrong_format" else "wrong_format_rate") for k in KEYS})
                row["eligible"] = row["wrong_format"] <= 0.02 and row["accuracy"] >= base_acc - 0.02
                row["status"] = "ok" if row["eligible"] else "ineligible"
            else:
                row.update({k: None for k in KEYS}, eligible=False, status="failed")
            row["selected"] = bool(selected and row["eligible"] and selected[cfg["arm"]] == cfg["params"])
            runs.append(row)
    final = None
    if (search / "final" / "curve.json").exists():
        curve = json.loads((search / "final" / "curve.json").read_text())
        test = json.loads((search / "final" / "curve_test.json").read_text()) if (search / "final" / "curve_test.json").exists() else {}
        final = {label: dict(dev=points[max(points, key=int)], test=test.get(label)) for label, points in curve.items()}
    return dict(base_accuracy=base_acc, runs=runs, history=history, final=final,
                current=json.loads((search / "current.json").read_text()), stages=STAGES)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hyperparameter Search</title>
<style>
:root { color-scheme: light; --page:#f6f7f9; --surface:#ffffff; --fg:#15181d; --muted:#5a616d; --faint:#8a909a; --line:#d8dce3; --grid:#eceef2;
  --ppo:#2a78d6; --exact:#eb6834; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#111317; --surface:#1a1d22; --fg:#eceef2; --muted:#a2a9b5; --faint:#7d838d; --line:#323842; --grid:#262a31; --ppo:#3987e5; --exact:#d95926; --bad:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#111317; --surface:#1a1d22; --fg:#eceef2; --muted:#a2a9b5; --faint:#7d838d; --line:#323842; --grid:#262a31; --ppo:#3987e5; --exact:#d95926; --bad:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1120px; margin:0 auto; padding-inline:16px; padding-block:28px 64px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:18px; margin:30px 0 10px; }
p { color:var(--muted); max-width:78ch; margin:0 0 12px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin:16px 0; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-width:0; }
.card h3 { margin:0 0 8px; font-size:15px; display:flex; align-items:center; gap:8px; }
.sw { width:12px; height:12px; border-radius:50%; display:inline-block; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-size:13px; font-variant-numeric:tabular-nums; }
.kv dt { color:var(--muted); } .kv dd { margin:0; }
.pill { display:inline-block; font-size:12px; padding:1px 8px; border-radius:99px; border:1px solid var(--line); color:var(--muted); }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:8px 0 12px; }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.seg button { font:inherit; font-size:13px; border:0; background:transparent; color:var(--muted); padding:5px 11px; cursor:pointer; }
.seg button[aria-pressed="true"] { background:var(--fg); color:var(--surface); }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--muted); }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px; min-width:0; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
.panel h4 { margin:0 0 4px; font-size:14px; } .panel .sub { font-size:12px; color:var(--faint); margin-bottom:4px; }
svg.chart { width:100%; height:auto; display:block; overflow:visible; }
.tick { fill:var(--faint); font-size:11px; font-variant-numeric:tabular-nums; } .axt { fill:var(--muted); font-size:12px; }
.tip { position:fixed; pointer-events:none; background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:12px; box-shadow:0 6px 18px rgba(0,0,0,.15); display:none; max-width:320px; z-index:5; }
.tip b { font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:12.5px; font-variant-numeric:tabular-nums; min-width:820px; }
th, td { text-align:right; padding:5px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; } th:nth-child(-n+3), td:nth-child(-n+3) { text-align:left; }
th { color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--surface); cursor:pointer; }
tr.sel td { font-weight:600; } tr.bad td { color:var(--faint); }
.note { font-size:13px; color:var(--faint); }
</style></head>
<body><main>
<h1>Hyperparameter search: exact + hinge vs PPO + hinge</h1>
<p>Llama-3-8B, 1,024 TriviaQA training questions, every run scored on a 512-question dev split (disjoint from test). Selection: the lowest dev Brier among <b>eligible</b> runs, meaning format failures ≤ 2% and answer accuracy within 2 points of the base model (__BASEACC__). Round 1 sampled 16 random configurations per arm; four greedy stages then swept one setting at a time around the current best.</p>
<div class="cards" id="cards"></div>

<div class="controls">
  <span class="note">Metric</span>
  <span class="seg" id="metric"><button data-v="brier" aria-pressed="true">Brier ↓</button><button data-v="ece" aria-pressed="false">ECE ↓</button><button data-v="auroc" aria-pressed="false">AUROC ↑</button><button data-v="accuracy" aria-pressed="false">Accuracy</button></span>
  <span class="legend">
    <span><i class="sw" style="background:var(--exact)"></i>Exact + hinge</span>
    <span><i class="sw" style="background:var(--ppo)"></i>PPO + hinge</span>
    <span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--muted)" stroke-width="2"/></svg>ineligible</span>
    <span><svg width="18" height="18"><circle cx="9" cy="9" r="5" fill="var(--muted)"/><circle cx="9" cy="9" r="8" fill="none" stroke="var(--fg)" stroke-width="1.5"/></svg>selected</span>
  </span>
</div>

<h2>Selected point after each stage</h2>
<div class="panel"><svg class="chart" id="path" viewBox="0 0 1000 260" role="img" aria-label="Dev metric of the selected configuration after each stage, per arm"></svg></div>

<h2>Round 1: random configurations</h2>
<p>Each dot is one configuration after 128 steps, placed by learning rate. Hollow dots damaged the answers or the format and could not be selected; the axis is scaled to eligible runs, and hollow dots beyond it sit at the edge.</p>
<div class="panel"><svg class="chart" id="round1" viewBox="0 0 1000 320" role="img" aria-label="Round 1 metric against learning rate, both arms"></svg></div>

<h2>Greedy sweeps</h2>
<p>Each panel varies one setting around the current point (labelled "current"). Axes are scaled to eligible runs; ineligible runs off that scale sit at the panel edge. "failed" runs broke so badly that nothing could be scored.</p>
<div class="grid2" id="sweeps"></div>

<h2 id="final-h">Final round</h2>
<div id="final"></div>

<h2>Every run</h2>
<div class="panel scroll"><table id="table"></table></div>
<div class="tip" id="tip" role="status"></div>
</main>
<script>
const D = __DATA__;
const COLOR = {exact: "var(--exact)", ppo: "var(--ppo)"};
const NAME = {exact: "Exact + hinge", ppo: "PPO + hinge"};
const LABEL = {brier: "Brier (dev)", ece: "ECE (dev)", auroc: "AUROC (dev)", accuracy: "Answer accuracy (dev)"};
let metric = "brier";
const NS = "http://www.w3.org/2000/svg", tip = document.getElementById("tip");
const el = (n, a, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
const fmt = v => v == null ? "—" : (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : (+v).toFixed(3));
const fmtLr = v => v.toExponential(1).replace("e-0", "e-");
function paramsText(p) { return Object.entries(p).map(([k, v]) => `${k} ${k === "lr" ? fmtLr(v) : v}`).join(" · "); }
function showTip(evt, r) {
  tip.textContent = "";
  const add = (t, cls) => { const d = document.createElement("div"); if (cls) d.className = cls; d.textContent = t; tip.appendChild(d); return d; };
  const h = add(`${NAME[r.arm]} · ${r.label}`); h.style.fontWeight = 600;
  add(paramsText(r.params)).style.color = "var(--muted)";
  if (r.status === "failed") add("failed: no well-formed output to score");
  else {
    add(`Brier ${fmt(r.brier)} · ECE ${fmt(r.ece)} · AUROC ${fmt(r.auroc)}`);
    add(`accuracy ${(100 * r.accuracy).toFixed(1)}% · format failures ${(100 * r.wrong_format).toFixed(1)}%`);
    if (!r.eligible) add(r.accuracy < D.base_accuracy - 0.02 ? "ineligible: answers damaged" : "ineligible: format broken").style.color = "var(--bad)";
  }
  if (r.selected) add("selected at this stage").style.fontWeight = 600;
  tip.style.display = "block";
  tip.style.left = Math.min(evt.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = (evt.clientY + 14) + "px";
}
const hideTip = () => tip.style.display = "none";
function niceTicks(lo, hi, n) { const s = (hi - lo) / n || 1, m = Math.pow(10, Math.floor(Math.log10(s))), st = [1, 2, 2.5, 5, 10].map(x => x * m).find(x => x >= s); const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-12; v += st) out.push(+v.toFixed(10)); return out; }
function yScale(values, top, bottom) {
  // Scale to the given values; points outside are pinned to the plot edge (see dot()).
  const v = values.filter(x => x != null && !isNaN(x)); let lo = Math.min(...v), hi = Math.max(...v);
  const pad = (hi - lo) * 0.12 || 0.02; lo -= pad; hi += pad; const ticks = niceTicks(lo, hi, 4);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 0.1;
  return {Y: x => Math.max(top, Math.min(bottom, bottom - (x - lo) / (hi - lo) * (bottom - top))), ticks,
          digits: step < 0.01 ? 3 : 2};
}
const scaleValues = rows => { const ok = rows.filter(r => r.eligible).map(r => r[metric]); return ok.length ? ok : rows.map(r => r[metric]); };
function dot(g, x, y, r, evt) {
  const filled = r.eligible;
  if (r.selected) el("circle", {cx: x, cy: y, r: 10, fill: "none", stroke: "var(--fg)", "stroke-width": 1.5}, g);
  const c = el("circle", {cx: x, cy: y, r: 6, fill: filled ? COLOR[r.arm] : "var(--surface)", stroke: COLOR[r.arm], "stroke-width": 2.5, tabindex: 0}, g);
  const hit = el("circle", {cx: x, cy: y, r: 13, fill: "transparent"}, g);
  for (const t of [c, hit]) { t.addEventListener("pointermove", e => showTip(e, r)); t.addEventListener("pointerleave", hideTip); }
  c.addEventListener("focus", () => { const b = c.getBoundingClientRect(); showTip({clientX: b.right, clientY: b.bottom}, r); });
  c.addEventListener("blur", hideTip);
}
function axes(svg, scale, x0, x1) {
  for (const t of scale.ticks) { el("line", {x1: x0, x2: x1, y1: scale.Y(t), y2: scale.Y(t), stroke: "var(--grid)"}, svg); el("text", {x: x0 - 8, y: scale.Y(t) + 4, "text-anchor": "end", class: "tick"}, svg).textContent = t.toFixed(scale.digits); }
}

function cards() {
  const box = document.getElementById("cards"); box.textContent = "";
  for (const arm of ["exact", "ppo"]) {
    const sel = D.runs.filter(r => r.arm === arm && r.selected).pop(), p = D.current[arm];
    const c = document.createElement("div"); c.className = "card";
    const h = document.createElement("h3"); const s = document.createElement("i"); s.className = "sw"; s.style.background = COLOR[arm];
    h.append(s, document.createTextNode(NAME[arm] + ": chosen"));
    const dl = document.createElement("dl"); dl.className = "kv";
    const rows = [["learning rate", fmtLr(p.lr)], ["updates per batch", p.updates], ["hinge", `weight ${p.hinge_weight}, threshold ${p.hinge_threshold}`]];
    if (arm === "ppo") rows.push(["clip / value weight", `${p.cliprange} / ${p.vf_coef}`]);
    if (sel) rows.push(["dev Brier · ECE · AUROC", `${fmt(sel.brier)} · ${fmt(sel.ece)} · ${fmt(sel.auroc)}`], ["dev accuracy", (100 * sel.accuracy).toFixed(1) + "%"]);
    for (const [k, v] of rows) { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); }
    c.append(h, dl); box.appendChild(c);
  }
  const n = D.runs.length, bad = D.runs.filter(r => !r.eligible).length;
  const c = document.createElement("div"); c.className = "card";
  const h = document.createElement("h3"); h.textContent = "Search so far"; c.appendChild(h);
  const dl = document.createElement("dl"); dl.className = "kv";
  for (const [k, v] of [["runs scored or failed", n], ["ineligible or failed", bad], ["final round", D.final ? "done" : "running"]]) {
    const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); }
  c.appendChild(dl); box.appendChild(c);
}

function pathChart() {
  const svg = document.getElementById("path"); svg.textContent = "";
  const stages = D.stages.filter(([s]) => D.runs.some(r => r.stage === s));
  const x0 = 70, x1 = 960, top = 20, bottom = 210, X = i => x0 + 40 + i * (x1 - x0 - 80) / Math.max(1, stages.length - 1);
  const pts = {exact: [], ppo: []};
  stages.forEach(([s], i) => { for (const arm of ["exact", "ppo"]) { const r = D.runs.find(r => r.stage === s && r.arm === arm && r.selected); if (r) pts[arm].push([i, r]); } });
  const sc = yScale([...pts.exact, ...pts.ppo].map(([, r]) => r[metric]), top, bottom), Y = sc.Y;
  axes(svg, sc, x0, x1);
  stages.forEach(([s, name], i) => el("text", {x: X(i), y: bottom + 24, "text-anchor": "middle", class: "tick"}, svg).textContent = name.split(":")[0]);
  el("text", {x: 14, y: (top + bottom) / 2, transform: `rotate(-90 14 ${(top + bottom) / 2})`, "text-anchor": "middle", class: "axt"}, svg).textContent = LABEL[metric];
  for (const arm of ["exact", "ppo"]) {
    if (!pts[arm].length) continue;
    el("path", {d: pts[arm].map(([i, r], j) => `${j ? "L" : "M"}${X(i)},${Y(r[metric])}`).join(""), fill: "none", stroke: COLOR[arm], "stroke-width": 2}, svg);
    for (const [i, r] of pts[arm]) dot(svg, X(i), Y(r[metric]), r);
  }
}

function round1() {
  const svg = document.getElementById("round1"); svg.textContent = "";
  const rows = D.runs.filter(r => r.stage === "round1" && r.status !== "failed" && r[metric] != null && !isNaN(r[metric]));
  const x0 = 70, x1 = 960, top = 20, bottom = 270;
  const lo = Math.log10(2e-6), hi = Math.log10(4e-4), X = v => x0 + (Math.log10(v) - lo) / (hi - lo) * (x1 - x0);
  const sc = yScale(scaleValues(rows), top, bottom), Y = sc.Y;
  axes(svg, sc, x0, x1);
  for (const v of [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]) { el("line", {x1: X(v), x2: X(v), y1: bottom, y2: bottom + 5, stroke: "var(--line)"}, svg); el("text", {x: X(v), y: bottom + 20, "text-anchor": "middle", class: "tick"}, svg).textContent = fmtLr(v); }
  el("text", {x: (x0 + x1) / 2, y: bottom + 42, "text-anchor": "middle", class: "axt"}, svg).textContent = "learning rate (log scale) · paper default 1e-5";
  el("line", {x1: X(1e-5), x2: X(1e-5), y1: top, y2: bottom, stroke: "var(--faint)", "stroke-dasharray": "3 4"}, svg);
  el("text", {x: 14, y: (top + bottom) / 2, transform: `rotate(-90 14 ${(top + bottom) / 2})`, "text-anchor": "middle", class: "axt"}, svg).textContent = LABEL[metric];
  for (const r of rows) dot(svg, X(r.params.lr), Y(r[metric]), r);
}

function sweepLabel(stage, r, current) {
  if (stage === "stage1" || stage === "stage3") return fmtLr(r.params.lr);
  if (stage === "stage2") return `${r.params.updates}×`;
  const c = Object.entries(r.params).filter(([k, v]) => current && current[k] !== v);
  return c.length ? c.map(([k, v]) => `${k.replace("hinge_", "").replace("cliprange", "clip").replace("vf_coef", "vf")} ${v}`).join(", ") : "current";
}
function sweeps() {
  const box = document.getElementById("sweeps"); box.textContent = "";
  for (const [stage, name] of D.stages.slice(1)) {
    const prev = D.history[D.history.findIndex(h => h.stage === stage) - 1];
    for (const arm of ["exact", "ppo"]) {
      const rows = D.runs.filter(r => r.stage === stage && r.arm === arm);
      if (!rows.length) continue;
      const current = prev ? prev.current[arm] : null;
      const order = stage === "stage2" ? (a, b) => a.params.updates - b.params.updates : (stage === "stage4" ? () => 0 : (a, b) => a.params.lr - b.params.lr);
      rows.sort(order);
      const panel = document.createElement("div"); panel.className = "panel";
      const h = document.createElement("h4"); h.textContent = `${name} · ${NAME[arm]}`; panel.appendChild(h);
      const sub = document.createElement("div"); sub.className = "sub";
      sub.textContent = stage === "stage2" ? "x: optimizer updates per batch" : stage === "stage4" ? "x: setting changed from the current point" : "x: learning rate";
      panel.appendChild(sub);
      const svg = el("svg", {class: "chart", viewBox: "0 0 460 210", role: "img", "aria-label": `${name}, ${NAME[arm]}: ${LABEL[metric]} for each setting`});
      panel.appendChild(svg);
      const x0 = 56, x1 = 445, top = 14, bottom = 160, X = i => x0 + (i + 0.5) * (x1 - x0) / rows.length;
      const scored = rows.filter(r => r[metric] != null && !isNaN(r[metric]));
      if (scored.length) {
        const sc = yScale(scaleValues(scored), top, bottom), Y = sc.Y;
        axes(svg, sc, x0, x1);
        rows.forEach((r, i) => {
          const lab = sweepLabel(stage, r, current);
          const t = el("text", {x: X(i), y: bottom + 18, "text-anchor": "middle", class: "tick"}, svg); t.textContent = lab.length > 16 ? lab.slice(0, 15) + "…" : lab;
          if (r.status === "failed") { el("text", {x: X(i), y: bottom - 6, "text-anchor": "middle", class: "tick", fill: "var(--bad)"}, svg).textContent = "failed"; return; }
          if (r[metric] == null || isNaN(r[metric])) return;
          dot(svg, X(i), Y(r[metric]), r);
        });
      }
      box.appendChild(panel);
    }
  }
}

function finalSection() {
  const box = document.getElementById("final"); box.textContent = "";
  if (!D.final) { const p = document.createElement("p"); p.textContent = "Running: each arm's chosen configuration with 3 fresh seeds for 256 steps, scored on dev and once on test. This page refreshes from the run directory when it finishes."; box.appendChild(p); return; }
  const wrap = document.createElement("div"); wrap.className = "panel scroll"; const t = document.createElement("table");
  const hr = t.insertRow(); for (const c of ["run", "split", "Brier", "ECE", "AUROC", "accuracy", "format failures"]) { const th = document.createElement("th"); th.textContent = c; hr.appendChild(th); }
  for (const [label, v] of Object.entries(D.final)) for (const [split, m] of [["dev", v.dev], ["test", v.test]]) {
    if (!m) continue; const r = t.insertRow();
    for (const x of [label, split, fmt(m.brier), fmt(m.ece), fmt(m.auroc), (100 * m.accuracy).toFixed(1) + "%", (100 * m.wrong_format_rate).toFixed(1) + "%"]) r.insertCell().textContent = x;
  }
  wrap.appendChild(t); box.appendChild(wrap);
}

let sortKey = "stage", sortDir = 1;
function table() {
  const t = document.getElementById("table"); t.textContent = "";
  const cols = [["stage", "stage"], ["arm", "arm"], ["params", "settings"], ["brier", "Brier"], ["ece", "ECE"], ["auroc", "AUROC"], ["accuracy", "accuracy"], ["wrong_format", "format fail"], ["status", "status"]];
  const hr = t.insertRow();
  for (const [k, name] of cols) { const th = document.createElement("th"); th.textContent = name + (sortKey === k ? (sortDir > 0 ? " ▲" : " ▼") : ""); th.addEventListener("click", () => { sortDir = sortKey === k ? -sortDir : 1; sortKey = k; table(); }); hr.appendChild(th); }
  const stageIndex = s => D.stages.findIndex(([x]) => x === s);
  const rows = [...D.runs].sort((a, b) => {
    const va = sortKey === "stage" ? stageIndex(a.stage) : sortKey === "params" ? a.label : a[sortKey], vb = sortKey === "stage" ? stageIndex(b.stage) : sortKey === "params" ? b.label : b[sortKey];
    if (va == null) return 1; if (vb == null) return -1; return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir; });
  for (const r of rows) {
    const tr = t.insertRow(); if (r.selected) tr.className = "sel"; else if (!r.eligible) tr.className = "bad";
    for (const v of [r.stage, NAME[r.arm], paramsText(r.params), fmt(r.brier), fmt(r.ece), fmt(r.auroc), r.accuracy == null ? "—" : (100 * r.accuracy).toFixed(1) + "%", r.wrong_format == null ? "—" : (100 * r.wrong_format).toFixed(1) + "%", r.selected ? "selected" : r.status]) tr.insertCell().textContent = v;
  }
}
function render() { cards(); pathChart(); round1(); sweeps(); finalSection(); table(); }
for (const b of document.querySelectorAll("#metric button")) b.addEventListener("click", () => { document.querySelectorAll("#metric button").forEach(x => x.setAttribute("aria-pressed", x === b)); metric = b.dataset.v; render(); });
render();
</script></body></html>
"""


def main(search: Path):
    data = load(search)
    page = PAGE.replace("__DATA__", json.dumps(data, default=str).replace("</", "<\\/")).replace(
        "__BASEACC__", f"{100 * data['base_accuracy']:.1f}%")
    out = search / "hparam_search.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
