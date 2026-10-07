"""Interactive learning curves of the minimal trainer's runs, one page in the browser.

    python scripts/plot_learning_curves.py [PATTERN ...]   # default runs/minimal/llama-*; writes docs/learning_curves.html

Per arm (a run name without its -s<seed> suffix), the mean over seeds at each evaluation step of: Brier and AUROC of
the expected confidence on the cached dev answers, the change in regenerated dev accuracy from the run's step 0,
and the answer KL to the base model; plus a scatter of answer KL against that accuracy change for every evaluation.
Plotly (from a CDN) draws them: hover for values, click a legend entry to hide or show an arm.
"""
import collections
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arms(patterns):
    runs = collections.defaultdict(list)
    for f in sorted({f for p in patterns for f in glob.glob(str(ROOT / p / "train.log"))}):
        name = Path(f).parent.name
        if "-s" not in name or "timing" in name or "baseseed" in name:
            continue
        curve = {r["step"]: r for r in (json.loads(l) for l in open(f) if l.startswith('{"step"'))}
        if len(curve) > 1:
            runs[name.rsplit("-s", 1)[0]].append(curve)
    out = {}
    for arm, cs in sorted(runs.items()):
        steps = sorted(set.intersection(*(set(c) for c in cs)))
        mean = lambda key, s: st.fmean(c[s][key] for c in cs if key in c[s])
        regen_steps = [s for s in steps if all("regen_accuracy" in c[s] for c in cs)]
        base = st.fmean(c[0]["regen_accuracy"] for c in cs) if 0 in regen_steps else None
        out[arm] = dict(
            seeds=len(cs), steps=steps,
            brier=[mean("brier_expected", s) for s in steps], auroc=[mean("auroc_expected", s) for s in steps],
            kl=[mean("answer_kl", s) for s in steps],
            regen_steps=regen_steps, dacc=[mean("regen_accuracy", s) - base for s in regen_steps] if base is not None else [],
            points=[dict(kl=c[s]["answer_kl"], dacc=c[s]["regen_accuracy"] - c[0]["regen_accuracy"], step=s)
                    for c in cs for s in steps if s > 0 and "regen_accuracy" in c[s] and "regen_accuracy" in c.get(0, {})])
    return out


def main():
    patterns = sys.argv[1:] or ["runs/minimal/llama-*"]
    data = arms(patterns)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learning curves</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root { --fg:#1d2433; --muted:#5b6475; --bg:#ffffff; --line:#e3e6ec; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6e8ec; --muted:#a3abb9; --bg:#15181e; --line:#2c313b; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:1400px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:21px; margin:0 0 4px; } p { color:var(--muted); margin:0 0 12px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(520px, 1fr)); gap:14px; }
.panel { border:1px solid var(--line); border-radius:6px; height:430px; }
#hovered { position:sticky; top:0; z-index:5; min-height:24px; padding:4px 0 8px; background:var(--bg); font-weight:600; }
</style></head><body><main>
<h1>Learning curves</h1>
<p>__SUBTITLE__</p>
<div id="hovered">&nbsp;</div>
<div class="grid">
<div id="brier" class="panel"></div><div id="auroc" class="panel"></div>
<div id="dacc" class="panel"></div><div id="kl" class="panel"></div>
<div id="scatter" class="panel" style="grid-column:1/-1"></div>
</div>
<script>
const data = __DATA__;
const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const fg = dark ? "#e6e8ec" : "#1d2433", grid = dark ? "#2c313b" : "#e3e6ec";
const palette = ["#4c78a8","#f58518","#54a24b","#e45756","#72b7b2","#b279a2","#ff9da6","#9d755d","#bab0ac","#eeca3b","#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const names = Object.keys(data);
const color = Object.fromEntries(names.map((n, i) => [n, palette[i % palette.length]]));
function layout(title, ytitle, extra) {
  return Object.assign({title: {text: title, font: {size: 15}}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: fg, size: 11}, margin: {l: 55, r: 10, t: 40, b: 45}, hovermode: "closest",
    xaxis: {title: "step", gridcolor: grid}, yaxis: {title: ytitle, gridcolor: grid},
    legend: {font: {size: 10}}}, extra || {});
}
function lines(key, xkey) {
  return names.filter(n => data[n][key].length).map(n => ({x: data[n][xkey || "steps"], y: data[n][key], name: `${n} (${data[n].seeds})`,
    legendgroup: n, mode: "lines+markers", marker: {size: 4}, line: {color: color[n], width: 1.6},
    hovertemplate: "<b>" + n + "</b><br>step %{x}<br>%{y:.3f}<extra></extra>"}));
}
const cfg = {responsive: true, displaylogo: false};
Plotly.newPlot("brier", lines("brier"), layout("Brier (expected confidence, cached dev answers) ↓", "Brier"), cfg);
Plotly.newPlot("auroc", lines("auroc"), layout("AUROC ↑", "AUROC"), cfg);
Plotly.newPlot("dacc", lines("dacc", "regen_steps"), layout("Regenerated dev accuracy − step 0", "Δ accuracy",
  {shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: -0.02, y1: 0.02, fillcolor: "#999", opacity: 0.15, line: {width: 0}}]}), cfg);
Plotly.newPlot("kl", lines("kl"), layout("Answer KL to the base model (nats per answer)", "answer KL", {yaxis: {type: "log", title: "answer KL", gridcolor: grid}}), cfg);
Plotly.newPlot("scatter", names.filter(n => data[n].points.length).map(n => ({
    x: data[n].points.map(p => Math.max(p.kl, 1e-3)), y: data[n].points.map(p => p.dacc), text: data[n].points.map(p => `step ${p.step}`),
    name: n, legendgroup: n, mode: "markers", marker: {size: 7, color: color[n], opacity: 0.75},
    hovertemplate: "<b>" + n + "</b><br>%{text}<br>KL %{x:.2f}<br>Δacc %{y:+.3f}<extra></extra>"})),
  layout("Answer KL against accuracy change, every evaluation (each seed)", "Δ accuracy",
    {xaxis: {type: "log", title: "answer KL (nats per answer)", gridcolor: grid},
     shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: -0.02, y1: 0.02, fillcolor: "#999", opacity: 0.15, line: {width: 0}}]}), cfg);
// Hovering a line or point highlights that arm in every panel (thicker, others faded) and names it above the plots.
const panels = ["brier", "auroc", "dacc", "kl", "scatter"].map(id => document.getElementById(id));
const label = document.getElementById("hovered");
let current = null;
function highlight(group) {
  if (group === current) return;
  current = group;
  for (const el of panels) {
    const idx = el.data.map((_, i) => i);
    Plotly.restyle(el, {opacity: el.data.map(t => group === null || t.legendgroup === group ? 1 : 0.12)}, idx);
    if (el.id !== "scatter")
      Plotly.restyle(el, {"line.width": el.data.map(t => t.legendgroup === group ? 3.6 : 1.6)}, idx);
  }
  label.innerHTML = group === null ? "&nbsp;" : `<span style="color:${color[group]}">■</span> ${group} (${data[group].seeds} seeds)`;
}
for (const el of panels) {
  el.on("plotly_hover", e => highlight(e.points[0].data.legendgroup));
  el.on("plotly_unhover", () => highlight(null));
}
</script>
</main></body></html>"""
    subtitle = (f"{len(data)} arms from {', '.join(patterns)}; each line is the mean over seeds (count in the legend). "
                "Grey band: ±2 points of accuracy. Click a legend entry to hide or show an arm; double-click to isolate it.")
    out = ROOT / "docs/learning_curves.html"
    out.write_text(html.replace("__DATA__", json.dumps(data)).replace("__SUBTITLE__", subtitle))
    print(f"{len(data)} arms -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
