"""Unselected confidence mass (1 - sum q^2: what one sampled confidence leaves out) across runs, in the browser.

    python scripts/plot_unselected_mass.py OUT.html

Start and end points of every run with per-example dumps (runs/minimal/unselected_mass_endpoints.json, from the dumps'
base and fitted level distributions; dev and train), and full curves of the runs whose train.log logs it
(minimal_trainer.py since 2026-10-07): runs/minimal/*/train.log with "unselected_mass".
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ends = json.loads((ROOT / "runs/minimal/unselected_mass_endpoints.json").read_text())
    curves = {}
    for f in sorted(glob.glob(str(ROOT / "runs/minimal/*/train.log"))):
        rows = [json.loads(l) for l in open(f) if l.startswith('{"step"') and '"unselected_mass"' in l]
        if len(rows) > 1:
            curves[Path(f).parent.name] = dict(steps=[r["step"] for r in rows], mass=[r["unselected_mass"] for r in rows],
                                               brier=[r["brier_expected"] for r in rows])
    data = dict(ends={k: {s: dict(mean=st.fmean(v), lo=min(v), hi=max(v)) for s, v in d.items()} for k, d in ends.items()}, curves=curves)
    page = PAGE.replace("__DATA__", json.dumps(data))
    Path(sys.argv[1]).write_text(page)
    print(f"{len(ends)} endpoint arms, {len(curves)} curves -> {sys.argv[1]}")


PAGE = """<title>Unselected Confidence Mass</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root { --fg:#18222b; --muted:#56636e; --bg:#f4f6f7; --surface:#ffffff; --line:#d7dde2; --grid:#e8ecef; --base:#8b97a1; --dev:#0b7a83; --train:#c25a12; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --fg:#e6ecf0; --muted:#a3b0ba; --bg:#11171c; --surface:#172028; --line:#2c3842; --grid:#222c35; --base:#73818c; --dev:#3fb7c0; --train:#ee8a45; color-scheme: dark } }
:root[data-theme="dark"] { --fg:#e6ecf0; --muted:#a3b0ba; --bg:#11171c; --surface:#172028; --line:#2c3842; --grid:#222c35; --base:#73818c; --dev:#3fb7c0; --train:#ee8a45; color-scheme: dark }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding-inline:16px; padding-block:24px 48px; }
h1 { font-size:24px; margin:0 0 6px; text-wrap:balance; } p { color:var(--muted); max-width:72ch; margin:0 0 12px; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:6px; margin:14px 0; }
#ends { height:560px; } #curve { height:380px; }
</style>
<main>
<h1>Unselected confidence mass</h1>
<p>If the stated confidence is sampled once from the model's distribution q over the 11 levels, the sample leaves out 1 − q(k) of the mass;
on average 1 − Σ q². It is 0 for a one-hot q (sampling loses nothing against the exact expectation) and 0.91 for a uniform one.
Below: each run's base model and final checkpoint (from the per-example dumps), then the curve of the first run that logs it every evaluation.</p>
<div id="ends" class="panel"></div>
<div id="curve" class="panel"></div>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue("--" + n).trim();
function draw() {
  const fg = css("fg"), muted = css("muted"), grid = css("grid");
  const arms = Object.keys(D.ends).reverse();
  const trace = (key, name, color, symbol) => ({
    type: "scatter", mode: "markers", name, y: arms, x: arms.map(a => D.ends[a][key].mean),
    error_x: { type: "data", symmetric: false, array: arms.map(a => D.ends[a][key].hi - D.ends[a][key].mean),
               arrayminus: arms.map(a => D.ends[a][key].mean - D.ends[a][key].lo), color, thickness: 1.2, width: 4 },
    marker: { color, size: 10, symbol, line: { color, width: 1.5 } },
    hovertemplate: "<b>%{y}</b><br>" + name + ": %{x:.3f}<extra></extra>" });
  const links = arms.map(a => ({ type: "scatter", mode: "lines", x: [D.ends[a].base.mean, D.ends[a].dev.mean], y: [a, a],
                                 line: { color: grid, width: 3 }, hoverinfo: "skip", showlegend: false }));
  const layout = (title, xt, yt, extra) => Object.assign({ title: { text: title, font: { size: 15, color: fg }, x: 0.01 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: muted, size: 12 },
    margin: { l: 220, r: 20, t: 46, b: 50 }, legend: { orientation: "h", y: -0.12 },
    xaxis: { title: xt, gridcolor: grid, zeroline: false }, yaxis: { title: yt, gridcolor: grid } }, extra || {});
  Plotly.react("ends", [...links, trace("base", "base model (dev)", css("base"), "circle"),
                        trace("dev", "final checkpoint, dev", css("dev"), "circle"),
                        trace("train", "final checkpoint, training set", css("train"), "circle-open")],
    layout("Start and end of each run (mean over seeds; whiskers: seed range)", "unselected mass (1 − Σ q²)", "",
           { xaxis: { title: "unselected mass (1 − Σ q²)", range: [0, 0.45], gridcolor: grid, zeroline: false } }), { responsive: true, displaylogo: false });
  const names = Object.keys(D.curves);
  Plotly.react("curve", names.map((n, i) => ({ type: "scatter", mode: "lines+markers", name: n + " (dev)", x: D.curves[n].steps, y: D.curves[n].mass,
      line: { color: css("dev"), width: 2 }, hovertemplate: n + "<br>step %{x}: %{y:.3f}<extra></extra>" })),
    layout("Over training (runs that log it at every evaluation)", "training step", "unselected mass",
           { margin: { l: 60, r: 20, t: 46, b: 50 }, yaxis: { title: "unselected mass", range: [0, 0.6], gridcolor: grid } }), { responsive: true, displaylogo: false });
}
draw();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script>
"""

if __name__ == "__main__":
    main()
