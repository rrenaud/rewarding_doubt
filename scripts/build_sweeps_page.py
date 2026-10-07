"""Build docs/sweeps/index.html, one web page with every sweep report from docs/sweeps/*.md plus charts.

    python scripts/build_sweeps_page.py

The Markdown files stay the source of truth: each becomes a section (links between reports become in-page anchors,
links to repository files point at GitHub). CHARTS below adds plots after a report's results; their numbers are the
ones in the reports' tables, or come from docs/sweeps/page_data.json (round-1 search points, 2,000-step curves).
Charts are drawn with Chart.js from cdnjs and follow the page's light/dark theme.
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "docs" / "sweeps"
GITHUB = "https://github.com/rrenaud/rewarding_doubt/blob/main/"
DATA = json.loads((SWEEPS / "page_data.json").read_text())

E, P, R, G, B = "exact", "ppo", "paper", "faint", "blue"  # color tokens: teal, orange, purple, grey, blue


def line(x, y, label, color, dashed=False):
    return dict(label=label, color=color, dashed=dashed, data=[dict(x=a, y=b) for a, b in zip(x, y) if b is not None])


def flat(label, x0, x1, y, color=R):
    return line([x0, x1], [y, y], label, color, dashed=True)


def bars(labels, series):
    return dict(labels=labels, datasets=[dict(label=n, color=c, data=v) for n, v, c in series])


r1 = DATA["round1"]
curves = DATA["lr_schedule_curves"]
llama_points = [  # (label, answer KL, regenerated accuracy, small adapter?) at step 500, from reports 18-23
    ("o_proj bias 3e-3", 4.9, .534, 1), ("o_proj bias 1e-2", 10.9, .418, 1), ("hooked bias 1e-2", 17.7, .403, 1),
    ("norm gains 3e-3", .24, .651, 1), ("o_proj bias 3e-4, unpenalized", .93, .610, 1),
    ("o_proj bias 1e-3, unpenalized", 19.5, .262, 1), ("norm gains 3e-3, unpenalized", .51, .639, 1),
    ("norm gains 1e-2, unpenalized", 16.2, .295, 1), ("o-LoRA 16-31 3e-4", .007, .674, 1), ("o-LoRA 16-31 1e-3", .17, .664, 1),
    ("full LoRA, W=1", .78, .661, 0), ("full LoRA, released schedule", .94, .646, 0), ("LoRA 8-31", .49, .649, 0),
    ("LoRA 16-31", .15, .659, 0), ("LoRA 24-31", .045, .661, 0), ("LoRA 16-31 rank 1", .005, .667, 0),
    ("LoRA 16-31 rank 2", .006, .664, 0), ("LoRA 16-31 rank 4", .024, .668, 0),
]
CHARTS = {
    "01": [
        dict(type="scatter", title="Round 1: every scored configuration after 128 steps", xlabel="learning rate", ylabel="dev Brier (lower is better)",
             xlog=True, datasets=[
                 dict(label=f"{arm} {'eligible' if el else 'ineligible'}", color=E if arm == "exact" else P, hollow=not el,
                      data=[dict(x=p["lr"], y=p["brier"], label=f"lr {p['lr']:.2g}, accuracy {p['accuracy']:.2f}")
                            for p in r1 if p["arm"] == arm and p["eligible"] == el])
                 for arm in ("exact", "ppo") for el in (True, False)],
             note="Hollow points damaged the answers or the format; several of them have the best Brier, which is the loophole the eligibility rule closed."),
        dict(type="bar", title="Final round, test set", ylabel="value", **bars(["ECE (lower is better)", "AUROC (higher is better)"], [
            ("exact + hinge (3 seeds)", [.052, .839], E), ("PPO + hinge (3 seeds)", [.128, .757], P), ("PPO + hinge, F1 labels (5 seeds)", [.114, .721], G)])),
    ],
    "02": [dict(type="line", title="Reward mix m against dev Brier (mean over seeds)", xlabel="m (share of Brier in the reward)", ylabel="dev Brier",
                datasets=[line([0, .25, .3, .4, .5, .6, .75, 1], [.164, .175, .173, .174, .162, .163, .241, .202], "512 questions, 128 steps", E),
                          line([0, .5], [.156, .169], "full scale, test (3 seeds)", P)])],
    "03": [dict(type="line", title="Learning rate with frozen answers (dev, mean of 2–5 seeds)", xlabel="learning rate", ylabel="dev Brier", xlog=True,
                datasets=[line([2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4], [.171, .169, .171, .246, .250], "exact + hinge", E),
                          line([2.27e-5, 4.5e-5, 9e-5], [.205, .195, .174], "PPO + hinge", P)],
                note="Exact is flat from 2e-5 to 8e-5 and collapses above; PPO's tuned rate (2.27e-5) is too low once answers are frozen.")],
    "04": [dict(type="bar", title="Dev ECE by KL setting (mean of steps 1,000–3,000)", ylabel="ECE (lower is better)", **bars(
        ["adaptive, target 6", "adaptive, target 20", "fixed 0.05", "none"], [("PPO", [.106, .128, .162, .114], P), ("exact", [.035, .051, .055, .071], E)]))],
    "06": [dict(type="bar", title="Dev AUROC by arm", ylabel="AUROC (higher is better)", ymin=.75, **bars(
        ["no check", "filler", "frozen check", "trained check, β 0.05", "trained check, β 0.005"], [("AUROC", [.783, .782, .823, .792, .815], E)]))],
    "07": [dict(type="line", title="One update per batch of 32: learning rate against dev Brier (300 steps)", xlabel="learning rate", ylabel="dev Brier",
                xlog=True, datasets=[line([1e-5, 3e-5, 1e-4], [.183, .155, .117], "batch 32, 5.4 min", E),
                                     flat("released schedule, 1,000 steps, 48 min", 1e-5, 1e-4, .133)])],
    "08": [dict(type="line", title="Dev Brier at step 300 by optimizer and learning rate", xlabel="learning rate", ylabel="dev Brier (collapse ≈ 0.31)",
                xlog=True, datasets=[
                    line([1e-4, 3e-4, 1e-3], [.122, .120, .202], "Adam (3 seeds at 3e-4 and 1e-3)", E),
                    line([3e-5, 1e-4, 3e-4, 1e-3], [.153, .117, .314, .326], "Muon", P),
                    line([3e-5, 1e-4, 3e-4, 1e-3], [.163, .142, .139, .208], "Scaled AdamW", G),
                    line([3e-3, 1e-2, 3e-2], [.124, .135, .123], "PoLoRA", B)],
                note="Each optimizer is best near its own rate; at matched rates none beats Adam. PoLoRA's rates are on a different scale.")],
    "09": [dict(type="scatter", title="Trainable parameters against dev Brier at step 300", xlabel="trainable parameters", ylabel="dev Brier", xlog=True, datasets=[
        dict(label="projections, all layers", color=E, data=[dict(x=a, y=b, label=l) for l, a, b in [
            ("all 7", 15.0e6, .120), ("gate, up, down", 11.3e6, .113), ("q, k, v, o", 3.7e6, .122), ("o, down", 4.9e6, .118), ("q, v", 1.8e6, .120),
            ("gate", 3.8e6, .117), ("up", 3.8e6, .118), ("down", 3.8e6, .121), ("o", 1.2e6, .115), ("q", 1.2e6, .129), ("v", .7e6, .129), ("k", .7e6, .145)]]),
        dict(label="all 7, layer range", color=P, data=[dict(x=a, y=b, label=l) for l, a, b in [
            ("layers 0–17", 7.5e6, .117), ("layers 18–35", 7.5e6, .117), ("layers 27–35", 3.7e6, .152), ("layers 32–35", 1.7e6, .233)]])],
        note="Size barely matters within the projections; the last layers alone fail at any size.")],
    "10": [dict(type="scatter", title="Bias-vector and small adapters: parameters against dev Brier", xlabel="trainable parameters", ylabel="dev Brier", xlog=True, datasets=[
        dict(label="layers 18–35 or all", color=E, data=[dict(x=a, y=b, label=l) for l, a, b in [
            ("o LoRA 18–35", 590e3, .118), ("gate biases 3e-3", 396e3, .114), ("attention bias 1e-2", 36.9e3, .114), ("MLP bias 1e-2", 36.9e3, .130),
            ("attention + MLP 1e-2", 73.7e3, .128), ("all-layer LoRA", 15e6, .120)]]),
        dict(label="layers 27–35", color=P, data=[dict(x=a, y=b, label=l) for l, a, b in [
            ("o LoRA 27–35", 295e3, .166), ("MLP bias", 18.4e3, .203), ("attention bias", 18.4e3, .169), ("attention + MLP", 36.9e3, .182)]])])],
    "11": [dict(type="bar", title="Regenerated dev accuracy at step 300 (base 0.426)", ylabel="accuracy", ymin=.2, **bars(
        ["W = 0", "W = 0.1", "W = 1", "W = 10"], [("attention bias", [.217, .423, .439, .430], E), ("all-layer LoRA", [.404, None, .439, None], P)])),
           dict(type="bar", title="Dev Brier at step 300", ylabel="Brier (lower is better)", ymin=.1, **bars(
               ["W = 0", "W = 0.1", "W = 1", "W = 10"], [("attention bias", [.114, .115, .128, .156], E), ("all-layer LoRA", [.112, None, .112, None], P)]))],
    "16": [dict(type="line", title="Recovery at lr 1e-2: penalty weight against accuracy and Brier", xlabel="answer-KL weight W", ylabel="value", xlog=True,
                datasets=[line([.3, 1, 3, 10, 30], [.412, .433, .435, .440, .445], "regenerated accuracy", E),
                          flat("base accuracy 0.426", .3, 30, .426, G)])],
    "17": [dict(type="line", title="Weight floor against dev Brier (steps 500–1,000)", xlabel="floor on the answer-KL weight", ylabel="Brier", xlog=True,
                datasets=[line([.001, .1, .3, 1, 3], [.113, .118, .120, .133, .144], "target 1.5", E),
                          line([.001, .1, .3, 1, 3], [.113, .114, .123, .131, .144], "target 2", P)]),
           dict(type="line", title="Weight floor against answer KL at step 1,000", xlabel="floor on the answer-KL weight", ylabel="answer KL (nats per answer)", xlog=True,
                datasets=[line([.001, .1, .3, 1, 3], [1.68, 1.05, .65, .33, .20], "target 1.5", E),
                          line([.001, .1, .3, 1, 3], [2.26, 1.08, .61, .34, .22], "target 2", P)])],
    "18": [dict(type="scatter", title="Llama: answer drift against regenerated accuracy, every 500-step arm", xlabel="answer KL at step 500 (nats per answer)",
                ylabel="regenerated dev accuracy", xlog=True, datasets=[
                    dict(label="small adapters (biases, norm gains, o-LoRA)", color=P, data=[dict(x=k, y=a, label=l) for l, k, a, s in llama_points if s]),
                    dict(label="LoRA on all projections", color=E, data=[dict(x=k, y=a, label=l) for l, k, a, s in llama_points if not s]),
                    flat("base model 0.669", .004, 25, .669, G)],
                note="LoRA stays near the base accuracy up to about 1 nat; small adapters fall off from a few tenths.")],
    "20": [dict(type="bar", title="o_proj-only LoRA, step 500", ylabel="dev Brier (lower is better)", **bars(
        ["16–31, 3e-4", "16–31, 1e-3", "16–31, 3e-3", "all, 1e-3", "all, 1e-3, W 0"], [("Brier", [.149, .141, .289, .321, .208], E)]))],
    "21": [dict(type="bar", title="Depth of LoRA: dev Brier at step 500", ylabel="Brier (lower is better)", ymin=.1, **bars(
        ["layers 0–31", "8–31", "16–31", "24–31"], [("Brier", [.136, .134, .128, .138], E)])),
           dict(type="bar", title="Depth of LoRA: answer KL at step 500", ylabel="answer KL (nats)", **bars(
               ["layers 0–31", "8–31", "16–31", "24–31"], [("answer KL", [.78, .49, .15, .05], P)]))],
    "22": [dict(type="line", title="Released evaluation on dev: ECE by checkpoint", xlabel="training step", ylabel="ECE (lower is better)", datasets=[
        line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.0925, .0656, .0571, .0309, .0481, .0677, .0705, .0838], "late-half rank 8 (continued past 1,000)", E),
        line([250, 500, 750, 1000], [.0681, .0689, .0475, .0637], "all-layer rank 8", P),
        line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.0604, .1008, .0477, .0410, .0770, .0594, .0809, .0821], "late-half rank 4", B),
        flat("paper (full set) 0.0226", 250, 2000, .0226)]),
           dict(type="line", title="Released evaluation on dev: AUROC by checkpoint", xlabel="training step", ylabel="AUROC (higher is better)", datasets=[
               line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.788, .858, .845, .862, .849, .828, .811, .836], "late-half rank 8 (continued past 1,000)", E),
               line([250, 500, 750, 1000], [.837, .866, .831, .827], "all-layer rank 8", P),
               line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.815, .823, .827, .841, .792, .835, .801, .826], "late-half rank 4", B),
               flat("paper (full set) 0.859", 250, 2000, .859)]),
           dict(type="bar", title="Full validation set (11,313 questions)", ylabel="value", **bars(["ECE (lower is better)", "AUROC (higher is better)"], [
               ("untrained Llama-3-8B", [.303, .625], G), ("paper", [.0226, .859], R), ("ours: late-half, step 1,000", [.031, .877], E),
               ("ours: all-layer, step 500", [.063, .867], P)]))],
    "23": [dict(type="line", title="LoRA rank: dev Brier and AUROC at step 500 (mean of 2 seeds)", xlabel="rank", ylabel="value", xlog=True, datasets=[
        line([1, 2, 4, 8], [.1335, .1295, .1275, .1285], "Brier (lower is better)", E)], note="AUROC: 0.881, 0.887, 0.892, 0.889."),
           dict(type="line", title="LoRA rank: answer KL at step 500", xlabel="rank", ylabel="answer KL (nats)", xlog=True, ylog=True, datasets=[
               line([1, 2, 4, 8], [.005, .006, .023, .155], "answer KL", P)])],
    "24": [dict(type="bar", title="Training-set Brier at the end (fit to the training data; lower is better)", ylabel="train Brier", ymin=.12, **bars(
        ["rank 4 alone", "+ MLP 3e-5", "+ MLP 1e-4", "+ MLP 3e-4", "+ attn+MLP 3e-5", "+ attn+MLP 1e-4"],
        [("train Brier", [.139, .147, .148, .159, .148, .140], E)]))],
    "25": [dict(type="line", title="Dev Brier over 2,000 steps on 8,000 repeated questions (mean of 2 seeds)", xlabel="training step", ylabel="dev Brier",
                datasets=[line(c["steps"][1:], c["brier"][1:], arm, col) for (arm, c), col in zip(curves.items(), (E, B, P))] + [
                    line([2000], [.130], "weights averaged, constant 3e-4", E), line([2000], [.127], "weights averaged, constant 1e-4", B)],
                note="Single points at step 2,000 are the averaged weights (steps 1,000–2,000)."),
           dict(type="bar", title="Overfitting: training vs dev Brier at the end (exact-match labels)", ylabel="Brier", **bars(
               ["500 steps, constant 3e-4", "2,000, constant 3e-4", "2,000, constant 1e-4", "2,000, cosine 3e-4"],
               [("training set", [.139, .052, .048, .030], E), ("dev", [.149, .172, .168, .175], P)]))],
    "26": [dict(type="bar", title="Cosine vs constant: dev AUROC", ylabel="AUROC (higher is better)", ymin=.8, **bars(
        ["1 epoch, 3e-4", "1 epoch, 1e-4", "2 epochs, 3e-4", "2 epochs, 1e-4"],
        [("cosine", [.837, .819, .883, .854], P), ("constant", [.874, .839, .891, .874], E)])),
           dict(type="bar", title="Cosine vs constant: dev Brier", ylabel="Brier (lower is better)", ymin=.1, **bars(
               ["1 epoch, 3e-4", "1 epoch, 1e-4", "2 epochs, 3e-4", "2 epochs, 1e-4"],
               [("cosine", [.154, .165, .131, .144], P), ("constant", [.141, .160, .129, .141], E)]),
               note="Constant 1e-4 at 1 epoch is the mean of steps 200 and 300 (its runs were evaluated every 100 steps).")],
}


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?!\*)", r"<em>\1</em>", text)

    def link(m):
        label, target = m.group(1), m.group(2)
        local = re.match(r"(\d\d)-[\w-]+\.md$", target)
        if local:
            return f'<a href="#s{local.group(1)}">{label}</a>'
        if target.startswith("http"):
            return f'<a href="{target}">{label}</a>'
        path = (SWEEPS / target).resolve().relative_to(ROOT)
        return f'<a href="{GITHUB}{path}">{label}</a>'
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)


def markdown(md):
    """The subset our reports use: headings, paragraphs, bullet and numbered lists (with continuation lines), tables."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1
        elif l.startswith("#"):
            level = len(l) - len(l.lstrip("#"))
            out.append(f"<h{level + 1}>{inline(l[level:].strip())}</h{level + 1}>")
            i += 1
        elif l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            t = ["<div class='table'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
            t += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
            out.append("".join(t) + "</tbody></table></div>")
        elif re.match(r"(- |\d+\. )", l):
            ordered = bool(re.match(r"\d+\. ", l))
            items = []
            while i < len(lines) and (re.match(r"(- |\d+\. )", lines[i]) or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|\||- |\d+\. )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def ratings(md):
    m = re.search(r"\*\*Motivation (★+)\*\*", md)
    k = re.search(r"\*\*Knowledge: ([^*]+)\*\*", md)
    return (len(m.group(1)) if m else 0), (k.group(1).strip() if k else "")


KNOWLEDGE_ORDER = ["high", "medium–high", "medium", "low–medium", "low"]


def sweep_section(path):
    md = path.read_text()
    num = path.name[:2]
    title = re.match(r"# \d\d\. (.+)", md).group(1)
    body = md.split("\n", 1)[1]
    stars, knowledge = ratings(md)
    parts = re.split(r"(?m)^(?=## )", body)
    html_parts = []
    for part in parts:
        html_parts.append(markdown(part))
        if part.startswith("## Results") and num in CHARTS:
            for j, spec in enumerate(CHARTS[num]):
                cid = f"c{num}-{j}"
                CHART_SPECS[cid] = spec
                note = f"<figcaption>{inline(spec['note'])}</figcaption>" if spec.get("note") else ""
                html_parts.append(f"<figure class='chart'><div class='chart-box'><canvas id='{cid}' role='img' "
                                  f"aria-label='{html.escape(spec['title'])}'></canvas></div>{note}</figure>")
    badge = f"<span class='stars' title='motivation'>{'★' * stars}<span class='dim'>{'★' * (3 - stars)}</span></span>"
    kclass = knowledge.replace("–", "-")
    return (num, title, stars, knowledge,
            f"<section class='sweep' id='s{num}'><header><p class='eyebrow'>Sweep {num}</p><h2>{inline(title)}</h2>"
            f"<p class='badges'>{badge}<span class='know k-{kclass}'>knowledge: {html.escape(knowledge)}</span></p></header>"
            + "\n".join(html_parts) + "<p class='top'><a href='#index'>Back to the index</a></p></section>")


CHART_SPECS = {}


def main():
    readme = (SWEEPS / "README.md").read_text()
    sweeps = [sweep_section(p) for p in sorted(SWEEPS.glob("[0-9][0-9]-*.md"))]
    intro, rest = readme.split("## Index", 1)
    index_table, after = rest.split("Not covered here", 1)
    after = "Not covered here" + after
    one_liners = {}
    for row in index_table.splitlines():
        m = re.match(r"\| \[(\d\d)\]\([^)]+\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| [^|]+ \| [^|]+ \| (.+) \|$", row)
        if m:
            one_liners[m.group(1)] = dict(model=m.group(3).strip(), swept=m.group(4).strip(), result=m.group(5).strip())
    intro_html = markdown(intro.split("\n", 1)[1])  # drop the Markdown title
    rows = []
    for num, title, stars, know, _ in sweeps:
        o = one_liners.get(num, {})
        rows.append(f"<tr><td class='num'><a href='#s{num}'>{num}</a></td><td><a href='#s{num}'>{inline(title)}</a>"
                    f"<div class='sub'>{inline(o.get('model', ''))} · {inline(o.get('swept', ''))}</div></td>"
                    f"<td class='stars'>{'★' * stars}<span class='dim'>{'★' * (3 - stars)}</span></td>"
                    f"<td><span class='know k-{know.replace('–', '-')}'>{html.escape(know)}</span></td><td>{inline(o.get('result', ''))}</td></tr>")
    index_html = ("<div class='table'><table class='index'><thead><tr><th>#</th><th>sweep</th><th>motivation</th><th>knowledge</th>"
                  "<th>result</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    grid = ["<div class='matrix' role='table' aria-label='Sweeps by motivation and knowledge'>",
            "<div class='mh' role='columnheader'></div>" + "".join(f"<div class='mh' role='columnheader'>{'★' * s}</div>" for s in (1, 2, 3))]
    for k in KNOWLEDGE_ORDER:
        grid.append(f"<div class='ml' role='rowheader'>{k}</div>")
        for s in (1, 2, 3):
            chips = "".join(f"<a class='chip' href='#s{n}' title='{html.escape(t)}'>{n}</a>" for n, t, st, kn, _ in sweeps if st == s and kn == k)
            grid.append(f"<div class='mc'>{chips}</div>")
    grid.append("</div>")
    toc = "".join(f"<li><a href='#s{n}'><span class='tn'>{n}</span>{inline(t)}</a></li>" for n, t, *_ in sweeps)
    page = (SWEEPS / "page_template.html").read_text()
    page = (page.replace("{{INTRO}}", intro_html).replace("{{MATRIX}}", "".join(grid)).replace("{{INDEX}}", index_html)
            .replace("{{AFTER}}", markdown(after)).replace("{{SWEEPS}}", "\n".join(s[4] for s in sweeps)).replace("{{TOC}}", toc)
            .replace("{{CHARTS}}", json.dumps(CHART_SPECS)))
    (SWEEPS / "index.html").write_text(page)
    print(f"{len(sweeps)} sweeps, {len(CHART_SPECS)} charts -> docs/sweeps/index.html ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
