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
COSTS = json.loads((SWEEPS / "costs.json").read_text())  # scripts/sweep_costs.py



r1 = DATA["round1"]
SEEDS = DATA["seeds"]
BANDS = DATA["lr_schedule_bands"]


def R(key, index=None):
    """Mean over seeds with the seed range: {y, yMin, yMax} (error bars on the page); a plain number for one seed."""
    v = SEEDS[key] if isinstance(key, str) else key
    if index is not None:
        v = [x[index] for x in v]
    if len(v) == 1:
        return v[0]
    return dict(y=round(sum(v) / len(v), 5), yMin=min(v), yMax=max(v))


def pts(xs, ys):
    """Points for line and scatter charts; ys are numbers, R() dicts or None."""
    out = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        out.append(dict(x=x, **y) if isinstance(y, dict) else dict(x=x, y=y))
    return out


def line(x, y, label, color, dashed=False):
    return dict(label=label, color=color, dashed=dashed, data=pts(x, y))


def flat(label, x0, x1, y, color="paper"):
    return dict(label=label, color=color, dashed=True, data=[dict(x=x0, y=y), dict(x=x1, y=y)])


def bars(labels, series):
    return dict(labels=labels, datasets=[dict(label=n, color=c, data=v) for n, v, c in series])


E, P, PAPER, G, B = "exact", "ppo", "paper", "faint", "blue"  # colour tokens: teal, orange, purple, grey, blue
fast = lambda arm: R(f"fast/{arm}")
llama = lambda arm, m="brier": R(f"llama/{arm}/{m}")
SEED_NOTE = "Points are means over seeds; whiskers span the lowest to highest seed."
BASE11 = 0.426  # regenerated dev accuracy of the bf16 Qwen base model


def xy(x, y):
    """A 2-D point from two R() values (numbers or {y, yMin, yMax}), with whiskers on both axes."""
    out = {}
    for axis, v in (("x", x), ("y", y)):
        if isinstance(v, dict):
            out.update({axis: v["y"], f"{axis}Min": v["yMin"], f"{axis}Max": v["yMax"]})
        else:
            out[axis] = v
    return out

KL_SPLIT = 1.2  # sweep 18: answer KL (nats per answer) dividing the mostly stable region from the unstable one


def drift(keep):
    """Sweep 18's per-seed points (answer KL, regenerated accuracy), split by adapter type, filtered by keep(point)."""
    out = []
    for label, small, color in (("small adapters (biases, norm gains, o-LoRA)", True, P), ("LoRA on all projections", False, E)):
        data = [dict(x=max(q["kl"], 1e-3), y=q["acc"], label=f"{q['name']}, seed {q['seed']}")
                for q in SEEDS["18/points"] if q["small"] == small and keep(q)]
        if data:
            out.append(dict(label=label, color=color, data=data))
    return out


CHARTS = {
    "01": [
        dict(type="scatter", title="Round 1: every scored configuration after 128 steps (one seed each)", xlabel="learning rate",
             ylabel="dev Brier (lower is better)", xlog=True, datasets=[
                 dict(label=f"{arm} {'eligible' if el else 'ineligible'}", color=E if arm == "exact" else P, hollow=not el,
                      data=[dict(x=q["lr"], y=q["brier"], label=f"lr {q['lr']:.2g}, accuracy {q['accuracy']:.2f}")
                            for q in r1 if q["arm"] == arm and q["eligible"] == el])
                 for arm in ("exact", "ppo") for el in (True, False)],
             note="Hollow points damaged the answers or the format; several of them have the best Brier, which is the loophole the eligibility rule closed."),
        dict(type="bar", title="Final round, test set", ylabel="value", note=SEED_NOTE, **bars(["ECE (lower is better)", "AUROC (higher is better)"], [
            ("exact + hinge (3 seeds)", [R("01/exact/ece"), R("01/exact/auroc")], E),
            ("PPO + hinge (3 seeds)", [R("01/ppo/ece"), R("01/ppo/auroc")], P),
            ("PPO + hinge, F1 labels (5 seeds)", [R("01/ppo_f1/ece"), R("01/ppo_f1/auroc")], G)])),
    ],
    "02": [dict(type="line", title="Reward mix m against dev Brier", xlabel="m (share of Brier in the reward)", ylabel="dev Brier", note=SEED_NOTE,
                datasets=[line([0, .25, .3, .4, .5, .6, .75, 1], [R(f"02/dev/{m}") for m in ("0.00", "0.25", "0.30", "0.40", "0.50", "0.60", "0.75", "1.00")],
                               "512 questions, 128 steps (2–4 seeds)", E),
                          line([0, .5], [R("02/test/0.00"), R("02/test/0.50")], "full scale, test (3 seeds)", P)])],
    "03": [dict(type="line", title="Learning rate with frozen answers (dev Brier)", xlabel="learning rate", ylabel="dev Brier", xlog=True,
                datasets=[line([2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4], [R("03/exact/2.0e-05"), R("03/exact/tuned"), R("03/exact/8.0e-05"), R("03/exact/1.6e-04"),
                                                                   R("03/exact/3.2e-04")], "exact + hinge", E),
                          line([2.27e-5, 4.5e-5, 9e-5], [R("03/ppo/tuned"), R("03/ppo/4.5e-05"), R("03/ppo/9.0e-05")], "PPO + hinge", P)],
                note="2 seeds per rate, 5 at the tuned rate; whiskers span the seeds. Exact is flat from 2e-5 to 8e-5 and collapses above; "
                     "PPO's tuned rate (2.27e-5) is too low once answers are frozen.")],
    "04": [dict(type="bar", title="Dev ECE by KL setting (mean of steps 1,000–3,000, one seed each)", ylabel="ECE (lower is better)", **bars(
        ["adaptive, target 6", "adaptive, target 20", "fixed 0.05", "none"], [("PPO", [.106, .128, .162, .114], P), ("exact", [.035, .051, .055, .071], E)]))],
    "06": [dict(type="bar", title="Dev AUROC by arm (mean over 2–4 seeds)", ylabel="AUROC (higher is better)", ymin=.75, **bars(
        ["no check", "filler", "frozen check", "trained check, β 0.05", "trained check, β 0.005"], [("AUROC", [.783, .782, .823, .792, .815], E)]),
        note="Per-seed AUROC was not kept; the report gives standard errors of 0.008–0.012 for the 4-seed arms.")],
    "07": [dict(type="line", title="One update per batch of 32: learning rate against dev Brier (300 steps, one seed)", xlabel="learning rate",
                ylabel="dev Brier", xlog=True, datasets=[line([1e-5, 3e-5, 1e-4], [.183, .155, .117], "batch 32, 5.4 min", E),
                                                         flat("released schedule, 1,000 steps, 48 min", 1e-5, 1e-4, .133)])],
    "08": [dict(type="line", title="Dev Brier at step 300 by optimizer and learning rate", xlabel="learning rate", ylabel="dev Brier (collapse ≈ 0.31)",
                xlog=True, datasets=[
                    line([1e-4, 3e-4, 1e-3], [.122, R("08/seeds-adam-lr3e-4"), R("08/seeds-adam-lr1e-3")], "Adam", E),
                    line([3e-5, 1e-4, 3e-4, 1e-3], [.153, .117, .314, .326], "Muon", P),
                    line([3e-5, 1e-4, 3e-4, 1e-3], [.163, .142, R("08/seeds-scaled-adamw-lr3e-4"), R("08/seeds-scaled-adamw-lr1e-3")], "Scaled AdamW", G),
                    line([3e-3, 1e-2, 3e-2], [R("08/seeds-polora-lr3e-3"), .135, .123], "PoLoRA", B)],
                note="Whiskers: 3 seeds (Adam at 3e-4 and 1e-3, Scaled AdamW at 3e-4 and 1e-3, PoLoRA at 3e-3); other points are one seed. "
                     "At matched rates none beats Adam. PoLoRA's rates are on a different scale.")],
    "09": [dict(type="scatter", title="Trainable parameters against dev Brier at step 300", xlabel="trainable parameters", ylabel="dev Brier", xlog=True, datasets=[
        dict(label="projections, all layers", color=E, data=[dict(x=a, label=l, **fast(arm)) for l, a, arm in [
            ("all 7", 15.0e6, "baseline"), ("gate, up, down", 11.3e6, "ablate-mlp"), ("q, k, v, o", 3.7e6, "ablate-attn"), ("o, down", 4.9e6, "ablate-o-down"),
            ("q, v", 1.8e6, "ablate-qv"), ("gate", 3.8e6, "ablate-gate"), ("up", 3.8e6, "ablate-up"), ("down", 3.8e6, "ablate-down"), ("o", 1.2e6, "ablate-o"),
            ("q", 1.2e6, "ablate-q"), ("v", .7e6, "ablate-v"), ("k", .7e6, "ablate-k")]]),
        dict(label="all 7, layer range", color=P, data=[dict(x=a, label=l, **fast(arm)) for l, a, arm in [
            ("layers 0–17", 7.5e6, "ablate-layers0-17"), ("layers 18–35", 7.5e6, "ablate-layers18-35"), ("layers 27–35", 3.7e6, "ablate-layers27-35"),
            ("layers 32–35", 1.7e6, "ablate-layers32-35")]])],
        note="2 seeds per arm (3 for all 7); whiskers span the seeds. Size barely matters within the projections; the last layers alone fail at any size.")],
    "10": [dict(type="scatter", title="Bias-vector and small adapters: parameters against dev Brier", xlabel="trainable parameters", ylabel="dev Brier", xlog=True, datasets=[
        dict(label="layers 18–35 or all", color=E, data=[dict(x=a, label=l, **fast(arm)) for l, a, arm in [
            ("o LoRA 18–35", 590e3, "ablate-o-layers18-35"), ("gate biases 3e-3", 396e3, "gatebias-lr3e-3"), ("attention bias 1e-2", 36.9e3, "attnbias18-35"),
            ("MLP bias 1e-2", 36.9e3, "resbias18-35-lr1e-2"), ("attention + MLP 1e-2", 73.7e3, "attnmlpbias18-35"), ("all-layer LoRA", 15e6, "baseline")]]),
        dict(label="layers 27–35", color=P, data=[dict(x=a, label=l, **fast(arm)) for l, a, arm in [
            ("o LoRA 27–35", 295e3, "ablate-o-layers27-35"), ("MLP bias", 18.4e3, "resbias27-35-lr1e-2"), ("attention bias", 18.4e3, "attnbias27-35"),
            ("attention + MLP", 36.9e3, "attnmlpbias27-35")]])], note=SEED_NOTE + " 2 seeds per arm.")],
    "11": [dict(type="scatter", title="Answer-KL weight: accuracy lost against dev Brier (step 300)",
                xlabel="accuracy drop from the base model (0.426 regenerated; right is worse)", ylabel="dev Brier (lower is better)",
                datasets=[dict(label=name, color=col, connect=True, data=[
                    dict(label=f"{name}, W = {w}", **xy(R([BASE11 - a for a in SEEDS[f"11/{arm}-b{w}/regen"]]), R(f"11/{arm}-b{w}/brier")))
                    for w in ws]) for name, arm, ws, col in (("attention bias", "akl-attnbias", ("0", "0.1", "1", "10"), E),
                                                             ("all-layer LoRA", "akl-lora", ("0", "1"), P))] + [
                    dict(label="no accuracy lost", color=G, dashed=True, data=[dict(x=0, y=.105), dict(x=0, y=.165)])],
                note=SEED_NOTE + " 2 seeds, whiskers in both directions. Lines join each adapter's weights in order "
                     "(W = 0, 0.1, 1, 10): the unpenalized attention bias has a good Brier but loses about 21 points of accuracy.")],
    "16": [dict(type="line", title="Recovery at lr 1e-2: penalty weight against regenerated accuracy (step 300)", xlabel="answer-KL weight W", ylabel="accuracy",
                xlog=True, datasets=[line([.3, 1, 3, 10, 30], [R(f"16/{w}") for w in ("0.3", "1", "3", "10", "30")], "regenerated accuracy", E),
                                     flat("base accuracy 0.426", .3, 30, .426, G)], note=SEED_NOTE + " 2 seeds.")],
    "17": [dict(type="line", title="Weight floor against dev Brier (steps 500–1,000)", xlabel="floor on the answer-KL weight", ylabel="Brier", xlog=True,
                datasets=[line([.001, .1, .3, 1, 3], [R(f"17/1.5/{f}/brier") for f in ("0.001", "0.1", "0.3", "1", "3")], "target 1.5", E),
                          line([.001, .1, .3, 1, 3], [R(f"17/2/{f}/brier") for f in ("0.001", "0.1", "0.3", "1", "3")], "target 2", P)], note=SEED_NOTE + " 2 seeds."),
           dict(type="line", title="Weight floor against answer KL at step 1,000", xlabel="floor on the answer-KL weight", ylabel="answer KL (nats per answer)", xlog=True,
                datasets=[line([.001, .1, .3, 1, 3], [R(f"17/1.5/{f}/kl") for f in ("0.001", "0.1", "0.3", "1", "3")], "target 1.5", E),
                          line([.001, .1, .3, 1, 3], [R(f"17/2/{f}/kl") for f in ("0.001", "0.1", "0.3", "1", "3")], "target 2", P)])],
    "18": [dict(type="scatter", title=f"Llama drift, stable region: answer KL below {KL_SPLIT} nats (each seed)",
                xlabel="answer KL at step 500 (nats per answer)", ylabel="regenerated dev accuracy", xlog=True,
                datasets=drift(lambda q: q["kl"] < KL_SPLIT) + [flat("base model 0.669", 1e-3, KL_SPLIT, .669, G)],
                note="One point per seed. LoRA stays within about 3 points of the base model (0.640–0.673); small adapters lose more per nat, down to 0.598 (−7 points) near 1 nat."),
           dict(type="scatter", title=f"Llama drift, unstable region: answer KL of {KL_SPLIT} nats and above (each seed)",
                xlabel="answer KL at step 500 (nats per answer)", ylabel="regenerated dev accuracy", xlog=True,
                datasets=drift(lambda q: q["kl"] >= KL_SPLIT) + [flat("base model 0.669", KL_SPLIT, 40, .669, G)],
                note="One point per seed. Only small adapters reach this region, and no run landed between 1.2 and about 4 nats; accuracy falls from about 0.55 at 4–5 nats to near 0 at 35.")],
    "20": [dict(type="bar", title="o_proj-only LoRA, dev Brier at step 500", ylabel="Brier (lower is better)", note=SEED_NOTE, **bars(
        ["16–31, 3e-4", "16–31, 1e-3", "16–31, 3e-3", "all, 1e-3", "all, 1e-3, W 0"],
        [("Brier", [llama("llama-lorao-lr3e-4-w1"), llama("llama-lorao16-lr1e-3-w1"), llama("llama-lorao16-lr3e-3-w1"),
                    llama("llama-loraoall-lr1e-3-w1"), llama("llama-loraoall-lr1e-3-w0")], E)]))],
    "21": [dict(type="bar", title="Depth of LoRA: dev Brier at step 500", ylabel="Brier (lower is better)", ymin=.1, note=SEED_NOTE, **bars(
        ["layers 0–31", "8–31", "16–31", "24–31"],
        [("Brier", [llama(f"llama-lorafull{d}lr3e-4-w1") for d in ("-", "8-", "16-", "24-")], E)])),
           dict(type="bar", title="Depth of LoRA: answer KL at step 500", ylabel="answer KL (nats)", note=SEED_NOTE, **bars(
               ["layers 0–31", "8–31", "16–31", "24–31"],
               [("answer KL", [llama(f"llama-lorafull{d}lr3e-4-w1", "kl") for d in ("-", "8-", "16-", "24-")], P)]))],
    "22": [dict(type="line", title="Released evaluation on dev: ECE by checkpoint (one seed per run)", xlabel="training step", ylabel="ECE (lower is better)", datasets=[
        line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.0925, .0656, .0571, .0309, .0481, .0677, .0705, .0838], "late-half rank 8 (continued past 1,000)", E),
        line([250, 500, 750, 1000], [.0681, .0689, .0475, .0637], "all-layer rank 8", P),
        line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.0604, .1008, .0477, .0410, .0770, .0594, .0809, .0821], "late-half rank 4", B),
        flat("paper (full set) 0.0226", 250, 2000, .0226, PAPER)]),
           dict(type="line", title="Released evaluation on dev: AUROC by checkpoint (one seed per run)", xlabel="training step", ylabel="AUROC (higher is better)", datasets=[
               line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.788, .858, .845, .862, .849, .828, .811, .836], "late-half rank 8 (continued past 1,000)", E),
               line([250, 500, 750, 1000], [.837, .866, .831, .827], "all-layer rank 8", P),
               line([250, 500, 750, 1000, 1250, 1500, 1750, 2000], [.815, .823, .827, .841, .792, .835, .801, .826], "late-half rank 4", B),
               flat("paper (full set) 0.859", 250, 2000, .859, PAPER)]),
           dict(type="bar", title="Full validation set (11,313 questions): ECE", ylabel="ECE (lower is better)", **bars(
               ["untrained Llama-3-8B", "paper", "ours: late-half, step 1,000", "ours: all-layer, step 500"],
               [("ECE", [.303, .0226, .031, .063], E)]), colors=[G, PAPER, E, P]),
           dict(type="bar", title="Full validation set (11,313 questions): AUROC", ylabel="AUROC (higher is better)", ymin=.5, **bars(
               ["untrained Llama-3-8B", "paper", "ours: late-half, step 1,000", "ours: all-layer, step 500"],
               [("AUROC", [.625, .859, .877, .867], E)]), colors=[G, PAPER, E, P],
               note="The axis starts at 0.5, the AUROC of a confidence that carries no information. One seed per run; the paper reports one number per method.")],
    "23": [dict(type="line", title="LoRA rank: dev Brier at step 500", xlabel="rank", xticks=[1, 2, 4, 8], ylabel="Brier (lower is better)", xlog=True, datasets=[
        line([1, 2, 4, 8], [R(f"23/r{r}/brier") for r in (1, 2, 4, 8)], "Brier", E)], note=SEED_NOTE + " 2 seeds. AUROC: 0.881, 0.887, 0.892, 0.889."),
           dict(type="line", title="LoRA rank: answer KL at step 500", xlabel="rank", xticks=[1, 2, 4, 8], ylabel="answer KL (nats)", xlog=True, ylog=True, datasets=[
               line([1, 2, 4, 8], [R(f"23/r{r}/kl") for r in (1, 2, 4, 8)], "answer KL", P)], note=SEED_NOTE)],
    "24": [dict(type="bar", title="Training-set Brier at the end (fit to the training data; lower is better)", ylabel="train Brier", ymin=.12, note=SEED_NOTE, **bars(
        ["rank 4 alone", "+ MLP 3e-5", "+ MLP 1e-4", "+ MLP 3e-4", "+ attn+MLP 3e-5", "+ attn+MLP 1e-4"],
        [("train Brier", [R(f"24/{k}") for k in ("rank 4 alone", "+ MLP 3e-5", "+ MLP 1e-4", "+ MLP 3e-4", "+ attn+MLP 3e-5", "+ attn+MLP 1e-4")], E)]))],
    "25": [dict(type="line", title="Dev Brier over 2,000 steps on 8,000 repeated questions", xlabel="training step", ylabel="dev Brier",
                datasets=[line(b["steps"][1:], [R(v) for v in b["brier"][1:]], arm, col) for (arm, b), col in zip(BANDS.items(), (E, B, P))] + [
                    line([2000], [R("25/avg/constant 3e-4")], "weights averaged, constant 3e-4", E),
                    line([2000], [R("25/avg/constant 1e-4")], "weights averaged, constant 1e-4", B)],
                note=SEED_NOTE + " 2 seeds. Single points at step 2,000 are the averaged weights (steps 1,000–2,000)."),
           dict(type="bar", title="Overfitting: training vs dev Brier at the end (exact-match labels)", ylabel="Brier", note=SEED_NOTE, **bars(
               ["500 steps, constant 3e-4", "2,000, constant 3e-4", "2,000, constant 1e-4", "2,000, cosine 3e-4"],
               [("training set", [R(f"25/traindev/{k}", 0) for k in ("500 steps", "constant 3e-4", "constant 1e-4", "cosine 3e-4")], E),
                ("dev", [R(f"25/traindev/{k}", 1) for k in ("500 steps", "constant 3e-4", "constant 1e-4", "cosine 3e-4")], P)]))],
    "26": [dict(type="bar", title="Cosine vs constant: dev AUROC", ylabel="AUROC (higher is better)", ymin=.8, note=SEED_NOTE, **bars(
        ["1 epoch, 3e-4", "1 epoch, 1e-4", "2 epochs, 3e-4", "2 epochs, 1e-4"],
        [("cosine", [R(f"26/cos/{e}/{lr}/auroc") for e, lr in ((1, "3e-4"), (1, "1e-4"), (2, "3e-4"), (2, "1e-4"))], P),
         ("constant", [R(f"26/const/{e}/{lr}/auroc") for e, lr in ((1, "3e-4"), (1, "1e-4"), (2, "3e-4"), (2, "1e-4"))], E)])),
           dict(type="bar", title="Cosine vs constant: dev Brier", ylabel="Brier (lower is better)", ymin=.1, **bars(
               ["1 epoch, 3e-4", "1 epoch, 1e-4", "2 epochs, 3e-4", "2 epochs, 1e-4"],
               [("cosine", [R(f"26/cos/{e}/{lr}/brier") for e, lr in ((1, "3e-4"), (1, "1e-4"), (2, "3e-4"), (2, "1e-4"))], P),
                ("constant", [R(f"26/const/{e}/{lr}/brier") for e, lr in ((1, "3e-4"), (1, "1e-4"), (2, "3e-4"), (2, "1e-4"))], E)]),
               note=SEED_NOTE + " Constant 3e-4: the rank sweep's runs; constant 1e-4 at 1 epoch: the mean of steps 200 and 300 of the "
                    "2,000-step runs (evaluated every 100 steps).")],
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
        m = re.match(r"\| \[(\d\d)\]\([^)]+\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| [^|]+ \| [^|]+ \| [^|]+ \| (.+) \|$", row)
        if m:
            one_liners[m.group(1)] = dict(model=m.group(3).strip(), swept=m.group(4).strip(), result=m.group(5).strip())
    intro_html = markdown(intro.split("\n", 1)[1])  # drop the Markdown title
    rows = []
    for num, title, stars, know, _ in sweeps:
        o = one_liners.get(num, {})
        rows.append(f"<tr><td class='num'><a href='#s{num}'>{num}</a></td><td><a href='#s{num}'>{inline(title)}</a>"
                    f"<div class='sub'>{inline(o.get('model', ''))} · {inline(o.get('swept', ''))}</div></td>"
                    f"<td class='stars'>{'★' * stars}<span class='dim'>{'★' * (3 - stars)}</span></td>"
                    f"<td><span class='know k-{know.replace('–', '-')}'>{html.escape(know)}</span></td>"
                    f"<td class='cost'>${COSTS[num]['usd']:.0f}<div class='sub'>{COSTS[num]['runs']} runs · {COSTS[num]['hours']:.1f} h</div></td>"
                    f"<td>{inline(o.get('result', ''))}</td></tr>")
    index_html = ("<div class='table'><table class='index'><thead><tr><th>#</th><th>sweep</th><th>motivation</th><th>knowledge</th>"
                  "<th>compute (est.)</th><th>result</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
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
            .replace("{{CHARTS}}", json.dumps(CHART_SPECS))
            .replace("{{TOTAL}}", f"about ${sum(c['usd'] for c in COSTS.values()):.0f} and {sum(c['hours'] for c in COSTS.values()):.0f} GPU-hours"))
    (SWEEPS / "index.html").write_text(page)
    print(f"{len(sweeps)} sweeps, {len(CHART_SPECS)} charts -> docs/sweeps/index.html ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
