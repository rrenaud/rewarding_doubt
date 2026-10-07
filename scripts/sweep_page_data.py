"""Per-seed values behind the sweeps page's charts: docs/sweeps/page_data.json.

    python scripts/sweep_page_data.py

Reads the run logs (runs/fast, runs/minimal, runs/hparam-search-*, runs/runpod) and, for RunPod runs whose
train.log or examples.jsonl is not local, fetches them from the network volume (train.log is saved next to the
run's record; examples.jsonl is only summarized). Every value is a list over seeds, so the page can show the spread.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SEARCH = ROOT / "runs/hparam-search-20261002T044831Z"
_s3 = None


def s3():
    global _s3
    if _s3 is None:
        from runpod_sync import client
        _s3 = client()
    return _s3


def runpod_file(name, key):
    from runpod_launch import VOLUME_ID
    return s3().get_object(Bucket=VOLUME_ID, Key=f"runs/{name}/{key}")["Body"].read().decode()


def log_rows(label):
    """Evaluation lines of a run's train.log: runs/minimal, runs/fast or runs/runpod (fetched if needed)."""
    for folder in ("runs/minimal", "runs/fast", "runs/runpod"):
        path = ROOT / folder / label / "train.log"
        if path.exists():
            break
    else:
        path = ROOT / "runs/runpod" / label / "train.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(runpod_file(label, "train.log"))
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.startswith('{"step"')]
    events = [json.loads(l) for l in path.read_text().splitlines() if l.startswith('{"event"')]
    return {r["step"]: r for r in rows}, events


def at(label, step, key):
    rows, _ = log_rows(label)
    return rows[step][key]


def last_regen(label):
    rows, _ = log_rows(label)
    return [r["regen_accuracy"] for s, r in sorted(rows.items()) if "regen_accuracy" in r][-1]


def seeds(arm, n=2, first=1):
    return [f"{arm}-s{s}" for s in range(first, first + n)]


def per_seed(arm, fn, n=2, first=1):
    return [round(fn(l), 5) for l in seeds(arm, n, first)]


def train_dev_brier(label):
    text = runpod_file(label, "examples.jsonl")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    g = lambda split: st.fmean(r["fitted"]["brier"] for r in rows if r["split"] == split)
    return round(g("train"), 5), round(g("dev"), 5)


def search_curve(stage, labels, test=False, metric="brier"):
    c = json.loads((SEARCH / stage / ("curve_test.json" if test else "curve.json")).read_text())
    out = []
    for l in labels:
        v = c[l]
        m = v if metric in v else v[max((k for k in v if str(k).isdigit()), key=int)]
        out.append(round(m[metric], 5))
    return out


def main():
    data = json.loads((ROOT / "docs/sweeps/page_data.json").read_text())
    S = {}
    # 01: final round, test
    for arm, stage, labels in (("exact", "final", seeds("exact-final", 3, 4)), ("ppo", "final_ppo", seeds("ppo-final", 3, 4)),
                               ("ppo_f1", "final_ppo_f1", seeds("ppo-f1", 5, 4))):
        for metric in ("ece", "auroc"):
            S[f"01/{arm}/{metric}"] = search_curve(stage, labels, test=True, metric=metric)
    # 02: Brier mix, dev per seed (sweeps 1 and 2 pooled), full scale test
    c1 = json.loads((SEARCH / "brier_sweep/curve.json").read_text())
    c2 = json.loads((SEARCH / "brier_sweep2/curve.json").read_text())
    for m in ("0.00", "0.25", "0.30", "0.40", "0.50", "0.60", "0.75", "1.00"):
        vals = []
        for c in (c1, c2):
            for l, v in c.items():
                if l.startswith(f"exact-brier{m}-s"):
                    last = v[max((k for k in v if str(k).isdigit()), key=int)]
                    vals.append(round(last["brier"], 5))
        S[f"02/dev/{m}"] = vals
    for m in ("0.00", "0.50"):
        S[f"02/test/{m}"] = search_curve("brier_full", seeds(f"exact-brier{m}", 3, 6), test=True)
    # 03: frozen answers, dev Brier by lr
    for lr in ("2.0e-05", "8.0e-05", "1.6e-04", "3.2e-04"):
        S[f"03/exact/{lr}"] = search_curve("frozen_lr", seeds(f"exact-frozen-lr{lr}", 2, 4))
    for lr in ("4.5e-05", "9.0e-05"):
        S[f"03/ppo/{lr}"] = search_curve("frozen_lr", seeds(f"ppo-frozen-lr{lr}", 2, 4))
    S["03/exact/tuned"] = search_curve("frozen_compare", seeds("exact-frozen", 5, 4))
    S["03/ppo/tuned"] = search_curve("frozen_compare", seeds("ppo-frozen", 5, 4))
    # 08: optimizers, 3 seeds at step 300
    for arm in ("seeds-adam-lr3e-4", "seeds-adam-lr1e-3", "seeds-polora-lr3e-3", "seeds-scaled-adamw-lr3e-4", "seeds-scaled-adamw-lr1e-3"):
        S[f"08/{arm}"] = per_seed(arm, lambda l: at(l, 300, "brier_expected"), 3)
    # 09 / 10: fast-loop ablations, Brier at step 300 (2 seeds; baseline 3)
    for arm in ("ablate-mlp", "ablate-attn", "ablate-o-down", "ablate-qv", "ablate-gate", "ablate-up", "ablate-down", "ablate-o", "ablate-q",
                "ablate-v", "ablate-k", "ablate-layers0-17", "ablate-layers18-35", "ablate-layers27-35", "ablate-layers32-35",
                "ablate-o-layers18-35", "ablate-o-layers27-35", "gatebias-lr3e-3", "attnbias18-35", "resbias18-35-lr1e-2",
                "attnmlpbias18-35", "resbias27-35-lr1e-2", "attnbias27-35", "attnmlpbias27-35"):
        S[f"fast/{arm}"] = per_seed(arm, lambda l: at(l, 300, "brier_expected"))
    S["fast/baseline"] = S["08/seeds-adam-lr3e-4"]
    # 11: answer-KL weight (minimal trainer), step 300
    for arm in ("akl-attnbias-b0", "akl-attnbias-b0.1", "akl-attnbias-b1", "akl-attnbias-b10", "akl-lora-b0", "akl-lora-b1"):
        S[f"11/{arm}/regen"] = per_seed(arm, lambda l: at(l, 300, "regen_accuracy"))
        S[f"11/{arm}/brier"] = per_seed(arm, lambda l: at(l, 300, "brier_expected"))
    # 16: recovery, regenerated accuracy at step 300
    for w in ("0.3", "1", "3", "10", "30"):
        S[f"16/{w}"] = per_seed(f"recover-lr1e-2-w{w}", lambda l: at(l, 300, "regen_accuracy"))
    # 17: weight floor: Brier mean of steps 500-1,000, answer KL at 1,000
    for t in ("1.5", "2"):
        for f in ("0.001", "0.1", "0.3", "1", "3"):
            arm = f"floor-t{t}-min{f}"
            S[f"17/{t}/{f}/brier"] = per_seed(arm, lambda l: st.fmean(r["brier_expected"] for s, r in log_rows(l)[0].items() if s >= 500))
            S[f"17/{t}/{f}/kl"] = per_seed(arm, lambda l: at(l, 1000, "answer_kl"))
    # 18: Llama arms, per seed at step 500: answer KL and the last regenerated accuracy
    llama = {"o_proj bias 3e-3": "llama-obias-lr3e-3", "o_proj bias 1e-2": "llama-obias-lr1e-2", "hooked bias 1e-2": "llama-hookbias-lr1e-2",
             "norm gains 3e-3": "llama-norm-lr3e-3", "o_proj bias 3e-4, unpenalized": "llama-drift-obias-lr3e-4",
             "o_proj bias 1e-3, unpenalized": "llama-drift-obias-lr1e-3", "norm gains 3e-3, unpenalized": "llama-drift-norm-lr3e-3",
             "norm gains 1e-2, unpenalized": "llama-drift-norm-lr1e-2", "o-LoRA 16-31 3e-4": "llama-lorao-lr3e-4-w1",
             "o-LoRA 16-31 1e-3": "llama-lorao16-lr1e-3-w1",
             "full LoRA, W=1": "llama-lorafull-lr3e-4-w1", "full LoRA, released schedule": "llama-lorafull-released-lr1e-5-w0",
             "LoRA 8-31": "llama-lorafull8-lr3e-4-w1", "LoRA 16-31": "llama-lorafull16-lr3e-4-w1", "LoRA 24-31": "llama-lorafull24-lr3e-4-w1",
             "LoRA 16-31 rank 1": "llama-lorafull16-r1-lr3e-4-w1-{s}-rp", "LoRA 16-31 rank 2": "llama-lorafull16-r2-lr3e-4-w1-{s}-rp",
             "LoRA 16-31 rank 4": "llama-lorafull16-r4-lr3e-4-w1-{s}-rp"}
    small = {k for k in llama if not k.startswith(("full", "LoRA"))}
    pts = []
    for name, arm in llama.items():
        for s in (1, 2):
            label = arm.format(s=f"s{s}") if "{s}" in arm else f"{arm}-s{s}"
            pts.append(dict(name=name, seed=s, small=name in small, kl=round(at(label, 500, "answer_kl"), 5), acc=round(last_regen(label), 5)))
    S["18/points"] = pts
    # 20, 21, 23: Llama LoRA variants, Brier (and answer KL) at step 500
    for arm in ("llama-lorao-lr3e-4-w1", "llama-lorao16-lr1e-3-w1", "llama-lorao16-lr3e-3-w1", "llama-loraoall-lr1e-3-w1", "llama-loraoall-lr1e-3-w0",
                "llama-lorafull-lr3e-4-w1", "llama-lorafull8-lr3e-4-w1", "llama-lorafull16-lr3e-4-w1", "llama-lorafull24-lr3e-4-w1"):
        S[f"llama/{arm}/brier"] = per_seed(arm, lambda l: at(l, 500, "brier_expected"))
        S[f"llama/{arm}/kl"] = per_seed(arm, lambda l: at(l, 500, "answer_kl"))
    for r in (1, 2, 4):
        labels = [f"llama-lorafull16-r{r}-lr3e-4-w1-s{s}-rp" for s in (1, 2)]
        S[f"23/r{r}/brier"] = [round(at(l, 500, "brier_expected"), 5) for l in labels]
        S[f"23/r{r}/kl"] = [round(at(l, 500, "answer_kl"), 5) for l in labels]
    S["23/r8/brier"], S["23/r8/kl"] = S["llama/llama-lorafull16-lr3e-4-w1/brier"], S["llama/llama-lorafull16-lr3e-4-w1/kl"]
    # 24: residual biases, training-set Brier per seed (per-example dumps)
    for key, arm in (("rank 4 alone", "llama-lorafull16-r4-lr3e-4-w1"), ("+ MLP 3e-5", "llama-lora16r4-mlp-blr3e-5"), ("+ MLP 1e-4", "llama-lora16r4-mlp-blr1e-4"),
                     ("+ MLP 3e-4", "llama-lora16r4-mlp-blr3e-4"), ("+ attn+MLP 3e-5", "llama-lora16r4-attnmlp-blr3e-5"),
                     ("+ attn+MLP 1e-4", "llama-lora16r4-attnmlp-blr1e-4")):
        S[f"24/{key}"] = [train_dev_brier(f"{arm}-s{s}-rp")[0] for s in (1, 2)]
    # 25: 2,000-step curves (min and max over seeds per step), averaged weights, training vs dev Brier
    bands = {}
    for name, key in (("constant 3e-4", "const3e-4"), ("constant 1e-4", "const1e-4"), ("cosine 3e-4", "cos3e-4")):
        runs = [log_rows(f"llama-r4-2k-{key}-s{s}-rp") for s in (1, 2)]
        steps = sorted(set(runs[0][0]) & set(runs[1][0]))
        bands[name] = dict(steps=steps, brier=[[round(r[0][k]["brier_expected"], 5) for r in runs] for k in steps])
        S[f"25/avg/{name}"] = [round(next(e for e in r[1] if e.get("event") == "weight_avg")["brier_expected"], 5) for r in runs]
        S[f"25/traindev/{name}"] = [train_dev_brier(f"llama-r4-2k-{key}-s{s}-rp") for s in (1, 2)]
    S["25/traindev/500 steps"] = [train_dev_brier(f"llama-lorafull16-r4-lr3e-4-w1-s{s}-rp") for s in (1, 2)]
    data["lr_schedule_bands"] = bands
    # 26: cosine gate vs constant (constant 3e-4: the rank-sweep runs; constant 1e-4: the 2,000-step runs)
    r4 = [f"llama-lorafull16-r4-lr3e-4-w1-s{s}-rp" for s in (1, 2)]
    c1 = [f"llama-r4-2k-const1e-4-s{s}-rp" for s in (1, 2)]
    for metric, key in (("auroc", "auroc_expected"), ("brier", "brier_expected")):
        S[f"26/cos/1/3e-4/{metric}"] = [round(at(f"llama-r4-ep1-cos3e-4-s{s}-rp", 250, key), 5) for s in (1, 2)]
        S[f"26/cos/1/1e-4/{metric}"] = [round(at(f"llama-r4-ep1-cos1e-4-s{s}-rp", 250, key), 5) for s in (1, 2)]
        S[f"26/cos/2/3e-4/{metric}"] = [round(at(f"llama-r4-ep2-cos3e-4-s{s}-rp", 500, key), 5) for s in (1, 2)]
        S[f"26/cos/2/1e-4/{metric}"] = [round(at(f"llama-r4-ep2-cos1e-4-s{s}-rp", 500, key), 5) for s in (1, 2)]
        S[f"26/const/1/3e-4/{metric}"] = [round(at(l, 250, key), 5) for l in r4]
        S[f"26/const/1/1e-4/{metric}"] = [round((at(l, 200, key) + at(l, 300, key)) / 2, 5) for l in c1]
        S[f"26/const/2/3e-4/{metric}"] = [round(at(l, 500, key), 5) for l in r4]
        S[f"26/const/2/1e-4/{metric}"] = [round(at(l, 500, key), 5) for l in c1]
    data["seeds"] = S
    (ROOT / "docs/sweeps/page_data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(S)} seeded series -> docs/sweeps/page_data.json")


if __name__ == "__main__":
    main()
