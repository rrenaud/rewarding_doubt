"""Estimated compute of each sweep in docs/sweeps: number of runs, serial GPU-hours (all runs one after another) and
dollars. Writes docs/sweeps/costs.json and the "**Compute (estimated):**" line of each report.

    python scripts/sweep_costs.py

Measured where the runs logged it, estimated where they did not:
- Modal search runs (runs/modal-hparam-*): training time from timing.jsonl's per-step clock (exact: step end times;
  PPO: step enter/exit times), plus about 3 minutes
  per evaluated snapshot (eval_step*.json) and 2.5 minutes of model loading per run.
- Modal fast-loop and minimal-trainer runs: the last "minutes" of train.log (training and dev evaluations), plus
  2.5 minutes of loading per run.
- RunPod runs: entry.log's attempt start to trainer exit, plus the post command (adapter evaluation), plus about
  2.5 minutes of pod provisioning (measured 1.5-3.5).
Prices: Modal L40S $1.95/h for the GPU plus about $0.35/h for CPU and memory ($2.30/h); RunPod at the pod's price
(promoted_from.json), or $0.74/h for the earlier RTX 4090 pods that did not record one.
"""
import datetime
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODAL = 2.30
RUNPOD_DEFAULT = 0.74
LOAD_H = 2.5 / 60


def configs(*names, folder="runs/minimal"):
    labels = []
    for n in names:
        labels += list(json.loads((ROOT / folder / n).read_text()))
    return labels


def trainer_hours(label, folder="runs/minimal"):
    """Last 'minutes' of a fast-loop or minimal-trainer train.log, plus loading; None if the run left no log."""
    path = ROOT / folder / label / "train.log"
    if not path.exists():
        return None
    minutes = [json.loads(l)["minutes"] for l in path.read_text().splitlines() if l.startswith('{"step"') and '"minutes"' in l]
    return (minutes[-1] / 60 if minutes else 0) + LOAD_H


def modal_runs(labels, folder="runs/minimal"):
    hours = [h for h in (trainer_hours(l, folder) for l in labels) if h is not None]
    return dict(runs=len(hours), hours=sum(hours), usd=sum(hours) * MODAL, where="Modal L40S")


def hparam_runs(stages):
    hours = []
    for stage in stages:
        for d in glob.glob(str(ROOT / f"runs/modal-hparam-{stage}-*/*/")):
            t = Path(d) / "timing.jsonl"
            if not t.exists():
                continue
            rows = [json.loads(l) for l in t.read_text().splitlines() if l.strip()]
            if "time" in rows[0]:  # exact_llama.py: step end times and the first step's length
                train = rows[-1]["time"] - rows[0]["time"] + rows[0]["seconds"]
            else:  # subset.py (PPO): enter / exit times of each step
                train = rows[-1]["exit"] - rows[0]["enter"]
            evals = len(list(Path(d).glob("eval_step*.json")))
            hours.append(train / 3600 + evals * 3 / 60 + LOAD_H)
    return hours


def stamps(entry_log):
    t = lambda l: datetime.datetime.strptime(l[:20], "%Y-%m-%dT%H:%M:%SZ")
    lines = [l for l in entry_log.splitlines() if l[:4].isdigit()]
    seconds, start = 0.0, None
    for l in lines:
        if "attempt start" in l or "post:" in l:
            start = t(l)
        elif any(k in l for k in ("trainer exited", "outcome:", "SIGTERM", "terminate requested")) and start:
            seconds += (t(l) - start).total_seconds()
            start = None
    if start and lines:  # an attempt still open at the end of the log (stopped by hand, or still running)
        seconds += max(0.0, (t(lines[-1]) - start).total_seconds())
    return seconds


_s3 = None


def entry_log(name):
    local = ROOT / "runs/runpod" / name / "entry.log"
    if local.exists():
        return local.read_text()
    global _s3
    from runpod_launch import VOLUME_ID
    from runpod_sync import client
    _s3 = _s3 or client()
    try:
        return _s3.get_object(Bucket=VOLUME_ID, Key=f"runs/{name}/entry.log")["Body"].read().decode()
    except Exception:
        return ""


def runpod_runs(patterns):
    names = sorted({Path(p).name for pat in patterns for p in glob.glob(str(ROOT / "runs/runpod" / pat))})
    hours, usd = 0.0, 0.0
    for n in names:
        meta = ROOT / "runs/runpod" / n / "promoted_from.json"
        price = (json.loads(meta.read_text()).get("cost_per_hr") if meta.exists() else None) or RUNPOD_DEFAULT
        h = stamps(entry_log(n)) / 3600 + LOAD_H
        hours += h
        usd += h * price
    return dict(runs=len(names), hours=hours, usd=usd, where="RunPod")


def add(*parts, note=""):
    out = dict(runs=sum(p["runs"] for p in parts), hours=sum(p["hours"] for p in parts), usd=sum(p["usd"] for p in parts),
               where=" + ".join(dict.fromkeys(p["where"] for p in parts)), note=note)
    return out


def main():
    h01 = hparam_runs(["round1", "stage1", "stage2", "stage3", "stage4", "final", "final_ppo", "final_ppo_f1", "final_exact_extra"])
    finals = hparam_runs(["final", "final_ppo", "final_ppo_f1", "final_exact_extra"])
    h02 = hparam_runs(["brier_sweep", "brier_sweep2", "brier_full"])
    est03 = 22 * st.fmean(finals)  # frozen_compare 10 runs + frozen_lr 12 runs, at the finals' mean run time
    paper_modal = modal_runs(configs("llama_paper_configs.json"))
    paper_evals = dict(runs=0, hours=9 * 0.08 + 4 * 0.75, usd=(9 * 0.08 + 4 * 0.75) * MODAL, where="Modal L40S")
    costs = {
        "01": dict(runs=len(h01), hours=sum(h01), usd=sum(h01) * MODAL, where="Modal L40S",
                   note="training from per-step clocks; evaluation estimated at 3 min per snapshot"),
        "02": dict(runs=len(h02), hours=sum(h02), usd=sum(h02) * MODAL, where="Modal L40S", note=""),
        "03": dict(runs=22, hours=est03, usd=est03 * MODAL, where="Modal L40S",
                   note="estimated: 22 runs at the final round's mean run time (their logs stayed on the volume)"),
        "04": add(runpod_runs(["klgrid-*"]), note="the 3,000-step comparison and seed check run elsewhere are not included"),
        "05": add(runpod_runs(["long-*", "long2-*", "long3-*"])),
        "06": dict(runs=18, hours=12.8 / MODAL, usd=12.8, where="Modal L40S", note="recorded spend"),
        "07": add(modal_runs(configs("first_configs.json", "speed_configs.json", "schedule_configs.json", "bucket_configs.json",
                                     folder="runs/fast"), "runs/fast")),
        "08": add(modal_runs(configs("muon_configs.json", "lora_optim_configs.json", "seed_sweep_configs.json", folder="runs/fast"), "runs/fast")),
        "09": add(modal_runs(configs("ablation_configs.json", "ablation_single_configs.json", "cut_check_configs.json",
                                     "timing_configs.json", folder="runs/fast"), "runs/fast")),
        "10": add(modal_runs(configs("o_depth_gatebias_configs.json", "resbias_configs.json", "attnbias_configs.json",
                                     folder="runs/fast"), "runs/fast")),
        "11": add(modal_runs(configs("answerkl_configs.json", folder="runs/fast"), "runs/fast"),
                  modal_runs(configs("answerkl_configs.json"))),
        "12": add(modal_runs(configs("hsearch_configs.json"))),
        "13": add(modal_runs(configs("online_configs.json"))),
        "14": add(modal_runs(configs("stock_configs.json"))),
        "15": add(runpod_runs(["online-bias-*", "online-vbias-t2-s1"])),
        "16": add(modal_runs(configs("recover_configs.json"))),
        "17": add(modal_runs(configs("floor_configs.json", "online_floor_configs.json")), runpod_runs(["online-vbias-t1.5-min0.1-s1"])),
        "18": add(modal_runs(configs("llama_configs.json", "llama_drift_configs.json", "llama_baseseed_configs.json"))),
        "19": add(modal_runs(configs("llama_lorafull_configs.json"))),
        "20": add(modal_runs(configs("llama_lorao_configs.json"))),
        "21": add(modal_runs(configs("llama_latelora_configs.json", "llama_accum_configs.json"))),
        "22": add(paper_modal, runpod_runs(["llama-paper-*-rp"]), paper_evals,
                  note="released evaluations estimated: 9 dev runs at 5 min, 4 full-set runs at 45 min"),
        "23": add(runpod_runs(["llama-lorafull16-r*-rp"]), note="includes the later per-example dump reruns"),
        "24": add(runpod_runs(["llama-lora16r4-*-rp"]), note="includes 8 runs that crashed at start on 5090s and their reruns"),
        "25": add(runpod_runs(["llama-r4-2k-*-rp"])),
        "26": add(runpod_runs(["llama-r4-ep*-cos*-rp"])),
    }
    for k, c in costs.items():
        c["hours"], c["usd"] = round(c["hours"], 2), round(c["usd"], 2)
    (ROOT / "docs/sweeps/costs.json").write_text(json.dumps(costs, indent=1) + "\n")
    for path in sorted((ROOT / "docs/sweeps").glob("[0-9][0-9]-*.md")):
        c = costs[path.name[:2]]
        line = (f"**Compute (estimated):** {c['runs']} runs, {c['hours']:.1f} GPU-hours if run one after another, about "
                f"${c['usd']:.0f} ({c['where']})" + (f"; {c['note']}." if c.get("note") else ".") + " See `scripts/sweep_costs.py`.")
        md = path.read_text()
        md = re.sub(r"\n\*\*Compute \(estimated\):\*\*[^\n]*\n", "\n", md)
        md = re.sub(r"(\*\*When / where:\*\*.*?\n)(\n)", lambda m: m.group(1) + "\n" + line + "\n" + m.group(2), md, count=1, flags=re.S)
        path.write_text(md)
    total = sum(c["usd"] for c in costs.values())
    for k, c in costs.items():
        print(f"{k}: {c['runs']:3d} runs {c['hours']:6.1f} h ${c['usd']:7.2f} {c['where']}")
    print(f"total ${total:.0f}, {sum(c['hours'] for c in costs.values()):.0f} GPU-hours")


if __name__ == "__main__":
    main()
