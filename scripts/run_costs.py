"""Throughput per dollar of minimal_trainer.py runs on RunPod and Modal, with one-off costs separated from per-question
costs.

    python scripts/run_costs.py [--runpod NAME ...] [--modal LABEL ...] [--json OUT]
    (no arguments: every promoted RunPod run under runs/runpod/*/promoted_from.json)

Per run, the billed time is split into:
  provision  pod created -> trainer attempt starts (RunPod: scheduling, image pull, boot). RunPod's billing start
             within this window is not documented here, so it is counted as billed (an upper bound).
  load       attempt start -> first training step (model, cache and reference loading; process start up).
  train      training updates only (online runs: answer generation included), from the per-step seconds in
             metrics.jsonl: steps right after a dev evaluation or a checkpoint/adapter save are counted at the
             median step time and the excess goes to eval/save. Runs whose trainer logs eval_seconds and
             save_seconds use those directly.
  eval+save  dev evaluation during training plus checkpoint and adapter saving (shown together: from per-step
             records they cannot be separated when evaluations and saves fall on the same steps; the trainer's
             done event gives each exactly, in the --json output).
  post       trainer exit -> outcome (the post command: e.g. the released evaluation of each adapter).
  teardown   outcome -> pod termination requested.
One-off cost = provision + load (+ teardown); per-question cost = train / (steps x 32 questions per update).

Prices: RunPod, the pod's costPerHr at launch (GPU, CPU and RAM included). Modal, its price list (October 2026):
GPU per second by type, plus CPU $0.0000131/core/s and memory $0.00000222/GiB/s billed on usage; the memory and
CPU of a Llama trainer are assumed at 32 GiB and 2 cores (~$0.35/h) since Modal does not report them per run.
Modal's container boot before the function's first line is not visible here and not counted.
"""
import argparse
import datetime
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MODAL_GPU_PER_S = {"L40S": 0.000542, "A100-80GB": 0.000694, "A100": 0.000583, "H100": 0.001097, "L4": 0.000222,
                   "A10G": 0.000306, "T4": 0.000164, "B200": 0.001736}
MODAL_HOST_PER_S = 2 * 0.0000131 + 32 * 0.00000222  # assumed 2 cores and 32 GiB
QUESTIONS_PER_STEP = 32


def ts(text):
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def flag(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def train_split(command, metrics, events):
    """(train seconds, eval seconds, save seconds, updates) from the trainer's per-step records."""
    rows = sorted({r["step"]: r for r in metrics}.values(), key=lambda r: r["step"])
    if len(rows) < 3:
        return None
    done = next((e for e in events if e.get("event") == "done"), None)
    if done:
        return done["train_seconds"], done["eval_seconds"], done["save_seconds"], len(rows)
    eval_every, ckpt = int(flag(command, "--eval-every", 50)), int(flag(command, "--checkpoint-every", 0) or 0)
    adapters = int(flag(command, "--save-adapter-every", 0) or 0)
    diffs = {b["step"]: b["seconds"] - a["seconds"] for a, b in zip(rows, rows[1:])}
    after_eval = {s for s in diffs if (s - 1) % eval_every == 0}
    after_save = {s for s in diffs if (ckpt and (s - 1) % ckpt == 0) or (adapters and (s - 1) % adapters == 0)}
    plain = [d for s, d in diffs.items() if s not in after_eval | after_save]
    median = statistics.median(plain)
    excess = lambda ss: sum(max(0.0, diffs[s] - median) for s in ss)
    if after_eval & after_save:  # coinciding steps (e.g. regenerated accuracy and a checkpoint every 100): not separable
        return rows[-1]["seconds"] - excess(after_eval | after_save), excess(after_eval | after_save), None, len(rows)
    eval_s, save_s = excess(after_eval), excess(after_save)
    # (eval and save seconds; save None when they could not be separated.) rows[-1]["seconds"] ends at the last update: the final evaluation comes after it (the done event, or the pod
    # timeline's load_s, covers it)
    return rows[-1]["seconds"] - eval_s - save_s, eval_s, save_s, len(rows)


def parse_log(text):
    events, lines = [], text.splitlines()
    for l in lines:
        if l.startswith('{"event"'):
            events.append(json.loads(l))
    return events


def runpod_files(name):
    from runpod_launch import VOLUME_ID
    from runpod_sync import client
    s3 = client()
    out = {}
    for k in ("entry.log", "train.log", "metrics.jsonl"):
        try:
            out[k] = s3.get_object(Bucket=VOLUME_ID, Key=f"runs/{name}/{k}")["Body"].read().decode()
        except Exception:
            out[k] = ""
    return out


def runpod_run(name):
    record = json.loads((ROOT / "runs/runpod" / name / "promoted_from.json").read_text())
    files = runpod_files(name)
    entry = files["entry.log"].splitlines()
    stamp = lambda key: [ts(l[:20]) for l in entry if key in l and l[:4].isdigit()]
    created = ts(record["pod_created"]) if record.get("pod_created") else None
    starts, exits, outcomes, terms = stamp("attempt start"), stamp("trainer exited"), stamp("outcome:"), stamp("pod terminate requested")
    now = datetime.datetime.now(datetime.timezone.utc)
    events = parse_log(files["train.log"])
    metrics = [json.loads(l) for l in files["metrics.jsonl"].splitlines() if l.strip()]
    split = train_split(record["runpod_command"], metrics, events)
    rate = record.get("cost_per_hr") / 3600
    row = dict(platform="runpod", run=name, gpu=(record.get("gpu_type") or "?").replace("NVIDIA ", "").replace("GeForce ", ""),
               usd_per_hr=record.get("cost_per_hr"), accumulate=int(flag(record["runpod_command"], "--accumulate", 1)),
               finished=bool(terms))
    end = terms[-1] if terms else now
    row["billed_s"] = (end - created).total_seconds() if created else None
    row["provision_s"] = (starts[0] - created).total_seconds() if created and starts else None
    trainer_wall = sum((e - s).total_seconds() for s, e in zip(starts, exits)) if exits else None
    start_event = next((e for e in events if e.get("event") == "train_start"), None)
    if split:
        row.update(train_s=split[0], eval_s=split[1], save_s=split[2], updates=split[3])
        if start_event and starts:
            row["load_s"] = (ts(start_event["time"]) - starts[0]).total_seconds()
        elif trainer_wall is not None:  # trainer wall minus its training clock: loading plus the final evaluation
            row["load_s"] = trainer_wall - metrics[-1]["seconds"]
            row["load_note"] = "includes the final dev evaluation and curve writing (no train_start event)"
    if exits and outcomes:
        row["post_s"] = (outcomes[-1] - exits[-1]).total_seconds()
    if outcomes and terms:
        row["teardown_s"] = (terms[-1] - outcomes[-1]).total_seconds()
    return costs(row, rate)


def modal_record(label):
    """launch.json, or for runs launched before launch records existed, the command from the config file that
    names the label (the run's files are then the local copies minimal_train fetched)."""
    path = ROOT / "runs/minimal" / label / "launch.json"
    if path.exists():
        return json.loads(path.read_text())
    for cfg in sorted((ROOT / "runs/minimal").glob("*configs*.json")):
        configs = json.loads(cfg.read_text())
        if label in configs:
            return dict(label=label, modal_run=None, gpu="L40S", command=["minimal_trainer.py", "train", "CACHE", "OUT_DIR", *configs[label]])
    raise SystemExit(f"{label}: no launch.json and no config naming it")


def modal_run(label):
    record = modal_record(label)
    if record["modal_run"] is None:
        local = ROOT / "runs/minimal" / label
        texts = {k: (local / k).read_text() if (local / k).exists() else "" for k in ("train.log", "metrics.jsonl")}
    else:
        texts = modal_files(record, label)
    return modal_costs(record, label, texts)


def modal_files(record, label):
    with tempfile.TemporaryDirectory() as tmp:
        texts = {}
        for k in ("train.log", "metrics.jsonl"):
            subprocess.run(["modal", "volume", "get", "--force", "rewarding-doubt-repro", f"outputs/{record['modal_run']}/{label}/{k}",
                            f"{tmp}/{k}"], capture_output=True)
            texts[k] = Path(f"{tmp}/{k}").read_text() if Path(f"{tmp}/{k}").exists() else ""
    return texts


def modal_costs(record, label, texts):
    attempts = [ts(l.split()[2]) for l in texts["train.log"].splitlines() if l.startswith("=== attempt")]
    events = parse_log(texts["train.log"])
    metrics = [json.loads(l) for l in texts["metrics.jsonl"].splitlines() if l.strip()]
    split = train_split(record["command"], metrics, events)
    gpu = record.get("gpu", "L40S")
    rate = MODAL_GPU_PER_S.get(gpu, MODAL_GPU_PER_S["L40S"]) + MODAL_HOST_PER_S
    row = dict(platform="modal", run=label, gpu=gpu, usd_per_hr=round(rate * 3600, 3), accumulate=int(flag(record["command"], "--accumulate", 1)),
               finished=any(e.get("event") == "done" for e in events))
    start_event = next((e for e in events if e.get("event") == "train_start"), None)
    done = next((e for e in events if e.get("event") == "done"), None)
    if split:
        row.update(train_s=split[0], eval_s=split[1], save_s=split[2], updates=split[3])
    if start_event and attempts:
        row["provision_s"] = 0.0  # not visible: Modal's container boot happens before the attempt line
        row["load_s"] = (ts(start_event["time"]) - attempts[0]).total_seconds()
    if done and attempts:
        row["billed_s"] = (ts(done["time"]) - attempts[0]).total_seconds()
    return costs(row, rate)


def costs(row, rate):
    if row.get("train_s") and row.get("updates"):
        questions = row["updates"] * QUESTIONS_PER_STEP
        row["s_per_step"] = row["train_s"] / row["updates"]
        row["usd_per_1k_questions"] = row["train_s"] * rate / questions * 1000
        row["questions_per_usd"] = questions / (row["train_s"] * rate)
        row["usd_train"] = row["train_s"] * rate
        row["eval_save_s"] = row["eval_s"] + (row["save_s"] or 0)
        row["usd_eval_save"] = row["eval_save_s"] * rate
    one_off = [row.get(k) for k in ("provision_s", "load_s", "teardown_s")]
    if any(v is not None for v in one_off):
        row["one_off_s"] = sum(v for v in one_off if v is not None)
        row["usd_one_off"] = row["one_off_s"] * rate
    if row.get("post_s") is not None:
        row["usd_post"] = row["post_s"] * rate
    if row.get("billed_s") is not None:
        row["usd_total"] = row["billed_s"] * rate
    return row


def show(rows):
    cols = [("run", "{}", 44), ("gpu", "{}", 9), ("accumulate", "{}", 3), ("usd_per_hr", "{:.2f}", 5),
            ("updates", "{}", 5), ("s_per_step", "{:.2f}", 5), ("usd_per_1k_questions", "{:.4f}", 7), ("questions_per_usd", "{:,.0f}", 8),
            ("provision_s", "{:.0f}", 5), ("load_s", "{:.0f}", 5), ("eval_save_s", "{:.0f}", 9),
            ("post_s", "{:.0f}", 5), ("usd_one_off", "{:.3f}", 6), ("usd_total", "{:.2f}", 6), ("finished", "{}", 5)]
    heads = {"accumulate": "acc", "usd_per_hr": "$/h", "s_per_step": "s/upd", "usd_per_1k_questions": "$/1kQ",
             "questions_per_usd": "Q/$", "provision_s": "prov", "load_s": "load", "eval_save_s": "eval+save",
             "post_s": "post", "usd_one_off": "$1-off", "usd_total": "$total", "finished": "done"}
    print("  ".join(heads.get(c, c).rjust(w) if c != "run" else c.ljust(w) for c, _, w in cols))
    for r in rows:
        cells = []
        for c, f, w in cols:
            v = r.get(c)
            text = "-" if v is None else f.format(v)
            cells.append(text.ljust(w) if c == "run" else text.rjust(w))
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runpod", nargs="*", default=None)
    parser.add_argument("--modal", nargs="*", default=[])
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    names = args.runpod if args.runpod is not None else (
        [] if args.modal else sorted(p.parent.name for p in (ROOT / "runs/runpod").glob("*/promoted_from.json")))
    rows = [runpod_run(n) for n in names] + [modal_run(l) for l in args.modal]
    show(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
