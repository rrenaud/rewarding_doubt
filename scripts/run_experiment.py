"""Run a full experiment: cache answers once, then train/evaluate every arm for each seed.

Credentials are read from ~/.tinker_api_key and passed only via subprocess environment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import threading

MODES = ["discrete-ppo", "discrete-exact", "fractional", "paper-ppo"]
READOUTS = ["fractional", "argmax", "sampled"]
METRICS = ["ece", "brier", "nll", "auroc"]


def read_api_key(path):
    text = path.read_text().strip()
    if not text:
        raise ValueError("Credential file is empty")
    # Accept a raw key or a single shell-style assignment, without executing it.
    if text.startswith(("export ", "TINKER_API_KEY=", "tinker_api_key=")):
        words = shlex.split(text)
        assignment = words[-1]
        name, sep, key = assignment.partition("=")
        if sep != "=" or name.lower() != "tinker_api_key":
            raise ValueError("Expected TINKER_API_KEY assignment in credential file")
        return key
    if len(text.splitlines()) != 1:
        raise ValueError("Expected a raw API key or one assignment")
    return text.strip("\"'")


def aggregate(summary, seeds, modes):
    """Mean and sample std across seeds for each mode/readout/metric."""
    table = {}
    for mode in modes:
        for readout in READOUTS:
            for metric in METRICS:
                values = [summary[f"{mode}-s{seed}-eval"][readout][metric] for seed in seeds
                          if summary[f"{mode}-s{seed}-eval"][readout] is not None]
                table.setdefault(mode, {}).setdefault(readout, {})[metric] = dict(
                    mean=statistics.fmean(values), std=statistics.stdev(values) if len(values) > 1 else 0.,
                    values=values)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pilot")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=0, help="0 trains for --epochs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=9, help="concurrent train+eval jobs")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES[:3])
    parser.add_argument("--data-from", type=Path,
                        help="copy data/train.jsonl and data/eval.jsonl from this earlier run instead of preparing")
    parser.add_argument("--skip-base", action="store_true", help="skip the base-model evaluation")
    parser.add_argument("--train-args", default="",
                        help="extra arguments for every train stage, e.g. '--format-weight 1 --save-every 40'")
    parser.add_argument("--eval-args", default="", help="extra arguments for every evaluate stage, e.g. '--prompt paper'")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["TINKER_API_KEY"] = read_api_key(Path.home() / ".tinker_api_key")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    run = root / "runs" / (f"{args.name}-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True)
    (run / "data").mkdir()
    print(f"Run: {run}", flush=True)
    status = {"run": str(run), "args": vars(args), "completed": [], "running": [], "state": "running"}
    lock = threading.Lock()

    def save_status():
        (run / "status.json").write_text(json.dumps(status, indent=2, default=str) + "\n")

    def stage(name, cli_args):
        with lock:
            status["running"].append(name)
            save_status()
        print(f"Starting {name}", flush=True)
        with (run / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "rewarding_doubt.cli", *cli_args],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            status["running"].remove(name)
            if result.returncode:
                status.setdefault("failed", []).append(name)
                save_status()
                raise RuntimeError(f"Stage {name} failed; inspect {name}.log")
            status["completed"].append(name)
            save_status()
        print(f"Completed {name}", flush=True)

    if args.data_from:
        # Reusing the cached answers keeps every label identical, so runs compare directly.
        for name in ["train", "eval"]:
            shutil.copyfile(args.data_from / "data" / f"{name}.jsonl", run / "data" / f"{name}.jsonl")
    else:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda job: stage(*job), [
                (f"prepare-{name}", ["prepare", "--split", split, "--limit", str(limit),
                                     "--output", str(run / "data" / f"{name}.jsonl")])
                for split, name, limit in [("train", "train", args.train_limit),
                                           ("validation", "eval", args.eval_limit)]]))

    evaluation = ["evaluate", "--data", str(run / "data/eval.jsonl"), *shlex.split(args.eval_args)]

    def arm(job):
        if job == "base":
            stage("base-eval", [*evaluation, "--output", str(run / "base-eval")])
            return
        mode, seed = job
        name = f"{mode}-s{seed}"
        stage(name, ["train", "--data", str(run / "data/train.jsonl"), "--mode", mode,
                     "--seed", str(seed), "--epochs", str(args.epochs),
                     "--max-steps", str(args.max_steps), *shlex.split(args.train_args),
                     "--output", str(run / name)])
        checkpoint = json.loads((run / name / "checkpoint.json").read_text())
        stage(f"{name}-eval", [*evaluation, "--checkpoint", checkpoint["sampler_path"],
                               "--output", str(run / f"{name}-eval")])

    jobs = ([] if args.skip_base else ["base"]) + [(mode, seed) for seed in args.seeds
                                                   for mode in args.modes]
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [pool.submit(arm, job) for job in jobs]
        errors = [f.exception() for f in futures if f.exception()]
    if errors:
        status["state"] = "failed"
        save_status()
        raise errors[0]

    evals = ([] if args.skip_base else ["base-eval"]) + [f"{mode}-s{seed}-eval" for seed in args.seeds
                                                         for mode in args.modes]
    summary = {name: json.loads((run / name / "metrics.json").read_text()) for name in evals}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run / "aggregate.json").write_text(json.dumps(aggregate(summary, args.seeds, args.modes), indent=2) + "\n")
    status.update(state="complete")
    save_status()
    print(f"Finished: {run / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
