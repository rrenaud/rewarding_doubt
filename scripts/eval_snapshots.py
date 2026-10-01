"""Evaluate every saved snapshot of a training run on the held-out set, in parallel.

Usage: scripts/eval_snapshots.py RUN_DIR ARM [--eval-args '--prompt paper']
Writes RUN_DIR/ARM-snapshots/step-NNNNN/ and RUN_DIR/ARM-snapshots/curve.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import read_api_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("arm", help="training directory name inside the run, e.g. paper-ppo-s1")
    parser.add_argument("--eval-args", default="")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, TINKER_API_KEY=read_api_key(Path.home() / ".tinker_api_key"),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    checkpoint = json.loads((args.run / args.arm / "checkpoint.json").read_text())
    snapshots = {s["step"]: s["sampler_path"] for s in checkpoint["snapshots"]}
    snapshots.setdefault(checkpoint["steps"], checkpoint["sampler_path"])
    out = args.run / f"{args.arm}-snapshots"
    out.mkdir(exist_ok=True)

    def evaluate(step):
        target = out / f"step-{step:05d}"
        if not (target / "metrics.json").exists():
            with (out / f"step-{step:05d}.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "rewarding_doubt.cli", "evaluate",
                                "--data", str(args.run / "data/eval.jsonl"), "--checkpoint", snapshots[step],
                                "--output", str(target), *shlex.split(args.eval_args)],
                               cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((target / "metrics.json").read_text())
        print(f"step {step} done", flush=True)
        return dict(step=step, sampler_path=snapshots[step],
                    valid_mass=metrics["mean_valid_confidence_mass"],
                    invalid_rate=metrics["invalid_confidence_rate"],
                    **{readout: {k: metrics[readout][k] for k in ["ece", "brier", "nll", "auroc"]}
                       for readout in ["fractional", "argmax", "sampled"] if metrics[readout]})

    with ThreadPoolExecutor(args.parallel) as pool:
        curve = list(pool.map(evaluate, sorted(snapshots)))
    (out / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    for point in curve:
        f = point["fractional"]
        print(f"step {point['step']:4d}  ECE {f['ece']:.3f}  Brier {f['brier']:.3f}  NLL {f['nll']:.3f}  "
              f"AUROC {f['auroc']:.3f}  mass {point['valid_mass']:.3f}  invalid {point['invalid_rate']:.1%}")


if __name__ == "__main__":
    main()
