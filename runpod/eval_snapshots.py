"""Score every snapshot of a run on the validation split (POST_CMD of a long run).

    python /opt/runpod/eval_snapshots.py OUT_DIR IDS_JSON     # run from SingleAnswerSetting/

Each snapshot-step<N>/ gets the released evaluation (subset.py evaluate: eval_dev.json and
eval_dev_metrics.json, skipped if already there, so a rerun only does what is missing). The
curve goes to OUT_DIR/curve.json ({step: metrics}) and, when W&B is configured, to the run's
"eval/*" charts.
"""
import json
import subprocess
import sys
from pathlib import Path

from rewarding_doubt.tracking import Tracker


def main(out_dir, ids):
    out = Path(out_dir)
    curve = {}
    for snapshot in sorted(out.glob("snapshot-step*"), key=lambda p: int(p.name.split("step")[1])):
        step = int(snapshot.name.split("step")[1])
        metrics_path = snapshot / "eval_dev_metrics.json"
        if not metrics_path.exists():
            print(f"evaluating {snapshot.name}", flush=True)
            code = subprocess.run([sys.executable, "subset.py", "evaluate", ids, str(snapshot),
                                   str(snapshot / "eval_dev.json")]).returncode
            if code or not metrics_path.exists():
                print(f"evaluation of {snapshot.name} failed (exit {code})", flush=True)
                continue
        curve[step] = json.loads(metrics_path.read_text())
        print(step, {k: round(v, 4) for k, v in curve[step].items() if isinstance(v, float)}, flush=True)
    (out / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")
    tracker = Tracker(out_dir, job_type="train")
    for step, metrics in curve.items():
        tracker.log({"eval/step": step, **{f"eval/{k}": v for k, v in metrics.items()}})
    tracker.finish()
    return 0 if curve else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
