"""Optional live tracking in Weights & Biases, alongside the jsonl logs (which stay the record).

Active only when WANDB_API_KEY is set and WANDB_MODE is not "disabled"; otherwise every call is a
no-op, so the trainers run unchanged on Modal and in tests. The W&B run id is derived from the run
directory's name, so a run resumed after preemption continues the same W&B run, and the snapshot
evaluation that runs after training adds its curve to it on a separate "eval/step" axis.
"""
import hashlib
import math
import os


class Tracker:
    def __init__(self, run_dir, config=None, job_type="train"):
        self.run = None
        if not os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "disabled":
            return
        import wandb
        name = os.path.basename(os.path.normpath(run_dir))
        self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "rewarding-doubt"), name=name,
                              id=hashlib.sha1(name.encode()).hexdigest()[:16], resume="allow", job_type=job_type,
                              config=config or {}, dir=run_dir)
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")

    def log(self, record, step=None):
        if self.run is None:
            return
        flat = {}
        for key, value in record.items():
            if isinstance(value, bool):
                flat[key] = int(value)
            elif isinstance(value, (int, float)) and math.isfinite(value):
                flat[key] = value
            elif key == "flags":
                flat["flag_count"] = len(value)
        self.run.log(flat, step=step)

    def alert(self, title, text):
        if self.run is not None:
            self.run.alert(title=title, text=text)

    def finish(self):
        if self.run is not None:
            self.run.finish()
