"""Per-answer training log: OUT_DIR/generations.jsonl, plus OUT_DIR/run_meta.json.

One line per generated answer at every training step, so runs can be reused as data (e.g. offline
or supervised confidence training) and compared across experiments. Each line carries:

    expt         experiment id (env RD_EXPT, set by the launcher; "adhoc" otherwise): groups the
                 runs launched together for one comparison, e.g. "long2"
    run          run name (the run directory's name), e.g. "long2-exact-1e5-s1"
    trainer      "exact" | "ppo"
    step         training step; question_id (TriviaQA id), answer (parsed), correct (F1 > 0.5;
                 null when the answer could not be parsed), confidence (sampled level 0-10 or null)
    exact only:  pi (11 probabilities, renormalized over the levels, 4 d.p.), mass (total
                 probability on the 11 levels), stop_prob, loss
    ppo only:    reward (Train.py's scalar reward)

run_meta.json holds what is constant for the run: expt, run, trainer, arguments, image git sha,
start time. A resumed run truncates the log to its checkpoint step and continues it.
"""
import json
import os
import time

from rewarding_doubt.checkpoint import truncate_jsonl


class GenerationLog:
    def __init__(self, out_dir, trainer, args, resume_step=None):
        self.expt = os.environ.get("RD_EXPT", "adhoc")
        self.run = os.path.basename(os.path.normpath(out_dir))
        self.trainer = trainer
        path = os.path.join(out_dir, "generations.jsonl")
        if resume_step is not None:
            truncate_jsonl(path, resume_step)
        self.file = open(path, "a" if resume_step is not None else "w")
        meta_path = os.path.join(out_dir, "run_meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        meta.update(expt=self.expt, run=self.run, trainer=trainer, args=args,
                    image_git_sha=os.environ.get("IMAGE_GIT_SHA"))
        meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        meta.setdefault("resumed_at_steps", [])
        if resume_step is not None:
            meta["resumed_at_steps"].append(resume_step)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1, default=str)

    def write(self, step, rows):
        for row in rows:
            self.file.write(json.dumps(dict(expt=self.expt, run=self.run, trainer=self.trainer, step=step, **row)) + "\n")
        self.file.flush()
