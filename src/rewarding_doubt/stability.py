"""Training-stability monitor shared by the exact trainer and the PPO wrapper.

Each training step reports the batch's sampled confidences (None for a malformed response), the
answers' correctness, and any scalars worth watching (loss, gradient norm, entropy, ...). The
monitor keeps a rolling window of steps and raises flags for the failure modes seen so far:

* collapsed: after warm-up, almost every confidence in the window is the same level (the exact
  arm's seed 8 locked on 2 for every question; once pi is one-hot its gradient vanishes, so the
  run never recovers). The base model itself says 10 nearly always, hence the warm-up.
* format_broken: many malformed responses in the window (high-learning-rate search runs).
* answers_degraded: rolling answer accuracy fell well below its first full window. The adapter
  touches every token, so confidence training can damage the answers themselves.
* nonfinite: a reported scalar is NaN or infinite.

A flag is reported once, as an event, when it first rises; `record["flags"]` lists the flags
active at that step.
"""
import math
from collections import Counter, deque

FLAGS = ("collapsed", "format_broken", "answers_degraded", "nonfinite")


class StabilityMonitor:
    def __init__(self, window=16, warmup=32, min_samples=64, collapse_share=0.98, format_limit=0.2,
                 accuracy_drop=0.15):
        self.window, self.warmup, self.min_samples = window, warmup, min_samples
        self.collapse_share, self.format_limit, self.accuracy_drop = collapse_share, format_limit, accuracy_drop
        self.steps = deque(maxlen=window)  # (confidences, correct) per step
        self.baseline_accuracy = None
        self.raised = set()

    def update(self, step, confidences, correct=(), **scalars):
        """Add one step; return (record, new_events)."""
        confidences, correct = list(confidences), [float(c) for c in correct]
        self.steps.append((confidences, correct))
        valid = [c for c in confidences if c is not None]
        record = dict(step=step, batch_confidence_mean=_mean(valid), batch_confidence_std=_std(valid),
                      batch_invalid_rate=1 - len(valid) / len(confidences) if confidences else None,
                      batch_accuracy=_mean(correct))
        window_conf = [c for cs, _ in self.steps for c in cs]
        window_valid = [c for c in window_conf if c is not None]
        window_correct = [c for _, cs in self.steps for c in cs]
        mode_share = Counter(window_valid).most_common(1)[0][1] / len(window_valid) if window_valid else None
        record.update(window_confidence_std=_std(window_valid), window_mode_share=mode_share,
                      window_levels=len(set(window_valid)),
                      window_invalid_rate=1 - len(window_valid) / len(window_conf) if window_conf else None,
                      window_accuracy=_mean(window_correct))
        full = len(self.steps) == self.window
        if full and self.baseline_accuracy is None and len(window_correct) >= self.min_samples:
            self.baseline_accuracy = record["window_accuracy"]
        record["baseline_accuracy"] = self.baseline_accuracy
        flags = []
        if step > self.warmup and len(window_valid) >= self.min_samples and mode_share >= self.collapse_share:
            flags.append("collapsed")
        if len(window_conf) >= self.min_samples and record["window_invalid_rate"] > self.format_limit:
            flags.append("format_broken")
        if (self.baseline_accuracy is not None and len(window_correct) >= self.min_samples
                and record["window_accuracy"] < self.baseline_accuracy - self.accuracy_drop):
            flags.append("answers_degraded")
        bad = [k for k, v in scalars.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            flags.append("nonfinite")
        record.update(scalars)
        record["flags"] = flags
        events = [dict(step=step, flag=f, nonfinite=bad if f == "nonfinite" else None,
                       **{k: record[k] for k in ("window_mode_share", "window_invalid_rate", "window_accuracy",
                                                 "baseline_accuracy")}) for f in flags if f not in self.raised]
        self.raised.update(flags)
        return record, events

    def state_dict(self):
        return dict(steps=[list(s) for s in self.steps], baseline_accuracy=self.baseline_accuracy,
                    raised=sorted(self.raised))

    def load_state_dict(self, state):
        self.steps = deque((tuple(s) for s in state["steps"]), maxlen=self.window)
        self.baseline_accuracy = state["baseline_accuracy"]
        self.raised = set(state["raised"])


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
