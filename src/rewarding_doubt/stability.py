"""Training-stability monitor shared by the exact trainer and the PPO wrapper.

Each training step reports the batch's sampled confidences (None for a malformed response), the
answers' correctness, and any scalars worth watching (loss, gradient norm, entropy, ...). The
monitor keeps a rolling window of steps and raises flags for the failure modes seen so far:

* collapsed: after warm-up, almost every confidence in the window is the same level (the exact
  arm's seed 8 locked on 2 for every question; once pi is one-hot its gradient vanishes, so the
  run never recovers). The base model itself says 10 nearly always, hence the warm-up.
* format_broken: many malformed responses in the window (high-learning-rate search runs).
* answers_degraded: answer accuracy over the last `accuracy_window` steps fell `accuracy_drop`
  below its first such window. The adapter touches every token, so confidence training can damage
  the answers themselves. The window is long (64 steps, 512 answers) because a 16-step window
  (128 answers) has sampling noise of ~0.06 between windows, and checked every step over a long
  run that raised false alarms within 100 steps.
* nonfinite: a reported scalar is NaN or infinite.

Each record also carries a live calibration estimate over the same `accuracy_window` (512 answers
at batch 8): `window_brier` and `window_ece` (11 bins, as torchmetrics in the released evaluation)
of the sampled confidences against answer correctness. Confidences come in as levels 0-10, paired
with `correct` by position. It is a training-time trend, not the reported metric: training samples
the confidence at T=1 on ever-new questions, and 512 answers leave ~0.03 of noise in ECE.

A flag is reported once, as an event, when it first rises; `record["flags"]` lists the flags
active at that step.
"""
import math
from collections import Counter, deque

FLAGS = ("collapsed", "format_broken", "answers_degraded", "nonfinite")


class StabilityMonitor:
    def __init__(self, window=16, warmup=32, min_samples=64, collapse_share=0.98, format_limit=0.2,
                 accuracy_drop=0.15, accuracy_window=64):
        self.window, self.warmup, self.min_samples = window, warmup, min_samples
        self.accuracy_window = accuracy_window
        self.correct = deque(maxlen=accuracy_window)  # per-step answer correctness lists
        self.calibration = deque(maxlen=accuracy_window)  # per-step [(confidence / 10, correct)]
        self.collapse_share, self.format_limit, self.accuracy_drop = collapse_share, format_limit, accuracy_drop
        self.steps = deque(maxlen=window)  # (confidences, correct) per step
        self.baseline_accuracy = None
        self.raised = set()
        self.last_flagged_step = None  # most recent step with any active flag

    def update(self, step, confidences, correct=(), **scalars):
        """Add one step; return (record, new_events)."""
        confidences, correct = list(confidences), [float(c) for c in correct]
        self.steps.append((confidences, correct))
        self.correct.append(correct)
        self.calibration.append([(c / 10, y) for c, y in zip(confidences, correct) if c is not None])
        valid = [c for c in confidences if c is not None]
        record = dict(step=step, batch_confidence_mean=_mean(valid), batch_confidence_std=_std(valid),
                      batch_invalid_rate=1 - len(valid) / len(confidences) if confidences else None,
                      batch_accuracy=_mean(correct))
        window_conf = [c for cs, _ in self.steps for c in cs]
        window_valid = [c for c in window_conf if c is not None]
        window_correct = [c for cs in self.correct for c in cs]
        mode_share = Counter(window_valid).most_common(1)[0][1] / len(window_valid) if window_valid else None
        record.update(window_confidence_std=_std(window_valid), window_mode_share=mode_share,
                      window_levels=len(set(window_valid)),
                      window_invalid_rate=1 - len(window_valid) / len(window_conf) if window_conf else None,
                      window_accuracy=_mean(window_correct))
        pairs = [pair for step_pairs in self.calibration for pair in step_pairs]
        record.update(window_brier=_mean([(p - y) ** 2 for p, y in pairs]), window_ece=_ece(pairs),
                      window_calibration_n=len(pairs))
        full = len(self.correct) == self.accuracy_window
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
        if flags:
            self.last_flagged_step = step
        events = [dict(step=step, flag=f, nonfinite=bad if f == "nonfinite" else None,
                       **{k: record[k] for k in ("window_mode_share", "window_invalid_rate", "window_accuracy",
                                                 "baseline_accuracy")}) for f in flags if f not in self.raised]
        self.raised.update(flags)
        return record, events

    def clean_since(self, step):
        """No flag was active at any step after `step`."""
        return self.last_flagged_step is None or self.last_flagged_step <= step

    def state_dict(self):
        return dict(steps=[list(s) for s in self.steps], correct=[list(c) for c in self.correct],
                    calibration=[list(c) for c in self.calibration],
                    baseline_accuracy=self.baseline_accuracy,
                    raised=sorted(self.raised), last_flagged_step=self.last_flagged_step)

    def load_state_dict(self, state):
        self.steps = deque((tuple(s) for s in state["steps"]), maxlen=self.window)
        self.correct = deque((list(c) for c in state.get("correct", [s[1] for s in state["steps"]])),
                             maxlen=self.accuracy_window)
        self.calibration = deque((list(map(tuple, c)) for c in state.get("calibration", [])), maxlen=self.accuracy_window)
        self.baseline_accuracy = state["baseline_accuracy"]
        self.raised = set(state["raised"])
        self.last_flagged_step = state.get("last_flagged_step")


def _ece(pairs, bins=11):
    """Binary calibration error with equal-width bins over [0, 1] (torchmetrics' l1 form)."""
    if not pairs:
        return None
    groups = {}
    for p, y in pairs:
        groups.setdefault(min(int(p * bins), bins - 1), []).append((p, y))
    return sum(len(g) * abs(_mean([y for _, y in g]) - _mean([p for p, _ in g])) for g in groups.values()) / len(pairs)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
