"""Pure scoring, grading, and differentiable confidence objectives."""
from collections import Counter
import math
import re
import string

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

LEVELS = tuple(i / 10 for i in range(11))
EPS = 0.001


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    a, b = normalize(prediction).split(), normalize(reference).split()
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2 * overlap / (len(a) + len(b))


def grade(prediction: str, references: list[str], multiple_choice=False) -> bool:
    if multiple_choice:
        return prediction.strip() in [r.strip() for r in references]
    return max((token_f1(prediction, r) for r in references), default=0.0) > 0.5


def reward(confidence, correct, variant="paper"):
    """Tensor-valued clipped score; paper affine normalization or released code."""
    if variant not in {"paper", "released"}:
        raise ValueError(f"Unknown reward variant: {variant}")
    p = (confidence if isinstance(confidence, torch.Tensor) else
         torch.tensor(confidence, dtype=torch.float64)).clamp(EPS, 1 - EPS)
    y = torch.as_tensor(correct, device=p.device, dtype=p.dtype)
    score = y * p.log() + (1 - y) * torch.log1p(-p)
    if variant == "released":
        lo, hi = math.log(EPS) / 2, math.log(1 - EPS)
        return 10 * ((score - lo) / (hi - lo) + 0.25 * y)
    lo, hi = math.log(EPS), math.log(1 - EPS)
    return 2 * (score - lo) / (hi - lo) - 1


def parse_confidence(text: str) -> float | None:
    match = re.fullmatch(r"\s*(10|[0-9])\s*", text)
    return int(match[1]) / 10 if match else None


def objective(logps: torch.Tensor, labels: torch.Tensor, mode: str, variant="paper",
              format_weight=0.0, format_threshold=0.95):
    """logps has shape [batch, 11]; normalize over complete confidence sequences.

    Renormalizing makes the score blind to total valid mass M = sum(exp(logps)), so a
    hinge penalty format_weight * relu(log(threshold) - log M) stops the model drifting
    away from emitting a complete confidence string. It is zero while M >= threshold.
    """
    probs = logps.softmax(-1)
    levels = logps.new_tensor(LEVELS)
    confidence = (probs * levels).sum(-1)
    if mode == "fractional":
        scores = reward(confidence, labels, variant)
    elif mode == "discrete-exact":
        scores = (probs * reward(levels, labels[:, None], variant)).sum(-1)
    else:
        raise ValueError(f"Unknown exact objective: {mode}")
    hinge = (math.log(format_threshold) - logps.logsumexp(-1)).clamp(min=0)
    return -scores.mean() + format_weight * hinge.mean(), confidence


def calibration_metrics(confidences, labels, bins=10):
    p, y = np.asarray(confidences, dtype=float), np.asarray(labels, dtype=float)
    if len(p) == 0 or p.shape != y.shape:
        raise ValueError("Need nonempty, equally sized confidence and label arrays")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Confidences must be finite probabilities")
    if not np.all((y == 0) | (y == 1)) or bins < 1:
        raise ValueError("Labels must be binary and bins positive")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for b in range(bins):
        mask = indices == b
        if mask.any():
            acc, conf = float(y[mask].mean()), float(p[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            reliability.append(dict(bin=b, count=int(mask.sum()), accuracy=acc, confidence=conf))
    clipped = np.clip(p, EPS, 1 - EPS)
    return dict(n=len(p), accuracy=float(y.mean()), ece=float(ece),
                brier=float(np.mean((p - y) ** 2)),
                nll=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
                auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                reliability=reliability)
