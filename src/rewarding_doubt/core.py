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


def reward(confidence, correct, variant="paper", brier_mix=0.0):
    """Tensor-valued clipped log score, paper or released normalization, optionally mixed with Brier.

    brier_mix m returns (1 - m) * log reward + m * Brier reward, both proper scoring rules, so the mix
    is too. The Brier reward 1 - 2 (p - y)^2 lies in [-1, 1] like the paper-scale log reward; for
    "released" both parts get the same x10 scale and +2.5 bonus for a correct answer.
    """
    if variant not in {"paper", "released"}:
        raise ValueError(f"Unknown reward variant: {variant}")
    if not 0 <= brier_mix <= 1:
        raise ValueError("brier_mix must be in [0, 1]")
    raw = confidence if isinstance(confidence, torch.Tensor) else torch.tensor(confidence, dtype=torch.float64)
    p = raw.clamp(EPS, 1 - EPS)
    y = torch.as_tensor(correct, device=p.device, dtype=p.dtype)
    score = y * p.log() + (1 - y) * torch.log1p(-p)
    if variant == "released":
        lo, hi = math.log(EPS) / 2, math.log(1 - EPS)
        log_part = (score - lo) / (hi - lo)
    else:
        lo, hi = math.log(EPS), math.log(1 - EPS)
        log_part = 2 * (score - lo) / (hi - lo) - 1
    mixed = log_part if brier_mix == 0 else (1 - brier_mix) * log_part + brier_mix * (1 - 2 * (raw - y) ** 2)
    return 10 * (mixed + 0.25 * y) if variant == "released" else mixed


def parse_confidence(text: str) -> float | None:
    match = re.fullmatch(r"\s*(10|[0-9])\s*", text)
    return int(match[1]) / 10 if match else None


def objective(logps: torch.Tensor, labels: torch.Tensor, mode: str, variant="paper",
              format_weight=0.0, format_threshold=0.95, brier_mix=0.0):
    """logps has shape [batch, 11]; normalize over complete confidence sequences.

    Renormalizing makes the score blind to total valid mass M = sum(exp(logps)), so a
    hinge penalty format_weight * relu(log(threshold) - log M) stops the model drifting
    away from emitting a complete confidence string. It is zero while M >= threshold.
    """
    probs = logps.softmax(-1)
    levels = logps.new_tensor(LEVELS)
    confidence = (probs * levels).sum(-1)
    if mode == "fractional":
        scores = reward(confidence, labels, variant, brier_mix)
    elif mode == "discrete-exact":
        scores = (probs * reward(levels, labels[:, None], variant, brier_mix)).sum(-1)
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


def baseline_matched_objective(logq, label, mode, variant, invalid_reward, stop_logp=None, k_star=None,
                               ref_logq=None, ref_stop_logp=None, brier_mix=0.0):
    """The released PPO's expected reward and KL, computed from one forward pass.

    logq: [11] unnormalized log-probabilities of ": " followed by number k (not renormalized, so
    the remaining mass 1 - M is the chance of writing something other than a number).
    stop_logp: log P(<eot> | k_star) for the number k_star the policy sampled this step.

    Expected reward, with every invalid outcome scored `invalid_reward` as in the released code:
        J = sum_k q_k R(k/10, y) + (1 - M) R_invalid + (1 - s) (R_invalid - R(k*/10, y))
    The last term is a one-sample estimate of sum_k q_k (1 - s_k)(R_invalid - R_k), the cost of
    not stopping after the number, unbiased because k* is sampled with probability q_k.
    For "fractional", the valid part is M * R(mean of q/M, y) and R(k*/10) becomes that reward.

    KL to the reference model over the same outcomes: the number distribution with the invalid
    bucket, plus the stop distribution at k* (a one-sample estimate of its q-weighted average).
    Returns (J, KL); the training loss is -J + beta * KL.
    """
    eps = 1e-6
    levels = logq.new_tensor(LEVELS)
    q = logq.exp()
    mass = q.sum().clamp(max=1 - eps)
    rewards = reward(levels, label, variant, brier_mix)
    if mode == "discrete-exact":
        J = (q * rewards).sum()
        sampled_reward = rewards[k_star] if k_star is not None else None
    elif mode == "fractional":
        mean_reward = reward((q / mass * levels).sum(), label, variant, brier_mix)
        J = mass * mean_reward
        sampled_reward = mean_reward
    else:
        raise ValueError(f"Unknown exact objective: {mode}")
    J = J + (1 - mass) * invalid_reward
    if stop_logp is not None:
        J = J + (1 - stop_logp.exp()) * (invalid_reward - sampled_reward)
    if ref_logq is None:
        return J, logq.new_zeros(())
    ref_mass = ref_logq.exp().sum().clamp(max=1 - eps)
    kl = (q * (logq - ref_logq)).sum() + (1 - mass) * (torch.log1p(-mass) - torch.log1p(-ref_mass))
    if stop_logp is not None:
        s, s_ref = stop_logp.exp().clamp(max=1 - eps), ref_stop_logp.exp().clamp(max=1 - eps)
        kl = kl + s * (stop_logp - ref_stop_logp) + (1 - s) * (torch.log1p(-s) - torch.log1p(-s_ref))
    return J, kl
