"""Pure pieces of the released Rewarding Doubt PPO baseline (TRL 0.8.6 PPOTrainer semantics).

References: pasta99/RewardingDoubt SingleAnswerSetting/{Train.py, util/RLHelper.py,
util/ResponseHandling.py} and trl==0.8.6 PPOTrainer.{compute_rewards, compute_advantages}.
"""
import math
import re

# SingleAnswerSetting/util/Prompts.py, "open" task type (used for TriviaQA).
SYSTEM = (
    "You will get questions. Answer with the correct answer. Additionally provide a confidence "
    "between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value "
    "close to 0 means you think there is a high probability that the answer is wrong. The closer "
    "the value is to 10, the higher you think is the probability that the answer is correct. The "
    "output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else."
)

# trl 0.8.6 PPOConfig defaults, plus init_kl_coef=0.05 from Train.py.
CONFIG = dict(init_kl_coef=0.05, kl_target=6., kl_horizon=10000, gamma=1., lam=0.95,
              cliprange=0.2, ppo_epochs=4)


def parse_response(text):
    """ResponseHandling.parse_answer_confidence: regex *search*, so trailing text is allowed.

    Returns (answer, confidence); confidence is None when the format is wrong or the value is
    outside 0..10 (RLHelper.reward_function penalizes both identically).
    """
    match = re.search(r"Answer:\s*(?P<answer>.*?),\s*Confidence:\s*(?P<confidence>\d+)", text)
    if not match:
        return None, None
    confidence = int(match["confidence"])
    return match["answer"], (confidence if confidence <= 10 else None)


def token_rewards(score, logprobs, ref_logprobs, kl_coef):
    """PPOTrainer.compute_rewards with kl_penalty='kl': -beta*KL per token, score on the last."""
    rewards = [-kl_coef * (lp - ref) for lp, ref in zip(logprobs, ref_logprobs)]
    rewards[-1] += score
    return rewards


def advantages_without_value_head(rewards, gamma=1., lam=0.95):
    """PPOTrainer.compute_advantages (GAE) with the value head replaced by V = 0.

    Tinker trains a LoRA on the language model only, so there is no scalar value head; with
    V = 0, GAE reduces to lambda-discounted reward-to-go, and the batch whitening below acts
    as a batch-mean baseline.
    """
    advantages, running = [0.] * len(rewards), 0.
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * lam * running
        advantages[t] = running
    return advantages


def whiten(sequences, eps=1e-8):
    """trl.core.masked_whiten over every response token in the batch (unbiased variance)."""
    flat = [x for seq in sequences for x in seq]
    if len(flat) < 2:
        raise ValueError("Need at least two response tokens to whiten")
    mean = sum(flat) / len(flat)
    var = sum((x - mean) ** 2 for x in flat) / (len(flat) - 1)
    scale = 1 / math.sqrt(var + eps)
    return [[(x - mean) * scale for x in seq] for seq in sequences]


class AdaptiveKLController:
    """trl.trainer.AdaptiveKLController (Ziegler et al., 2019)."""

    def __init__(self, init_kl_coef, target, horizon):
        self.value, self.target, self.horizon = init_kl_coef, target, horizon

    def update(self, current_kl, n_steps):
        error = min(max(current_kl / self.target - 1, -0.2), 0.2)
        self.value *= 1 + error * n_steps / self.horizon
