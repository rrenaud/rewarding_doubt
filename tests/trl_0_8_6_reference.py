"""Verbatim excerpts from trl==0.8.6 (Apache-2.0, Hugging Face) used as a golden reference.

Sources: trl/core.py (masked_mean, masked_var, masked_whiten), trl/trainer/utils.py
(AdaptiveKLController), trl/trainer/ppo_trainer.py (compute_rewards, _kl_penalty,
compute_advantages, and the policy term of loss). Methods take a `self` providing `config`
and `kl_ctl`. Only the code paths used by the released Rewarding Doubt config are kept.
"""
import numpy as np
import torch


def masked_mean(values, mask, axis=None):
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
    rewards, non_score_rewards, kls = [], [], []
    for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
        kl = _kl_penalty(self, logprob, ref_logprob)
        kls.append(kl)
        non_score_reward = -self.kl_ctl.value * kl
        non_score_rewards.append(non_score_reward)
        reward = non_score_reward.clone()
        last_non_masked_index = mask.nonzero()[-1]
        reward[last_non_masked_index] += score
        rewards.append(reward)
    return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)


def _kl_penalty(self, logprob, ref_logprob):
    if self.config.kl_penalty == "kl":
        return logprob - ref_logprob
    raise NotImplementedError


def compute_advantages(self, values, rewards, mask):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]

    values = values * mask
    rewards = rewards * mask

    if self.config.whiten_rewards:
        rewards = masked_whiten(rewards, mask, shift_mean=False)

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

    returns = advantages + values
    advantages = masked_whiten(advantages, mask)
    advantages = advantages.detach()
    return values, advantages, returns


def policy_loss(self, old_logprobs, logprobs, mask, advantages):
    """The pg_loss term of PPOTrainer.loss (the value term is absent without a value head)."""
    ratio = torch.exp(logprobs - old_logprobs)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
    return masked_mean(torch.max(pg_losses, pg_losses2), mask)
