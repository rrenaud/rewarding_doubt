import math

import pytest
import torch

from rewarding_doubt.backend import candidate_datums, confidence_sequences, confidence_loss, datum
from rewarding_doubt.core import (LEVELS, calibration_metrics, grade, objective,
                                  parse_confidence, reward)


def test_reward_matches_released_formula():
    for correct in [False, True]:
        for p in LEVELS:
            clipped = min(.999, max(.001, p))
            score = math.log(clipped if correct else 1 - clipped)
            expected = 10 * ((score - math.log(.001) / 2) /
                             (math.log(.999) - math.log(.001) / 2) + .25 * correct)
            assert float(reward(p, correct, "released")) == pytest.approx(expected, abs=1e-5)


def test_paper_reward_endpoints_and_symmetry():
    assert float(reward(0., True)) == pytest.approx(-1.)
    assert float(reward(1., True)) == pytest.approx(1.)
    assert float(reward(1., False)) == pytest.approx(-1., abs=1e-5)
    for p in LEVELS:
        assert float(reward(p, True)) == pytest.approx(float(reward(1-p, False)), abs=1e-5)


def test_fractional_not_expected_discrete_reward():
    logits = torch.full((1, 11), -100., dtype=torch.float64)
    logits[0, 0] = logits[0, 10] = math.log(.5)
    fractional, confidence = objective(logits, torch.ones(1), "fractional")
    discrete, _ = objective(logits, torch.ones(1), "discrete-exact")
    assert confidence.item() == pytest.approx(.5)
    assert fractional.item() < discrete.item()  # Jensen: reward of mean > mean reward


@pytest.mark.parametrize("mode", ["fractional", "discrete-exact"])
def test_custom_loss_gradients_and_prefix_mask(mode):
    logprobs = [torch.tensor([-2., -3., -math.log(20)], dtype=torch.float64,
                            requires_grad=True) for _ in range(22)]
    loss, _ = confidence_loss([1., 0.], mode, "paper", [1] * 11)([], logprobs)
    loss.backward()
    assert all(torch.equal(p.grad[:-1], torch.zeros(2)) for p in logprobs)
    assert logprobs[0].grad[-1] > 0   # correct answer: reduce low confidence
    assert logprobs[10].grad[-1] < 0  # increase high confidence
    assert logprobs[11].grad[-1] < 0  # wrong answer: increase low confidence
    assert logprobs[21].grad[-1] > 0


def test_fractional_finite_difference():
    torch.manual_seed(7)
    x = torch.randn(2, 11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: objective(z, torch.tensor([1., 0.]),
                                                       "fractional")[0], (x,))


def test_shifted_targets_and_ppo_mask():
    d = datum([10, 11, 12], [20, 21], old_logps=[-.2, -.3], advantages=[2., 2.])
    assert d.model_input.to_ints() == [10, 11, 12, 20]
    assert d.loss_fn_inputs["target_tokens"].data == [11, 12, 20, 21]
    assert d.loss_fn_inputs["advantages"].data == [0., 0., 2., 2.]
    assert d.loss_fn_inputs["logprobs"].data == pytest.approx([0., 0., -.2, -.3])
    assert "weights" not in d.loss_fn_inputs


def test_candidate_order():
    data = candidate_datums([[1, 2], [3, 4, 5]], [[i] for i in range(20, 31)])
    assert len(data) == 22
    assert data[0].loss_fn_inputs["target_tokens"].data == [2, 20]
    assert data[11].loss_fn_inputs["target_tokens"].data == [4, 5, 20]


def test_multitoken_confidence_events_are_disjoint():
    class Tokenizer:
        eos_token_id = 99
        def encode(self, value, **kwargs):
            return [int(x) for x in value]
    sequences = confidence_sequences(Tokenizer())
    assert sequences[1] == [1, 99]
    assert sequences[10] == [1, 0, 99]


def test_multitoken_loss_sums_suffix_and_masks_context():
    lengths = [2] * 10 + [3]
    lps = [torch.zeros(6, requires_grad=True) for _ in range(11)]
    loss, _ = confidence_loss([1.], "fractional", "paper", lengths)([], lps)
    loss.backward()
    for lp, length in zip(lps, lengths):
        assert torch.all(lp.grad[:-length] == 0)
        assert torch.all(lp.grad[-length:] == lp.grad[-1])
    assert lps[-1].grad[-1] < 0


def test_grading_and_strict_confidence_parsing():
    assert grade("The Eiffel Tower", ["Eiffel Tower"])
    assert not grade("red green blue", ["red"])  # exactly .5 does not pass
    assert not grade("", [""])
    assert parse_confidence("10") == 1.
    assert parse_confidence(" 7\n") == .7
    for bad in ["100", "7.5", "7 because", "-1", ""]:
        assert parse_confidence(bad) is None


def test_metrics_ties_edges_and_one_class():
    report = calibration_metrics([0, .5, .5, 1], [0, 0, 1, 1])
    assert report["ece"] == 0.
    assert report["brier"] == .125
    assert report["auroc"] == .875
    assert calibration_metrics([.5], [1])["auroc"] is None
    with pytest.raises(ValueError):
        calibration_metrics([], [])


def test_format_hinge_only_penalizes_low_valid_mass():
    labels = torch.tensor([1.])
    healthy = torch.log(torch.full((1, 11), .99 / 11, dtype=torch.float64))
    collapsed = healthy - 4.  # same within-candidate distribution, mass ~0.018
    for mode in ["fractional", "discrete-exact"]:
        base, _ = objective(healthy, labels, mode)
        assert float(objective(healthy, labels, mode, format_weight=1.)[0]) == pytest.approx(float(base))
        assert float(objective(collapsed, labels, mode)[0]) == pytest.approx(float(base))
        penalized, _ = objective(collapsed, labels, mode, format_weight=1.)
        assert float(penalized - base) == pytest.approx(math.log(.95) - math.log(.99) + 4.)
        z = collapsed.clone().requires_grad_()
        objective(z, labels, mode, format_weight=1.)[0].backward()
        assert float(z.grad.sum()) < 0  # descent raises every candidate's log-probability


def test_baseline_matched_objective_reduces_to_exact_when_format_is_perfect():
    from rewarding_doubt.core import baseline_matched_objective
    pi = torch.tensor([0.] * 8 + [.2, .3, .5], dtype=torch.float64)
    logq = torch.where(pi > 0, pi.log(), torch.full_like(pi, -40.))
    perfect_stop = torch.tensor(0., dtype=torch.float64)
    for mode in ["discrete-exact", "fractional"]:
        J, kl = baseline_matched_objective(logq, 1., mode, "released", -30., perfect_stop, 9, logq, perfect_stop)
        expected = -objective(logq[None], torch.tensor([1.], dtype=torch.float64), mode, "released")[0]
        assert float(J) == pytest.approx(float(expected), abs=1e-4)
        assert float(kl) == pytest.approx(0., abs=1e-6)


def test_baseline_matched_objective_penalizes_invalid_mass_and_not_stopping():
    from rewarding_doubt.core import baseline_matched_objective
    pi = torch.tensor([0.] * 8 + [.2, .3, .5], dtype=torch.float64)
    logq = torch.where(pi > 0, pi.log(), torch.full_like(pi, -40.))
    stop = torch.tensor(0., dtype=torch.float64)
    base, _ = baseline_matched_objective(logq, 0., "discrete-exact", "released", -30., stop, 10)
    # 10% of the mass becomes invalid: J drops by 0.1 * (E[R] - R_invalid).
    leaky, _ = baseline_matched_objective(logq + math.log(.9), 0., "discrete-exact", "released", -30., stop, 10)
    assert float(base - leaky) == pytest.approx(0.1 * (float(base) + 30.), rel=1e-4)
    # Not stopping after the sampled 10 with probability 0.2 costs 0.2 * (R(1.0) - R_invalid).
    slow, _ = baseline_matched_objective(logq, 0., "discrete-exact", "released", -30., torch.tensor(math.log(.8), dtype=torch.float64), 10)
    r10 = float(reward(1.0, 0., "released"))
    assert float(base - slow) == pytest.approx(0.2 * (r10 + 30.), rel=1e-4)
    # Gradient raises the stop probability and the valid mass.
    l = logq.clone().requires_grad_(); s = torch.tensor(math.log(.8), dtype=torch.float64, requires_grad=True)
    J, _ = baseline_matched_objective(l + math.log(.9), 0., "discrete-exact", "released", -30., s, 10)
    J.backward()
    assert float(s.grad) > 0 and float(l.grad.sum()) > 0


def test_baseline_matched_kl_matches_brute_force():
    from rewarding_doubt.core import baseline_matched_objective
    q = torch.tensor([.01] * 8 + [.1, .2, .5], dtype=torch.float64)  # mass 0.88
    r = torch.tensor([.02] * 8 + [.2, .2, .4], dtype=torch.float64)  # mass 0.96
    s, s_ref = .9, .99
    _, kl = baseline_matched_objective(q.log(), 1., "discrete-exact", "released", -30.,
                                       torch.tensor(math.log(s), dtype=torch.float64), 10,
                                       r.log(), torch.tensor(math.log(s_ref), dtype=torch.float64))
    numbers = float((q * (q / r).log()).sum()) + (1 - .88) * math.log((1 - .88) / (1 - .96))
    stop = s * math.log(s / s_ref) + (1 - s) * math.log((1 - s) / (1 - s_ref))
    assert float(kl) == pytest.approx(numbers + stop, rel=1e-6)
