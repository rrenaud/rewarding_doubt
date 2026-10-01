import json
import math
from types import SimpleNamespace

import pytest

from rewarding_doubt import cli, paper_ppo


def test_parse_response_matches_released_regex():
    assert paper_ppo.parse_response("Answer: Paris, Confidence: 7") == ("Paris", 7)
    # re.search, not fullmatch: trailing text is accepted, as in the released code.
    assert paper_ppo.parse_response("Answer: Paris, Confidence: 10\nbecause...") == ("Paris", 10)
    assert paper_ppo.parse_response("Answer: Paris, Confidence: 11") == ("Paris", None)
    assert paper_ppo.parse_response("Paris, 7") == (None, None)


def test_rewards_and_advantages_follow_trl():
    rewards = paper_ppo.token_rewards(2., [-1., -2.], [-1.5, -1.], kl_coef=0.1)
    assert rewards == pytest.approx([-0.05, 0.1 + 2.])
    assert paper_ppo.advantages_without_value_head([1., 2., 3.], lam=0.5) == pytest.approx(
        [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.])
    white = paper_ppo.whiten([[1., 2.], [3.], [4., 5.]])
    flat = [x for seq in white for x in seq]
    assert [len(s) for s in white] == [2, 1, 2]
    assert sum(flat) == pytest.approx(0.)
    assert sum(x * x for x in flat) / (len(flat) - 1) == pytest.approx(1., rel=1e-6)


def test_adaptive_kl_controller_matches_trl():
    kl = paper_ppo.AdaptiveKLController(0.05, target=6., horizon=10000)
    kl.update(0., n_steps=8)  # error clipped to -0.2
    assert kl.value == pytest.approx(0.05 * (1 - 0.2 * 8 / 10000))
    kl.update(6.6, n_steps=8)  # error 0.1
    assert kl.value == pytest.approx(0.05 * (1 - 0.2 * 8 / 10000) * (1 + 0.1 * 8 / 10000))


VOCAB = {1: "Answer: Paris", 2: ",", 7: " Confidence", 8: ": ", 9: "9", 10: "Answer: Lyon", 99: ""}


class Tokenizer:
    eos_token_id, pad_token_id = 99, 98

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"] == paper_ppo.SYSTEM
        return [100, 101]

    def encode(self, text, **kwargs):
        return [7] if text == " Confidence" else [102]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(VOCAB.get(t, "") for t in tokens)


class Client:
    def __init__(self):
        self.calls = []

    def get_tokenizer(self):
        return Tokenizer()

    def save_weights_and_get_sampling_client(self):
        return self

    def sample(self, prompt, n, params):
        if 7 in params.stop:  # answer stage: alternate a correct and a wrong answer
            answer = [1, 2, 7] if params.seed % 2 == 0 else [10, 2, 7]
            return SimpleNamespace(result=lambda: SimpleNamespace(
                sequences=[SimpleNamespace(tokens=answer, logprobs=[-.1] * 3)]))
        assert prompt.to_ints()[-1] == 7 and params.max_tokens == 500
        return SimpleNamespace(result=lambda: SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[8, 9, 99], logprobs=[-.1, -.5, -.2])]))

    def compute_logprobs(self, prompt):
        return SimpleNamespace(result=lambda: [None] + [-.3] * (len(prompt.to_ints()) - 1))

    def forward_backward(self, datums, loss_fn):
        assert loss_fn == "ppo"
        self.calls.append(("fb", datums))
        return SimpleNamespace(result=lambda: SimpleNamespace(metrics={"loss:sum": 0.}))

    def optim_step(self, params):
        assert (params.beta2, params.eps) == (0.999, 1e-8)
        self.calls.append(("optim", None))
        return SimpleNamespace(result=lambda: None)

    def save_state(self, name):
        return SimpleNamespace(result=lambda: SimpleNamespace(path="tinker://state"))

    def save_weights_for_sampler(self, name):
        return SimpleNamespace(result=lambda: SimpleNamespace(path=f"tinker://{name}"))


def test_paper_ppo_training_loop(tmp_path, monkeypatch):
    client = Client()
    service = SimpleNamespace(create_lora_training_client=lambda **kwargs: client,
                              create_sampling_client=lambda **kwargs: client)
    monkeypatch.setattr(cli.tinker, "ServiceClient", lambda: service)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(dict(question=f"Capital {i}?", answer="", references=["Paris"],
                                              answer_format_valid=False)) for i in range(4)))
    args = SimpleNamespace(data=str(data), grading="f1", output=str(tmp_path / "train"), model="fake",
                           rank=8, seed=2, epochs=1, batch_size=4, group_size=8, mode="paper-ppo",
                           reward="released", lr=1e-5, max_steps=0, format_weight=1.,
                           format_threshold=.95, save_every=0, prompt="paper")
    cli.train(args)
    record = json.loads((tmp_path / "train/metrics.jsonl").read_text())
    rollouts = [json.loads(l) for l in (tmp_path / "train/rollouts.jsonl").read_text().splitlines()]
    assert [r["text"] for r in rollouts[:2]] == ["Answer: Paris, Confidence: 9", "Answer: Lyon, Confidence: 9"]
    assert record["invalid_rate"] == 0 and record["answer_stage_stopped"] == 1
    assert record["answer_accuracy"] == 0.5 and record["confidence_mean"] == pytest.approx(0.9)
    assert record["kl"] == pytest.approx(-.8 + .9)  # sum(sampler) - sum(reference) per response
    kinds = [kind for kind, _ in client.calls]
    assert kinds == ["fb", "optim"] * 8  # 4 PPO epochs x 2 minibatches of batch/2
    datums = [d for kind, ds in client.calls if kind == "fb" for d in ds]
    assert len(datums) == 16
    # Query = prompt + answer through " Confidence"; only the 3 response tokens carry advantages.
    adv = datums[0].loss_fn_inputs["advantages"].data
    assert len(adv) == 2 + 3 + 3 - 1 and adv[:4] == [0.] * 4 and all(a != 0 for a in adv[4:])
    # Correct answers get higher whitened advantages than wrong ones at the final token.
    finals = {tuple(d.model_input.to_ints()[2:3]): d.loss_fn_inputs["advantages"].data[-1] for d in datums}
    assert finals[(1,)] > finals[(10,)]


def test_matches_trl_0_8_6_on_padded_batches():
    """Golden check against TRL's own code, with left padding (query) and right padding."""
    from types import SimpleNamespace as NS
    import torch
    import trl_0_8_6_reference as trl

    gen = torch.Generator().manual_seed(0)
    cfg = paper_ppo.CONFIG
    trainer = NS(config=NS(kl_penalty="kl", whiten_rewards=False, gamma=cfg["gamma"], lam=cfg["lam"],
                           cliprange=cfg["cliprange"]),
                 kl_ctl=trl.AdaptiveKLController(0.07, cfg["kl_target"], cfg["kl_horizon"]))
    lengths, width, pad = [3, 1, 5, 2, 4, 4, 2, 3], 9, 2  # pad query positions on the left
    scores = torch.randn(len(lengths), generator=gen) * 10
    logps = [(-torch.rand(n, generator=gen) * 3).tolist() for n in lengths]
    refs = [(-torch.rand(n, generator=gen) * 3).tolist() for n in lengths]
    dense = lambda seqs: torch.tensor([[0.] * pad + s + [0.] * (width - pad - len(s)) for s in seqs])
    mask = dense([[1.] * n for n in lengths])
    rewards, _, kls = trl.compute_rewards(trainer, scores, dense(logps), dense(refs), mask)
    _, expected, _ = trl.compute_advantages(trainer, torch.zeros_like(rewards), rewards, mask)

    ours_rewards = [paper_ppo.token_rewards(float(s), lp, ref, 0.07) for s, lp, ref in zip(scores, logps, refs)]
    ours = paper_ppo.whiten([paper_ppo.advantages_without_value_head(r, cfg["gamma"], cfg["lam"])
                             for r in ours_rewards])
    assert torch.allclose(dense(ours_rewards), rewards * mask, atol=1e-5)
    assert torch.allclose(dense(ours), expected * mask, atol=1e-5)
    # objective/kl and the adaptive controller update.
    trl_kl = (kls * mask).sum(-1).mean().item()
    ours_kl = sum(sum(lp) - sum(r) for lp, r in zip(logps, refs)) / len(lengths)
    assert ours_kl == pytest.approx(trl_kl, abs=1e-5)
    ctl = paper_ppo.AdaptiveKLController(0.07, cfg["kl_target"], cfg["kl_horizon"])
    trainer.kl_ctl.update(trl_kl, len(lengths))
    ctl.update(ours_kl, len(lengths))
    assert ctl.value == pytest.approx(trainer.kl_ctl.value)

    # Policy loss: Tinker's summed ppo loss with advantages / n_tokens equals TRL's masked mean.
    new = [[lp + 0.3 * float(torch.randn(1, generator=gen)) for lp in seq] for seq in logps]
    idx = [0, 2, 5, 7]
    sub = lambda t: t[idx]
    expected_loss = trl.policy_loss(trainer, sub(dense(logps)), sub(dense(new)), sub(mask), sub(expected))
    n_tokens = sum(lengths[i] for i in idx)
    tinker_loss = 0.
    for i in idx:
        for old, cur, a in zip(logps[i], new[i], ours[i]):
            a = a / n_tokens
            ratio = math.exp(cur - old)
            tinker_loss -= min(ratio * a, min(max(ratio, 1 - cfg["cliprange"]), 1 + cfg["cliprange"]) * a)
    assert tinker_loss == pytest.approx(expected_loss.item(), abs=1e-5)
