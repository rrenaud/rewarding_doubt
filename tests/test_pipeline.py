"""Offline SDK boundary smoke tests; these do not validate the remote service."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from rewarding_doubt import cli


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, *args, **kwargs):
        return [100, 101]

    def encode(self, text, **kwargs):
        return [int(text)] if text.isdigit() else [102, 103]

    def decode(self, tokens, **kwargs):
        return str(tokens[0])


class FakeClient:
    def get_tokenizer(self):
        return Tokenizer()

    def sample(self, prompt, num_samples, sampling_params):
        return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[i % 11], logprobs=[-3.])
                                                 for i in range(num_samples)]))

    def compute_logprobs(self, prompt):
        return Future([None] + [-math.log(20)] * (len(prompt.to_ints()) - 1))

    def save_weights_and_get_sampling_client(self):
        return self

    def forward_backward(self, datums, loss_fn):
        assert loss_fn == "ppo"
        assert all(set(d.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
                   for d in datums)
        return Future(SimpleNamespace(metrics={"loss:sum": 0.}))

    def forward_backward_custom(self, datums, fn):
        lps = [torch.full((len(d.model_input.to_ints()),), -math.log(20), requires_grad=True)
               for d in datums]
        loss, metrics = fn(datums, lps)
        loss.backward()
        assert any(lp.grad[-1] != 0 for lp in lps)
        return Future(SimpleNamespace(metrics=metrics))

    def optim_step(self, params):
        return Future(None)

    def save_state(self, name):
        return Future(SimpleNamespace(path="tinker://fake-state"))

    def save_weights_for_sampler(self, name):
        return Future(SimpleNamespace(path="tinker://fake-sampler"))


@pytest.mark.parametrize("mode", ["discrete-ppo", "discrete-exact", "fractional"])
def test_train_and_evaluate(tmp_path, monkeypatch, mode):
    client = FakeClient()
    service = SimpleNamespace(create_lora_training_client=lambda **kwargs: client,
                              create_sampling_client=lambda **kwargs: client)
    monkeypatch.setattr(cli.tinker, "ServiceClient", lambda: service)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(dict(question="Capital?", answer=a,
                                               references=["Paris"])) for a in ["Paris", "Lyon"]))
    args = SimpleNamespace(data=str(data), grading="f1", output=str(tmp_path / "train"),
                           model="fake", rank=16, seed=2, epochs=2, batch_size=2,
                           group_size=4, mode=mode, reward="paper", lr=1e-5, max_steps=1,
                           format_weight=1., format_threshold=.95, save_every=1, prompt="ours")
    cli.train(args)
    checkpoint = json.loads((tmp_path / "train/checkpoint.json").read_text())
    assert checkpoint["steps"] == 1
    assert checkpoint["snapshots"] == [dict(step=1, sampler_path="tinker://fake-sampler")]
    args.output = str(tmp_path / "eval")
    args.checkpoint = checkpoint["sampler_path"]
    args.bins = 11
    cli.evaluate(args)
    metrics = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert metrics["fractional"]["brier"] == pytest.approx(.25)
    assert metrics["mean_valid_confidence_mass"] == pytest.approx(11 / 400)
    assert metrics["total_valid_answers"] == 2
