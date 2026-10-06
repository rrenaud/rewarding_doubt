"""CPU checks of modal_repro/minimal_trainer.py on a tiny random Qwen2: shared-prefix scoring equals a full
forward (levels and gradients, with LoRA and output biases), and the level arithmetic matches scoring
each level's tokens directly."""
import pathlib
import random
import sys

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "modal_repro"))
import minimal_trainer as mt  # noqa: E402


class DigitTokenizer:
    """Qwen-2.5's level shape: ": k" is [":", " ", digit], and "10" is "1" "0"."""
    def encode(self, text, add_special_tokens=False):
        assert text.startswith(": ")
        return [5, 6] + [10 + int(d) for d in text[2:]]


def tiny_qwen(dtype):
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    lm = transformers.Qwen2ForCausalLM(config).to(dtype)
    lm.requires_grad_(False)
    return lm


def rows(n=6, prefix=40, seed=0):
    rng = random.Random(seed)
    shared = [rng.randrange(20, 64) for _ in range(prefix)]
    return [dict(ids=shared + [rng.randrange(20, 64) for _ in range(rng.randrange(3, 15))]) for _ in range(n)]


def trained_model(dtype):
    lm = tiny_qwen(dtype)
    params = (mt.add_lora(lm, list(mt.LORA_MODULES), range(1, 4)) + mt.add_output_biases(lm, range(1, 4), "self_attn")
              + mt.add_output_biases(lm, range(2, 4), "mlp"))
    with torch.no_grad():  # move away from the zero initialization so every adapter matters
        for p in params:
            p.add_(0.1 * torch.randn_like(p))
    return lm, params


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2)])
def test_shared_prefix_matches_full_forward(dtype, tol):
    lm, params = trained_model(dtype)
    result = mt.check(lm, mt.Levels(DigitTokenizer()), rows(), pad=1, params=params)
    assert result["max_abs_level_diff"] < tol
    assert result["max_rel_grad_diff"] < (1e-4 if dtype == torch.float32 else 5e-2)


def test_levels_match_direct_scoring():
    lm, _ = trained_model(torch.float32)
    levels = mt.Levels(DigitTokenizer())
    data = rows()
    with torch.no_grad():
        got = mt.level_logps(lm, levels, [r["ids"] for r in data], pad=1)
        for b, r in enumerate(data):
            for k in range(11):
                tokens = DigitTokenizer().encode(f": {k}")
                logp = lm(input_ids=torch.tensor([r["ids"] + tokens])).logits[0].log_softmax(-1)
                n = len(r["ids"])
                want = sum(logp[n - 1 + i, t] for i, t in enumerate(tokens))
                if k == 1:  # "1" not followed by "0"
                    want = want + torch.log1p(-logp[n + 2, 10].exp())
                assert float(got[b, k]) == pytest.approx(float(want), abs=1e-4)


def test_bias_hooks_start_at_identity():
    lm = tiny_qwen(torch.float32)
    data = rows()
    levels = mt.Levels(DigitTokenizer())
    with torch.no_grad():
        before = mt.level_logps(lm, levels, [r["ids"] for r in data], pad=1)
        mt.add_lora(lm, ["o"], range(4))
        mt.add_output_biases(lm, range(4), "self_attn")
        mt.add_output_biases(lm, range(4), "mlp")
        after = mt.level_logps(lm, levels, [r["ids"] for r in data], pad=1)
    assert torch.equal(before, after)
