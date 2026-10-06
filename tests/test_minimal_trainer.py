"""CPU checks of modal_repro/minimal_trainer.py on a tiny random Qwen2: shared-prefix scoring equals a full
forward (levels, answer log-probs and gradients, with LoRA and output biases), the level arithmetic matches
scoring each level's tokens directly, and adapters_off gives the model without adapters."""
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
    """Rows sharing a `prefix`, then 3-14 more tokens of which the last 2-6 play the answer (answer_start)."""
    rng = random.Random(seed)
    shared = [rng.randrange(20, 64) for _ in range(prefix)]
    out = []
    for _ in range(n):
        ids = shared + [rng.randrange(20, 64) for _ in range(rng.randrange(8, 15))]
        out.append(dict(ids=ids, answer_start=len(ids) - rng.randrange(2, 7)))
    return out


def scored(lm, data, shared_prefix=True):
    return mt.score_rows(lm, mt.Levels(DigitTokenizer()), [r["ids"] for r in data], [r["answer_start"] for r in data],
                         pad=1, shared_prefix=shared_prefix)


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


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_shared_prefix_answers_match_full_forward(dtype, tol):
    lm, _ = trained_model(dtype)
    data = rows()
    with torch.no_grad():
        shared, full = scored(lm, data)[1], scored(lm, data, shared_prefix=False)[1]
        logits = lm(input_ids=torch.tensor([data[0]["ids"]])).logits[0].float().log_softmax(-1)
    for r, a, b in zip(data, shared, full):
        assert a.shape == (len(r["ids"]) - r["answer_start"], 64)
        assert float((a - b).abs().max()) < tol
    start = data[0]["answer_start"]  # position t predicts token t + 1
    assert float((full[0] - logits[start - 1:len(data[0]["ids"]) - 1]).abs().max()) < tol


def test_adapters_off_is_the_base_model():
    lm, _ = trained_model(torch.float32)
    base = tiny_qwen(torch.float32)
    data = rows()
    with torch.no_grad():
        with mt.adapters_off(lm):
            off_levels, off_answers = scored(lm, data)
        on_levels, on_answers = scored(lm, data)
        base_levels, base_answers = scored(base, data)
    assert float((off_levels - base_levels).abs().max()) < 1e-5
    assert max(float((a - b).abs().max()) for a, b in zip(off_answers, base_answers)) < 1e-5
    assert float((on_levels - base_levels).abs().max()) > 1e-3  # the adapters matter when on
    kl = mt.answer_kl(on_answers, off_answers)
    assert bool((kl > 0).all()) and float(mt.answer_kl(off_answers, off_answers).abs().max()) < 1e-6
    assert all(p.requires_grad for n, p in lm.named_parameters() if "lora_" in n)  # restored after adapters_off


def test_levels_match_direct_scoring():
    lm, _ = trained_model(torch.float32)
    data = rows()
    with torch.no_grad():
        got = scored(lm, data)[0]
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
    with torch.no_grad():
        before = scored(lm, data)[0]
        mt.add_lora(lm, ["o"], range(4))
        mt.add_output_biases(lm, range(4), "self_attn")
        mt.add_output_biases(lm, range(4), "mlp")
        after = scored(lm, data)[0]
    assert torch.equal(before, after)


def test_topk_answer_kl_bounds_exact():
    lm, _ = trained_model(torch.float32)
    base = tiny_qwen(torch.float32)
    data = rows()
    with torch.no_grad():
        policy = scored(lm, data)[1]
        reference = scored(base, data)[1]
    exact = mt.answer_kl(policy, reference)
    for k, check in ((64, "equal"), (8, "below"), (1, "below")):  # the tiny vocabulary has 64 tokens
        for r, ref in zip(data, reference):
            logp, idx = ref.topk(k, dim=-1)
            r["answer_ref_logp"], r["answer_ref_idx"] = logp, idx.to(torch.int32)
        coarse = mt.topk_answer_kl(policy, data)
        if check == "equal":
            assert torch.allclose(coarse, exact, atol=1e-4)
        else:
            assert bool((coarse <= exact + 1e-5).all()) and bool((coarse >= -1e-6).all())
    assert float(mt.topk_answer_kl(reference, data).abs().max()) < 1e-5  # policy = reference


def test_v_bias_shifts_attention_output_by_a_constant():
    """Attention weights sum to 1, so a change in v_proj's bias adds the same vector at every position."""
    lm = tiny_qwen(torch.float32)
    attn = lm.model.layers[2].self_attn
    seen = []
    handle = attn.register_forward_hook(lambda m, i, o: seen.append(o[0].detach().clone()))
    ids = torch.tensor([rows()[0]["ids"]])
    with torch.no_grad():
        lm(input_ids=ids)
        attn.v_proj.bias.add_(torch.randn_like(attn.v_proj.bias))
        lm(input_ids=ids)
    handle.remove()
    shift = seen[1] - seen[0]  # [1, T, hidden]
    assert float(shift.abs().max()) > 1e-2
    assert torch.allclose(shift, shift[:, :1].expand_as(shift), atol=1e-5)


def test_stock_params_train_through_masters_and_switch_off():
    lm = tiny_qwen(torch.float32)
    data = rows()
    masters = mt.add_stock_params(lm, range(1, 4), ["self_attn.v_proj.bias", "input_layernorm.weight"])
    with torch.no_grad():
        base_levels = scored(lm, data)[0]
        for m in masters.values():
            m.add_(0.1 * torch.randn_like(m))
        mt.sync_stock(lm, "weights")
        changed = scored(lm, data)[0]
        with mt.adapters_off(lm):
            off = scored(lm, data)[0]
        back = scored(lm, data)[0]
    assert float((changed - base_levels).abs().max()) > 1e-3
    assert float((off - base_levels).abs().max()) < 1e-6 and torch.equal(back, changed)
    scored(lm, data)[0].sum().backward()  # gradients reach the bf16/fp32 weights, then the masters
    mt.sync_stock(lm, "grads")
    assert all(m.grad is not None and float(m.grad.abs().sum()) > 0 for m in masters.values())
