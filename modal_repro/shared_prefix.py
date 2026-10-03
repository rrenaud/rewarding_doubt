"""Score the 11 confidence strings without recomputing the shared prefix 11 times.

Two methods live here. `single_pass_logps` (the default in exact_llama.py) reads all 11 number
probabilities from one softmax in one ordinary forward pass, and checks that the model stops
after one sampled number. `candidate_logps` scores all 11 complete strings, including the stop
token, with a KV cache; it is described below.

The strings ": k<eot>" share every token up to the number. Llama-3 tokenizes them as
[":", " ", k, <eot>], with "10" a single token. So:

* pass 1 runs query + [":", " "] once. Its logits give log P(":"), log P(" ") and, at the last
  position, log P(k) for all 11 numbers (one softmax);
* pass 2 runs the 11 one-token continuations [k] against pass 1's KV cache (batch 11), giving
  log P(<eot> | ..., k).

That is about len(query) + 11 positions instead of 11 * len(query). Gradients flow through the
cached keys and values.

Unsloth's CausalLM forward routes any call with a cache to its inference path, and in training
mode it disables caching when no cache is passed. This module therefore calls the patched inner
LlamaModel directly: pass 1 with an empty cache (to keep `use_cache`) and xformers'
LowerTriangularMask, pass 2 with no mask (a one-token query attends to every cached key) and
explicit RoPE positions. Continuations longer than one token use a bottom-right causal mask;
only the one-token case (Llama-3) is verified against full sequences
(`verify_shared_prefix.py`).
"""
import contextlib

import torch
from xformers.ops import fmha
import unsloth.models.llama as unsloth_llama


@contextlib.contextmanager
def training_final_norm():
    """With use_cache=True, Unsloth's LlamaModel applies its *inference* RMSNorm kernel to the
    final hidden state. Use the training kernel instead, so gradients flow through it."""
    original = unsloth_llama.fast_rms_layernorm_inference
    unsloth_llama.fast_rms_layernorm_inference = lambda norm, X: unsloth_llama.fast_rms_layernorm(norm, X, gemma=False)
    try:
        yield
    finally:
        unsloth_llama.fast_rms_layernorm_inference = original


def _kv(entry):
    """A layer's cache entry as (K, V), unwrapping any extra nesting."""
    while isinstance(entry[0], (tuple, list)):
        entry = entry[0]
    return entry[0], entry[1]


def _parts(model):
    causal_lm = model.base_model.model  # PeftModel -> LlamaForCausalLM
    return causal_lm.model, causal_lm.lm_head


def _common_prefix(candidates):
    n = 0
    while all(len(c) > n + 1 for c in candidates) and len({c[n] for c in candidates}) == 1:
        n += 1
    return candidates[0][:n]


def candidate_logps(model, query, candidates):
    """[len(candidates)] summed log-probabilities of each candidate after `query`."""
    inner, lm_head = _parts(model)
    cfg = inner.config
    device = lm_head.weight.device
    dtype = lm_head.weight.dtype
    common = _common_prefix(candidates)
    rests = [c[len(common):] for c in candidates]
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    empty = tuple((torch.empty(1, cfg.num_key_value_heads, 0, head_dim, dtype=dtype, device=device),) * 2
                  for _ in range(cfg.num_hidden_layers))
    inner._has_no_labels = True
    prefix = torch.tensor([query + common], device=device)
    with training_final_norm():
        out = inner(input_ids=prefix, causal_mask=fmha.attn_bias.LowerTriangularMask(),
                    past_key_values=empty, use_cache=True, return_dict=True)
    logits = lm_head(out.last_hidden_state[0]).float().log_softmax(-1)  # [L1, V]
    cache = [_kv(entry) for entry in out.past_key_values]
    if cache[0][0].shape[-2] != len(query) + len(common):
        raise ValueError(f"Unexpected cache shape {tuple(cache[0][0].shape)} for prefix length {len(query) + len(common)}")
    L1 = prefix.shape[1]
    # Common tokens and every candidate's first remaining token come from pass 1.
    common_logp = sum(logits[len(query) - 1 + i, t] for i, t in enumerate(common)) if common else 0.
    first = torch.stack([logits[L1 - 1, r[0]] for r in rests])
    # Remaining tokens of each candidate: teacher-forced against the shared cache.
    width = max(len(r) for r in rests) - 1
    if width == 0:
        return common_logp + first
    # Pad short continuations with their first token; causal masking keeps padding out of the scored positions.
    inputs = torch.tensor([r[:-1] + [r[0]] * (width - len(r) + 1) for r in rests], device=device)
    batch = len(rests)
    expanded = tuple((k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in cache)
    positions = torch.arange(L1, L1 + width, device=device)[None].expand(batch, -1)
    mask = None if width == 1 else fmha.attn_bias.LowerTriangularFromBottomRightMask()
    # use_cache=True keeps Unsloth off its offloaded-checkpointing path, which hands every layer
    # the whole cache list; the returned cache is discarded.
    with training_final_norm():
        out2 = inner(input_ids=inputs, causal_mask=mask, past_key_values=expanded, position_ids=positions,
                     use_cache=True, return_dict=True)
    logits2 = lm_head(out2.last_hidden_state).float().log_softmax(-1)  # [B, width, V]
    rest_logp = torch.stack([sum(logits2[b, j, r[j + 1]] for j in range(len(r) - 1)) if len(r) > 1
                             else logits2.new_zeros(()) for b, r in enumerate(rests)])
    return common_logp + first + rest_logp


def full_sequence_logps(model, query, candidates, pad):
    """The current method: 11 full sequences through the model's own forward."""
    seqs = [query + c for c in candidates]
    width = max(map(len, seqs))
    device = model.base_model.model.lm_head.weight.device
    input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
    logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
    out = []
    for k, c in enumerate(candidates):
        positions = torch.arange(len(query) - 1, len(query) - 1 + len(c), device=device)
        out.append(logits[k, positions].log_softmax(-1).gather(-1, torch.tensor(c, device=device)[:, None]).sum())
    return torch.stack(out)


def split_candidates(candidates):
    """Split ": k<eot>"-style candidates into (common prefix, one number token per k, stop token).

    Single-pass scoring needs every candidate to be common + [number] + [stop] with a distinct
    single-token number (true for Llama-3, where "10" is one token).
    """
    common = _common_prefix(candidates)
    rests = [c[len(common):] for c in candidates]
    if any(len(r) != 2 for r in rests) or len({r[1] for r in rests}) != 1 or len({r[0] for r in rests}) != len(rests):
        raise ValueError("single-pass scoring needs candidates of the form common + [number] + [stop]")
    return common, [r[0] for r in rests], rests[0][1]


def single_pass_logps(model, query, common, numbers, stop, k_star=None):
    """One ordinary forward pass over query + common (+ the sampled number k_star).

    Returns ([len(numbers)] log P(common) + log P(number_k), log P(stop | ..., number_{k_star}) or None).
    The number distribution comes from one softmax at the position after `common`; the stop check
    rides along as one extra position when a sampled k_star is given.
    """
    device = next(model.parameters()).device  # works for PEFT-wrapped and plain models
    tokens = query + common + ([numbers[k_star]] if k_star is not None else [])
    logits = model(input_ids=torch.tensor([tokens], device=device)).logits[0]
    n = len(query) + len(common)
    common_logp = sum(logits[len(query) - 1 + i].float().log_softmax(-1)[t] for i, t in enumerate(common)) if common else 0.
    number_logps = common_logp + logits[n - 1].float().log_softmax(-1)[torch.tensor(numbers, device=device)]
    stop_logp = logits[n].float().log_softmax(-1)[stop] if k_star is not None else None
    return number_logps, stop_logp


def single_pass_logps_batch(model, queries, common, numbers, stop, k_stars):
    """`single_pass_logps` for several rows in one forward pass.

    Rows are right-padded with the stop token. Unsloth's training forward uses no attention mask,
    which is safe here: attention is causal and every position read precedes the padding.
    Returns a list of (number log-probs, stop log-prob or None), one per row.
    """
    if not queries:
        return []
    device = next(model.parameters()).device
    seqs = [q + common + ([numbers[k]] if k is not None else []) for q, k in zip(queries, k_stars)]
    width = max(map(len, seqs))
    input_ids = torch.tensor([s + [stop] * (width - len(s)) for s in seqs], device=device)
    logits = model(input_ids=input_ids).logits
    number_index = torch.tensor(numbers, device=device)
    out = []
    for b, (q, k) in enumerate(zip(queries, k_stars)):
        n = len(q) + len(common)
        common_logp = sum(logits[b, len(q) - 1 + i].float().log_softmax(-1)[t] for i, t in enumerate(common)) if common else 0.
        number_logps = common_logp + logits[b, n - 1].float().log_softmax(-1)[number_index]
        stop_logp = logits[b, n].float().log_softmax(-1)[stop] if k is not None else None
        out.append((number_logps, stop_logp))
    return out


def end_of_turn(tokenizer):
    """The chat end-of-turn token: Llama-3's <|eot_id|>, or the tokenizer's EOS (Qwen's <|im_end|>)."""
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    return eot if eot is not None and eot != tokenizer.unk_token_id else tokenizer.eos_token_id


class LevelScheme:
    """The 11 confidence levels ": k" as tokens, and one-pass scoring for either tokenizer shape.

    Llama-3: every level is common + one number token ("10" is one token); `batch` is
    single_pass_logps_batch. Qwen-2.5 splits numbers into digits, so "10" is "1" "0" and shares
    its first token with level 1. Then one forward over query + common + ["1"] gives, at the
    number position, log P(d) for the ten digits and, one position later, log P("0" | "1"):
        log q(10) = log P(common) + log P("1") + log P("0" | "1")
        log q(1)  = log P(common) + log P("1") + log(1 - P("0" | "1"))
        log q(d)  = log P(common) + log P(d)            for the other digits
    so the 11 values still sum to the probability of common followed by a level, as in the
    one-token case. The stop check after the sampled level k* reads log P(stop) one position after
    k*'s last token: from the same sequence for k* = 1 (after "1") or 10 (append "0"), and from a
    second short sequence, query + common + [d], for the other digits.
    """

    def __init__(self, tokenizer, stop):
        self.stop = stop
        levels = [tokenizer.encode(f": {k}", add_special_tokens=False) for k in range(11)]
        self.common = _common_prefix(levels)
        self.suffixes = [lv[len(self.common):] for lv in levels]
        self.single = all(len(s) == 1 for s in self.suffixes)
        if self.single:
            self.numbers = [s[0] for s in self.suffixes]
        else:
            if not (all(len(s) == 1 for s in self.suffixes[:10]) and len(self.suffixes[10]) == 2
                    and self.suffixes[10][0] == self.suffixes[1][0]):
                raise ValueError(f"unsupported level tokenization: {self.suffixes}")
            self.digits = [s[0] for s in self.suffixes[:10]]
            self.one, self.zero = self.suffixes[10]

    def tokens(self, k):
        return self.suffixes[k]

    def parse(self, generated):
        """Level index from generated tokens after the query (common, then the level), or None."""
        c = len(self.common)
        if generated[:c] != self.common or len(generated) <= c:
            return None
        rest = generated[c:]
        if self.single:
            return self.numbers.index(rest[0]) if rest[0] in self.numbers else None
        if rest[0] == self.one and len(rest) > 1 and rest[1] == self.zero:
            return 10
        return self.digits.index(rest[0]) if rest[0] in self.digits else None

    def batch(self, model, queries, k_stars):
        """[(log q over the 11 levels, log P(stop | k*) or None)] for each query, one forward call."""
        if not queries:  # a batch whose answers were all malformed
            return []
        if self.single:
            return single_pass_logps_batch(model, queries, self.common, self.numbers, self.stop, k_stars)
        device = next(model.parameters()).device
        seqs, extra = [], {}
        for i, (q, k) in enumerate(zip(queries, k_stars)):
            seqs.append(q + self.common + [self.one] + ([self.zero] if k == 10 else []))
        for i, (q, k) in enumerate(zip(queries, k_stars)):
            if k is not None and k not in (1, 10):
                extra[i] = len(seqs)
                seqs.append(q + self.common + [self.digits[k]])
        width = max(map(len, seqs))
        logits = model(input_ids=torch.tensor([s + [self.stop] * (width - len(s)) for s in seqs], device=device)).logits
        digits = torch.tensor(self.digits, device=device)
        out = []
        for b, (q, k) in enumerate(zip(queries, k_stars)):
            n = len(q) + len(self.common)
            common_logp = sum(logits[b, len(q) - 1 + i].float().log_softmax(-1)[t] for i, t in enumerate(self.common))
            first = logits[b, n - 1].float().log_softmax(-1)
            after_one = logits[b, n].float().log_softmax(-1)
            log_zero = after_one[self.zero]
            log_not_zero = torch.log1p(-log_zero.exp().clamp(max=1 - 1e-7))
            levels = common_logp + first[digits]
            levels = torch.cat([levels[:1], (levels[1] + log_not_zero)[None], levels[2:], (levels[1] + log_zero)[None]])
            if k is None:
                stop_logp = None
            elif k == 1:
                stop_logp = after_one[self.stop]
            elif k == 10:
                stop_logp = logits[b, n + 1].float().log_softmax(-1)[self.stop]
            else:
                stop_logp = logits[extra[b], n].float().log_softmax(-1)[self.stop]
            out.append((levels, stop_logp))
        return out
