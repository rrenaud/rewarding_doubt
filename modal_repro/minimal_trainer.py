"""A minimal trainer for the exact confidence objective: Hugging Face transformers, PEFT and PyTorch only.

    python minimal_trainer.py ref CACHE.pt OUT.pt          # bf16 reference levels, untrained dev metrics
    python minimal_trainer.py diagnose CACHE.pt            # shared prefix vs full forward, bf16 and fp32
    python minimal_trainer.py answer-ref CACHE.pt OUT.pt   # reference top-k log-probs at every answer token
    python minimal_trainer.py train CACHE.pt OUT_DIR [--steps 300 --lr 3e-4 --attn-bias 18-35 --answer-kl 1 ...]

The data, objective, schedule and metrics are fast_loop.py's (docs/fast_loop_log.md) without Unsloth or TRL:
the model is bf16 with PyTorch SDPA attention, gradient checkpointing is off, and inputs never require grad,
so backward stops by itself at the first layer with anything to train.

Scoring: the 11 levels ": k" from one forward pass, with logits only at the positions they read (as
fast_loop.score_rows). Qwen-2.5 splits "10" into "1" "0", so one extra position gives P("0" | "1"):
    log q(10) = log P(common) + log P("1") + log P("0" | "1")
    log q(1)  = log P(common) + log P("1") + log(1 - P("0" | "1"))
    log q(d)  = log P(common) + log P(d)            for the other digits.

Shared prefix: about 132 of the ~164 tokens of a training row are a prompt prefix common to every row. Each
minibatch runs that prefix once (batch 1, with grad, since adapters change it), expands its key/value cache
over the batch and runs only the suffixes. In fp32 this matches a full forward to 3e-3 nats, the same as a
batched full forward against one row at a time; in bf16 both comparisons differ by up to ~2 nats on some
rows (`diagnose`), so bf16 scoring carries that much noise whichever way it is computed.

ref: the cache's reference levels came from the 4-bit Unsloth model, which differs from bf16 by up to 10 nats;
this recomputes them with the bf16 model (no adapters) and saves a copy of the cache with them.

Answer drift: the adapters act at every position, so they can move the answers too, while the objective
trains only the confidence. The same forward pass gives the log-softmax at each answer token, and the
reference is the model with adapters off (adapters_off): dev metrics report KL(policy || reference) over the
vocabulary per answer (answer_kl), --answer-kl W adds W times it to each row's loss, and --regen-every N
answers the dev questions again with the policy and grades them (regen_accuracy; needs --gt).

answer-ref stores the reference's top-k log-probs at every cached answer token, so the penalty needs no
reference pass: topk_answer_kl is the KL between the two distributions coarse-grained to those k tokens
plus one bucket for the rest, which is never more than the exact KL. --answer-kl-target T adapts W after
each step to hold the batch's answer KL near T.
"""
import argparse
import collections
import contextlib
import json
import math
import os
import random
import re
import signal
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from rewarding_doubt.checkpoint import (announce_saved, load_checkpoint, rng_state, save_checkpoint, set_rng_state,
                                        truncate_jsonl, write_status)
from rewarding_doubt.core import baseline_matched_objective
from rewarding_doubt.paper_ppo import AdaptiveKLController, evaluation_metrics, is_correct_exact, is_correct_f1

LORA_MODULES = ("q", "k", "v", "o", "gate", "up", "down")
# The released util.ResponseHandling.parse_answer_confidence, applied to the decoded answer + ": 0".
ANSWER_PATTERN = re.compile(r"Answer:\s*(?P<answer>.*?),\s*Confidence:\s*(?P<confidence>\d+)")
_biases_on = [True]


class Levels:
    """The 11 confidence levels ": k" as tokens: a common prefix, then one digit, or "1" "0" for 10."""

    def __init__(self, tokenizer):
        levels = [tokenizer.encode(f": {k}", add_special_tokens=False) for k in range(11)]
        self.common = os.path.commonprefix(levels)
        suffixes = [lv[len(self.common):] for lv in levels]
        self.single = all(len(s) == 1 for s in suffixes)
        if self.single:  # Llama-3: "10" is one token
            self.numbers, self.tail = [s[0] for s in suffixes], []
        else:
            if not (all(len(s) == 1 for s in suffixes[:10]) and suffixes[10] == [suffixes[1][0], suffixes[10][1]]):
                raise ValueError(f"unsupported level tokenization: {suffixes}")
            self.numbers, (one, self.zero) = [s[0] for s in suffixes[:10]], suffixes[10]
            self.tail = [one]
        self.reads = len(self.common) + 1 + len(self.tail)  # positions whose logits are read

    def from_logp(self, logp):
        """[B, reads, V] log-softmax at the read positions -> [B, 11] log q."""
        c = len(self.common)
        common = torch.tensor(self.common, device=logp.device)
        common_logp = logp[:, :c].gather(-1, common[None, :, None].expand(len(logp), c, 1)).sum((1, 2))
        levels = common_logp[:, None] + logp[:, c, self.numbers]
        if self.single:
            return levels
        log_zero = logp[:, c + 1, self.zero]
        log_not_zero = torch.log1p(-log_zero.exp().clamp(max=1 - 1e-7))
        return torch.cat([levels[:, :1], levels[:, 1:2] + log_not_zero[:, None], levels[:, 2:], levels[:, 1:2] + log_zero[:, None]], 1)


def score_rows(lm, levels, queries, answer_starts, pad, shared_prefix=True):
    """One forward pass over each query + the levels' tokens, with logits only at the positions read:
    (levels [B, 11] log q, answers [B][T_b, V]). answers[b] is the log-softmax at the positions predicting
    query b's tokens from answer_starts[b] on, through the final " Confidence" (empty when it is len(query)).
    With shared_prefix, the queries' common prefix runs once and its key/value cache is expanded over the
    batch; otherwise one full forward."""
    device = lm.lm_head.weight.device
    seqs = [q + levels.common + levels.tail for q in queries]
    # The prefix stops before the first position read: the shortest query's last token or the earliest answer.
    n = min(len(os.path.commonprefix(queries)), min(map(len, queries)) - 1, min(answer_starts) - 1) if shared_prefix else 0
    cache = None
    if n:
        prefix = lm.model(input_ids=torch.tensor([queries[0][:n]], device=device), use_cache=True).past_key_values
        cache = DynamicCache.from_legacy_cache(tuple((k.expand(len(queries), -1, -1, -1), v.expand(len(queries), -1, -1, -1))
                                                     for k, v in prefix.to_legacy_cache()))
    width = max(map(len, seqs)) - n
    # Right padding is safe without a mask: attention is causal and every read position precedes the padding.
    ids = torch.tensor([s[n:] + [pad] * (width - len(s) + n) for s in seqs], device=device)
    hidden = lm.model(input_ids=ids, past_key_values=cache, use_cache=cache is not None).last_hidden_state
    rows = torch.arange(len(queries), device=device)[:, None]
    pos = torch.tensor([[len(q) - 1 - n + i for i in range(levels.reads)] for q in queries], device=device)
    spans = [range(start - 1 - n, len(q) - 1 - n) for q, start in zip(queries, answer_starts)]
    flat_rows = torch.tensor([b for b, span in enumerate(spans) for _ in span], device=device, dtype=torch.long)
    flat_pos = torch.tensor([t for span in spans for t in span], device=device, dtype=torch.long)
    answers = lm.lm_head(hidden[flat_rows, flat_pos]).float().log_softmax(-1).split([len(span) for span in spans])
    return levels.from_logp(lm.lm_head(hidden[rows, pos]).float().log_softmax(-1)), list(answers)


def answer_start(ids, tokenizer):
    """Index of a cached row's first answer token: right after the last "<|im_start|>assistant\\n" (rows are
    the chat prompt, which ends with that header, then the generated answer)."""
    start = len(ids) - 1 - ids[::-1].index(tokenizer.convert_tokens_to_ids("<|im_start|>")) + 3
    assert tokenizer.decode(ids[start - 3:start]) == "<|im_start|>assistant\n", tokenizer.decode(ids[start - 3:start])
    return start


def answer_kl(policy, reference):
    """KL(policy || reference) over the vocabulary, summed over each answer's positions: [B]."""
    return torch.stack([(p.exp() * (p - r)).sum() for p, r in zip(policy, reference)])


def topk_answer_kl(policy, rows):
    """answer_kl against the cached reference top-k (answer-ref): at each position the two distributions are
    coarse-grained to the reference's k most likely tokens plus one bucket for all others, which can only
    lower the KL (it is at most the exact one). [B]."""
    out = []
    for p, r in zip(policy, rows):
        idx, ref = r["answer_ref_idx"].to(p.device).long(), r["answer_ref_logp"].to(p.device)
        top = p.gather(-1, idx)
        rest_p = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
        rest_r = torch.log1p(-ref.exp().sum(-1).clamp(max=1 - 1e-6))
        out.append((top.exp() * (top - ref)).sum() + (rest_p.exp() * (rest_p - rest_r)).sum())
    return torch.stack(out)


@contextlib.contextmanager
def adapters_off(lm):
    """The reference model: LoRA layers disabled and the output-bias hooks passing outputs through."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    lora = [m for m in lm.modules() if isinstance(m, BaseTunerLayer)]
    for m in lora:
        m.enable_adapters(False)
    _biases_on[0] = False
    try:
        yield
    finally:
        for m in lora:
            m.enable_adapters(True)  # also sets requires_grad again; adapters exist only where they train
        _biases_on[0] = True


def load(model_name, device="cuda"):
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                              device_map={"": device})
    lm.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name)
    return lm, tok


def add_lora(lm, modules, layers):
    """Rank-8 LoRA (alpha 8, no dropout, no bias) on `modules` in decoder `layers`; adapters elsewhere would
    be frozen at B = 0 and change nothing, so they are not created. PEFT keeps adapter weights in fp32
    (the fast loop's were bf16: TRL's peft_module_casting_to_bf16)."""
    from peft import LoraConfig, get_peft_model
    if not modules or not layers:
        return []
    config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=[f"{m}_proj" for m in modules],
                        layers_to_transform=list(layers), layers_pattern="layers")
    get_peft_model(lm, config)  # injects the adapters into lm's modules in place
    return [p for n, p in lm.named_parameters() if "lora_" in n]


def add_output_biases(lm, layers, part):
    """A trainable fp32 vector added to the output of `part` ("self_attn" or "mlp"), that is to its write to
    the residual stream, in each of `layers`; zero-initialized, by forward hooks. Returns the parameters."""
    params = []
    for i in layers:
        module = getattr(lm.model.layers[i], part)
        bias = torch.nn.Parameter(torch.zeros(lm.config.hidden_size, device=lm.lm_head.weight.device))

        def hook(module, inputs, out, bias=bias):
            if not _biases_on[0]:  # adapters_off: the output passes through unchanged
                return None
            if isinstance(out, tuple):  # attention returns (output, weights)
                return (out[0] + bias.to(out[0].dtype), *out[1:])
            return out + bias.to(out.dtype)
        module.register_forward_hook(hook)
        params.append(bias)
    return params


def layer_range(spec, n_layers):
    if spec == "none":
        return range(0)
    if spec == "all":
        return range(n_layers)
    lo, hi = map(int, spec.split("-"))
    return range(lo, hi + 1)


def dev_metrics_from_levels(levels, labels, temperature=0.6, seed=0):
    """Metrics of the expected and of a T=0.6-sampled confidence, from per-answer level log-probs (fast_loop's)."""
    gen = torch.Generator().manual_seed(seed)
    expected, sampled, masses = [], [], []
    for lv, y in zip(levels, labels):
        lv = lv.double()
        q = lv.softmax(-1)
        masses.append(float(lv.logsumexp(-1).exp()))
        expected.append(dict(confidence=float((q * torch.arange(11, dtype=q.dtype)).sum()), correct=y))
        k = int(torch.multinomial((lv / temperature).softmax(-1), 1, generator=gen))
        sampled.append(dict(confidence=k, correct=y))
    out = {f"{k}_expected": v for k, v in evaluation_metrics(expected).items() if k in ("ece", "auroc", "brier")}
    out.update({f"{k}_sampled": v for k, v in evaluation_metrics(sampled).items() if k in ("ece", "auroc", "brier")})
    out.update(accuracy=statistics.fmean(labels), level_mass=statistics.fmean(masses), n=len(labels))
    return out


@torch.no_grad()
def score(lm, levels, rows, pad, chunk=64):
    """Level log-probs of every row, no answers."""
    out = []
    for s in range(0, len(rows), chunk):
        queries = [r["ids"] for r in rows[s:s + chunk]]
        out += list(score_rows(lm, levels, queries, list(map(len, queries)), pad)[0].float().cpu())
    return out


@torch.no_grad()
def score_dev(lm, levels, rows, pad, chunk=32):
    """Dev metrics of the stated confidence, plus KL(policy || reference) on the cached answers, per answer
    (answer_kl, against a live reference pass) and per token (answer_kl_token), and against the cached
    top-k reference when the cache has it (answer_kl_topk)."""
    lvs, kls, topk, tokens = [], [], [], 0
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        queries, starts = [r["ids"] for r in part], [r["answer_start"] for r in part]
        lv, answers = score_rows(lm, levels, queries, starts, pad)
        with adapters_off(lm):
            reference = score_rows(lm, levels, queries, starts, pad)[1]
        lvs += list(lv.float().cpu())
        kls += answer_kl(answers, reference).tolist()
        if "answer_ref_idx" in part[0]:
            topk += topk_answer_kl(answers, part).tolist()
        tokens += sum(len(a) for a in answers)
    metrics = dict(dev_metrics_from_levels(lvs, [r["f1"] for r in rows]), answer_kl=statistics.fmean(kls), answer_kl_token=sum(kls) / tokens)
    if topk:
        metrics["answer_kl_topk"] = statistics.fmean(topk)
    return metrics


@torch.no_grad()
def generate_answers(lm, tokenizer, prompts, seed):
    """Answers to `prompts` with the current model, the released way (T=0.6, top-p 0.9, up to 96 tokens, stopping
    at " Confidence"), sampled with the given seed: per prompt (tokens, answer text), answer text None when the
    answer never reached " Confidence" or does not match the released pattern."""
    pad, confidence = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("ĠConfidence")
    width = max(map(len, prompts))
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=lm.lm_head.weight.device)
    mask = (torch.arange(width, device=ids.device)[None, :] >= torch.tensor([width - len(p) for p in prompts], device=ids.device)[:, None]).long()
    torch.manual_seed(seed)
    out = lm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=96, do_sample=True, temperature=0.6, top_p=0.9,
                      eos_token_id=[pad, tokenizer.convert_tokens_to_ids("<|im_end|>"), confidence], pad_token_id=pad)
    answers = []
    for row in out[:, width:].tolist():
        while row and row[-1] == pad:
            row.pop()
        match = ANSWER_PATTERN.search(tokenizer.decode(row, skip_special_tokens=True) + ": 0") if row and row[-1] == confidence else None
        answers.append((row, match.group("answer") if match else None))
    return answers


def regenerate_dev(lm, tokenizer, rows, gt_candidates, batch=64):
    """Answer the dev questions again with the current model (generate_answers, seeded per batch, so models are
    compared on the same draws) and grade with F1 > 0.5: regen_accuracy over all questions, regen_malformed
    for answers that never reached " Confidence" or did not parse."""
    correct = malformed = 0
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        for r, (_, answer) in zip(part, generate_answers(lm, tokenizer, [r["ids"][:r["answer_start"]] for r in part], s)):
            if answer is None:
                malformed += 1
            else:
                correct += bool(is_correct_f1(answer, gt_candidates[str(r["qid"])]))
    return dict(regen_accuracy=correct / len(rows), regen_malformed=malformed / len(rows))


@torch.no_grad()
def online_rows(lm, tokenizer, levels, batch, gt_candidates, seed, pad):
    """--online: the batch's questions answered by the current model (generate_answers) and graded; answers that
    are malformed are dropped, as the exact objective gives them no gradient. Each kept row gets its reference
    from one pass with adapters off: levels (ref) and the full answer log-probs (answer_ref_live).
    Returns (rows, {online_accuracy, online_malformed, online_answer_tokens})."""
    prompts = [r["ids"][:r["answer_start"]] for r in batch]
    rows = []
    for r, prompt, (tokens, answer) in zip(batch, prompts, generate_answers(lm, tokenizer, prompts, seed)):
        if answer is not None:
            gt = gt_candidates[str(r["qid"])]
            rows.append(dict(qid=r["qid"], ids=prompt + tokens, answer_start=len(prompt),
                             f1=bool(is_correct_f1(answer, gt)), em=bool(is_correct_exact(answer, gt))))
    stats = dict(online_accuracy=sum(r["f1"] for r in rows) / len(batch), online_malformed=1 - len(rows) / len(batch),
                 online_answer_tokens=statistics.fmean(len(r["ids"]) - r["answer_start"] for r in rows) if rows else 0.0)
    if rows:
        with adapters_off(lm):
            ref_levels, ref_answers = score_rows(lm, levels, [r["ids"] for r in rows], [r["answer_start"] for r in rows], pad)
        for r, lv, a in zip(rows, ref_levels, ref_answers):
            r["ref"], r["answer_ref_live"] = lv.float(), a
    return rows, stats


def check(lm, levels, rows, pad, params=()):
    """Shared-prefix against full-forward scoring on `rows`: max |difference| of the level log-probs, and of
    the gradients of their sum with respect to `params` (relative to the largest gradient entry)."""
    out = {}
    grads = []
    for shared in (True, False):
        for p in params:
            p.grad = None
        queries = [r["ids"] for r in rows]
        lv = score_rows(lm, levels, queries, list(map(len, queries)), pad, shared_prefix=shared)[0]
        if params:
            lv.sum().backward()
            grads.append(torch.cat([p.grad.flatten() for p in params]))
        out[shared] = lv.detach().float()
    result = dict(rows=len(rows), max_abs_level_diff=float((out[True] - out[False]).abs().max()))
    if params:
        result["max_rel_grad_diff"] = float((grads[0] - grads[1]).abs().max() / grads[1].abs().max())
    return result


def diagnose(args):
    """Where differences between scorings come from, on the first --rows dev rows: shared prefix vs full
    forward, a batched full forward vs one row at a time (batching noise), in bf16 and fp32; and the cache's
    reference vs this model. Each comparison: max |diff| of log q over all levels and over levels with
    q > 1e-3, max |diff| of q, and max |diff| of the expected confidence."""
    cache = torch.load(args.cache, weights_only=False)
    rows = cache["splits"]["dev"][:args.rows]
    queries = [r["ids"] for r in rows]

    def compare(a, b):
        a, b = a.double().cpu(), b.double().cpu()
        qa, qb = a.exp(), b.exp()
        big = (qb > 1e-3)
        k = torch.arange(11, dtype=torch.double)
        return dict(max_abs_logq=float((a - b).abs().max()), max_abs_logq_q_gt_1e3=float((a - b).abs()[big].max()),
                    max_abs_q=float((qa - qb).abs().max()),
                    max_abs_expected=float(((qa / qa.sum(-1, keepdim=True)) @ k - (qb / qb.sum(-1, keepdim=True)) @ k).abs().max()))
    for dtype in (torch.bfloat16, torch.float32):
        lm = AutoModelForCausalLM.from_pretrained(cache["model"], torch_dtype=dtype, attn_implementation="sdpa").cuda()
        tok = AutoTokenizer.from_pretrained(cache["model"])
        levels, pad = Levels(tok), tok.eos_token_id
        with torch.no_grad():
            lens = list(map(len, queries))
            full = score_rows(lm, levels, queries, lens, pad, shared_prefix=False)[0]
            shared = score_rows(lm, levels, queries, lens, pad)[0]
            single = torch.cat([score_rows(lm, levels, [q], [len(q)], pad, shared_prefix=False)[0] for q in queries])
        print(json.dumps(dict(dtype=str(dtype), shared_vs_full=compare(shared, full), batched_vs_single=compare(full, single),
                              shared_vs_single=compare(shared, single),
                              cache_ref_vs_single=compare(torch.stack([r["ref"] for r in rows]), single))), flush=True)
        del lm
        torch.cuda.empty_cache()


def memory(args):
    """Peak GPU memory of one training update on a bucketed batch, after forward and after backward, for all-layer
    LoRA with and without the shared prefix."""
    cache = torch.load(args.cache, weights_only=False)
    lm, tok = load(cache["model"])
    params = add_lora(lm, list(LORA_MODULES), range(lm.config.num_hidden_layers))
    levels, pad = Levels(tok), tok.eos_token_id
    rows = cache["splits"]["train"]
    batch = [rows[i] for i in epoch_batches([len(r["ids"]) for r in rows], 32, 64, random.Random(1))[0]]
    queries = [r["ids"] for r in batch]
    print(json.dumps(dict(weights_gb=round(torch.cuda.memory_allocated() / 2**30, 2), prefix=len(os.path.commonprefix(queries)),
                          lengths=[min(map(len, queries)), max(map(len, queries))])), flush=True)
    for shared in (True, False):
        torch.cuda.reset_peak_memory_stats()
        lv = score_rows(lm, levels, queries, list(map(len, queries)), pad, shared_prefix=shared)[0]
        after_forward = torch.cuda.max_memory_allocated() / 2**30
        lv.logsumexp(-1).sum().backward()
        for p in params:
            p.grad = None
        print(json.dumps(dict(shared_prefix=shared, peak_after_forward_gb=round(after_forward, 2),
                              peak_after_backward_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))), flush=True)
        del lv
        torch.cuda.empty_cache()


def make_ref(args):
    cache = torch.load(args.cache, weights_only=False)
    lm, tok = load(cache["model"])
    levels = Levels(tok)
    dev = cache["splits"]["dev"]
    for split, rows in cache["splits"].items():
        t0 = time.time()
        old = torch.stack([r["ref"] for r in rows])
        new = score(lm, levels, rows, tok.eos_token_id)
        for r, lv in zip(rows, new):
            r["ref"] = lv
        diff = (torch.stack(new) - old).abs()
        print(json.dumps(dict(split=split, rows=len(rows), seconds=round(time.time() - t0, 1),
                              vs_4bit_ref_mean_abs=float(diff.mean()), vs_4bit_ref_max_abs=float(diff.max()))), flush=True)
    print(json.dumps(dict(untrained_bf16_dev=dev_metrics_from_levels([r["ref"] for r in dev], [r["f1"] for r in dev]))), flush=True)
    cache.update(ref_model=f"{cache['model']} bf16 (minimal_trainer.py ref)", ref_created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    torch.save(cache, args.out)


def make_answer_ref(args):
    """The bf16 reference model's top-k log-probs at every cached answer token, saved into a copy of the
    cache (answer_ref_idx [T, k] int32, answer_ref_logp [T, k]); reports how much mass the top k cover."""
    cache = torch.load(args.cache, weights_only=False)
    lm, tok = load(cache["model"])
    levels, pad = Levels(tok), tok.eos_token_id
    for split, rows in cache["splits"].items():
        t0, covered = time.time(), []
        for s in range(0, len(rows), 32):
            part = rows[s:s + 32]
            for r in part:
                r["answer_start"] = answer_start(r["ids"], tok)
            with torch.no_grad():
                answers = score_rows(lm, levels, [r["ids"] for r in part], [r["answer_start"] for r in part], pad)[1]
            for r, a in zip(part, answers):
                logp, idx = a.topk(args.k, dim=-1)
                r["answer_ref_logp"], r["answer_ref_idx"] = logp.cpu(), idx.to(torch.int32).cpu()
                covered += logp.exp().sum(-1).tolist()
        covered.sort()
        print(json.dumps(dict(split=split, rows=len(rows), seconds=round(time.time() - t0, 1), k=args.k,
                              topk_mass_mean=statistics.fmean(covered), topk_mass_p01=covered[len(covered) // 100],
                              topk_mass_min=covered[0])), flush=True)
    cache.update(answer_ref_k=args.k)
    torch.save(cache, args.out)


def epoch_batches(lengths, batchsize, window, rng):
    """One epoch of batches (every row once, ragged tail dropped). With window > 0, length bucketing:
    the shuffled epoch is cut into windows of `window` batches, each window is sorted by length before
    being cut into batches, and the batch order is shuffled (fast_loop's)."""
    order = rng.sample(range(len(lengths)), len(lengths))
    if window > 0:
        span = window * batchsize
        order = [i for s in range(0, len(order), span) for i in sorted(order[s:s + span], key=lengths.__getitem__)]
    batches = [order[s:s + batchsize] for s in range(0, len(order) - batchsize + 1, batchsize)]
    rng.shuffle(batches)
    return batches


def train(args):
    cache = torch.load(args.cache, weights_only=False)
    lm, tok = load(cache["model"])
    pad = tok.eos_token_id
    n_layers = lm.config.num_hidden_layers
    modules = [] if args.lora_modules == "none" else args.lora_modules.split(",")
    assert set(modules) <= set(LORA_MODULES), modules
    torch.manual_seed(args.seed)
    lora_layers, attn_layers, mlp_layers = (layer_range(s, n_layers) for s in (args.lora_layers, args.attn_bias, args.residual_bias))
    trained = dict(lora=add_lora(lm, modules, lora_layers), attn_bias=add_output_biases(lm, attn_layers, "self_attn"),
                   residual_bias=add_output_biases(lm, mlp_layers, "mlp"))
    params = [p for group in trained.values() for p in group]
    print(json.dumps(dict(lora_modules=modules, lora_layers=[lora_layers.start, lora_layers.stop - 1] if modules else None,
                          attn_bias=args.attn_bias, residual_bias=args.residual_bias, trainable_params=sum(p.numel() for p in params),
                          shared_prefix=not args.no_shared_prefix)), flush=True)
    levels = Levels(tok)
    rng = random.Random(args.seed)
    train_rows, dev_rows = cache["splits"]["train"], cache["splits"]["dev"]
    for r in train_rows + dev_rows:
        r["answer_start"] = answer_start(r["ids"], tok)
    gt_candidates = json.load(open(args.gt)) if args.regen_every or args.online else None
    # Adam as in the released code (TRL 0.8.6 PPOTrainer: torch.optim.Adam, default betas, no weight decay).
    optimizer = torch.optim.Adam(params, lr=args.lr)
    kl_ctl = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon) if args.adaptive_kl else None
    beta = args.kl_coef
    answer_w = args.answer_kl
    if answer_w and args.answer_ref == "topk" and not args.online and "answer_ref_idx" not in train_rows[0]:
        raise SystemExit("--answer-ref topk needs a cache from `minimal_trainer.py answer-ref`")
    if args.answer_kl_target and not answer_w:
        raise SystemExit("--answer-kl-target adapts --answer-kl, which must start above 0")

    def dev_metrics(step):
        metrics = score_dev(lm, levels, dev_rows, pad)
        if args.regen_every and step % args.regen_every == 0:
            metrics.update(regenerate_dev(lm, tok, dev_rows, gt_candidates))
        return metrics
    # Every trained tensor by a stable name, for checkpoints: PEFT's LoRA names, and the bias vectors by layer.
    named = {n: p for n, p in lm.named_parameters() if "lora_" in n}
    named.update({f"attn_bias.{i}": p for i, p in zip(attn_layers, trained["attn_bias"])})
    named.update({f"residual_bias.{i}": p for i, p in zip(mlp_layers, trained["residual_bias"])})
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_dir, metrics_path = os.path.join(args.out_dir, "checkpoint"), os.path.join(args.out_dir, "metrics.jsonl")
    # Resume (--checkpoint-every): rerunning the same command continues from the last checkpoint.
    state = load_checkpoint(checkpoint_dir) if args.checkpoint_every and not args.no_resume else None
    phase_seconds = collections.Counter()
    if state is None:
        log = open(metrics_path, "w")
        curve, batches, start, elapsed = {0: dev_metrics(0)}, [], 1, 0.0
        print(json.dumps(dict(step=0, **curve[0])), flush=True)
    else:
        with torch.no_grad():
            for n, value in state["params"].items():
                named[n].copy_(value.to(named[n].device, named[n].dtype))
        optimizer.load_state_dict(state["optimizer"])
        set_rng_state(state["rng"])
        rng.setstate(state["rng_local"])
        curve, batches, start, elapsed = state["curve"], state["batches"], state["step"] + 1, state["seconds"]
        beta, answer_w = state["beta"], state["answer_w"]
        if kl_ctl is not None:
            kl_ctl.value = beta
        phase_seconds.update(state["phase_seconds"])
        truncate_jsonl(metrics_path, state["step"])
        log = open(metrics_path, "a")
        print(f"resumed from {checkpoint_dir} at step {state['step']}", flush=True)
    t0 = time.time() - elapsed
    last = [time.time()]
    stopping = []  # SIGTERM (a preempted pod or container): checkpoint after the current step and exit 143
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append("SIGTERM"))

    def save(step):
        save_checkpoint(checkpoint_dir, dict(
            step=step, seconds=time.time() - t0, params={n: p.detach().cpu().clone() for n, p in named.items()},
            optimizer=optimizer.state_dict(), rng=rng_state(), rng_local=rng.getstate(), batches=batches, curve=curve,
            beta=beta, answer_w=answer_w, phase_seconds=dict(phase_seconds)))
        announce_saved(checkpoint_dir)

    def lap(phase=None):
        """--time-steps: accumulate seconds per phase of an update (synchronizes the GPU, so slightly slower)."""
        if not args.time_steps:
            return
        torch.cuda.synchronize()
        now = time.time()
        if phase:
            phase_seconds[phase] += now - last[0]
        last[0] = now
    for step in range(start, args.steps + 1):
        if not batches:
            batches = epoch_batches([len(r["ids"]) for r in train_rows], args.batchsize, args.bucket_window, rng)
        batch = [train_rows[i] for i in batches.pop()]
        online = {}
        if args.online:  # the questions answered now by the policy instead of the cached base-model answers
            lap()
            batch, online = online_rows(lm, tok, levels, batch, gt_candidates, step, pad)
            lap("generate")
            if not batch:
                continue
        stats = []
        for p in range(args.passes):
            idx = list(range(len(batch)))
            rng.shuffle(idx)
            for mb in range(0, len(idx), args.minibatch):
                chunk = [batch[i] for i in idx[mb:mb + args.minibatch]]
                lap()
                queries, starts = [r["ids"] for r in chunk], [r["answer_start"] for r in chunk]
                scored, answers = score_rows(lm, levels, queries, starts, pad, shared_prefix=not args.no_shared_prefix)
                if not answer_w:
                    kls = torch.zeros(len(chunk), device=scored.device)
                elif args.online:  # the reference pass ran with the generation (online_rows)
                    kls = answer_kl(answers, [r["answer_ref_live"] for r in chunk])
                elif args.answer_ref == "topk":
                    kls = topk_answer_kl(answers, chunk)
                else:
                    with torch.no_grad(), adapters_off(lm):
                        reference = score_rows(lm, levels, queries, starts, pad, shared_prefix=not args.no_shared_prefix)[1]
                    kls = answer_kl(answers, reference)
                lap("forward")
                losses = []
                for r, lv, a_kl in zip(chunk, scored, kls):
                    J, kl = baseline_matched_objective(lv.double(), float(r[args.grading]), "discrete-exact", "released",
                                                       -30.0, ref_logq=r["ref"].to(lv.device).double())
                    losses.append(-(J - beta * kl) + answer_w * a_kl)
                    if p == 0:
                        stats.append((J.item(), kl.item(), float((lv.double().softmax(-1) * torch.arange(11, device=lv.device)).sum()),
                                      float(a_kl)))
                lap("loss")
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                lap("backward")
                optimizer.step()
                lap("optimizer")
        if kl_ctl is not None:
            kl_ctl.update(statistics.fmean(s[1] for s in stats), args.batchsize)
            beta = kl_ctl.value
        if args.answer_kl_target:  # multiplicative: W *= exp(rate * clip(KL / T - 1, -1, 1))
            error = statistics.fmean(s[3] for s in stats) / args.answer_kl_target - 1
            answer_w *= math.exp(args.answer_kl_rate * max(-1.0, min(1.0, error)))
        record = dict(step=step, answer_w=answer_w, **online, expected_reward=statistics.fmean(s[0] for s in stats), kl=statistics.fmean(s[1] for s in stats),
                      mean_confidence=statistics.fmean(s[2] for s in stats), answer_kl=statistics.fmean(s[3] for s in stats),
                      beta=beta, seconds=time.time() - t0,
                      **({"phase_seconds": dict(phase_seconds)} if args.time_steps else {}))
        log.write(json.dumps(record) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            curve[step] = dev_metrics(step)
            print(json.dumps(dict(step=step, minutes=round((time.time() - t0) / 60, 1), beta=round(beta, 4),
                                  peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), **curve[step])), flush=True)
            log.flush()
        if step == args.stop_after:  # testing: behave as if preempted here
            stopping.append("--stop-after")
        if args.checkpoint_every and (step % args.checkpoint_every == 0 or stopping):
            log.flush()
            save(step)
        if stopping:
            print(f"stopping after step {step}: {stopping[0]}", flush=True)
            sys.exit(143)
    json.dump({str(k): v for k, v in curve.items()}, open(os.path.join(args.out_dir, "curve.json"), "w"), indent=1)
    write_status(args.out_dir, "completed", step=args.steps)
    if args.save:
        torch.save(dict(args=vars(args), lora={n: p.detach().cpu() for n, p in lm.named_parameters() if "lora_" in n},
                        attn_bias=[p.detach().cpu() for p in trained["attn_bias"]],
                        residual_bias=[p.detach().cpu() for p in trained["residual_bias"]]), os.path.join(args.out_dir, "trained.pt"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("ref")
    r.add_argument("cache"); r.add_argument("out")
    d = sub.add_parser("diagnose")
    d.add_argument("cache")
    d.add_argument("--rows", type=int, default=64)
    a = sub.add_parser("answer-ref")
    a.add_argument("cache"); a.add_argument("out")
    a.add_argument("--k", type=int, default=64)
    m = sub.add_parser("memory")
    m.add_argument("cache")
    t = sub.add_parser("train")
    t.add_argument("cache"); t.add_argument("out_dir")
    # Defaults: fast_loop's schedule. The released one is --batchsize 8 --passes 4 --minibatch 4 --lr 1e-5.
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batchsize", type=int, default=32)
    t.add_argument("--passes", type=int, default=1)
    t.add_argument("--minibatch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--kl-coef", type=float, default=0.05)
    t.add_argument("--adaptive-kl", action="store_true", help="the released controller: target 6, horizon 10000")
    t.add_argument("--kl-target", type=float, default=6.0)
    t.add_argument("--kl-horizon", type=float, default=10000.0)
    t.add_argument("--grading", choices=["em", "f1"], default="em", help="training label (released default: exact match)")
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--lora-modules", default=",".join(LORA_MODULES), help='projections with LoRA adapters, or "none"')
    t.add_argument("--lora-layers", default="all", help='"all" or an inclusive range of decoder layers, e.g. "18-35"')
    t.add_argument("--attn-bias", default="none", help='"none", "all" or a layer range: train a vector added to each attention output')
    t.add_argument("--residual-bias", default="none", help='"none", "all" or a layer range: train a vector added to each MLP output')
    t.add_argument("--bucket-window", type=int, default=64, help="batches per length-sorting window; 0 = random batches")
    t.add_argument("--no-shared-prefix", action="store_true", help="full forward per row instead of a shared prefix cache")
    t.add_argument("--answer-kl", type=float, default=0.0,
                   help="weight of KL(policy || reference) over the vocabulary, summed over each answer's tokens (reference: adapters off)")
    t.add_argument("--answer-ref", choices=["topk", "live"], default="topk",
                   help="reference for --answer-kl: the cache's top-k (answer-ref, no extra pass) or a live pass with adapters off")
    t.add_argument("--answer-kl-target", type=float, default=0.0, help="adapt the --answer-kl weight to hold the answer KL near this (nats per answer)")
    t.add_argument("--answer-kl-rate", type=float, default=0.05, help="per-step log change of the weight at full error")
    t.add_argument("--regen-every", type=int, default=0, help="also answer the dev questions again and grade them every N steps (and at 0)")
    t.add_argument("--gt", default="", help="JSON {question_id: gt_candidates} for --regen-every and --online (fast_loop.py gt)")
    t.add_argument("--online", action="store_true",
                   help="train on answers the policy generates for each batch (graded with --gt) instead of the cached ones")
    t.add_argument("--time-steps", action="store_true", help="log cumulative seconds per update phase (adds GPU syncs)")
    t.add_argument("--save", action="store_true", help="save the trained parameters to OUT_DIR/trained.pt")
    t.add_argument("--checkpoint-every", type=int, default=0,
                   help="resumable checkpoint every N steps and on SIGTERM (exit 143); rerunning the same command resumes")
    t.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    t.add_argument("--stop-after", type=int, default=0, help="testing: checkpoint and exit 143 after this step, as if preempted")
    args = parser.parse_args()
    {"ref": make_ref, "answer-ref": make_answer_ref, "diagnose": diagnose, "memory": memory, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
