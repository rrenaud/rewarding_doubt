"""A fast, approximate loop for iterating on confidence objectives (no generation after caching).

    python fast_loop.py cache IDS_JSON OUT.pt --model M --n-train 8000
    python fast_loop.py train CACHE.pt OUT_DIR [--steps 300 --lr 3e-4 --kl-coef 0.05 --adaptive-kl ...]
    python fast_loop.py eval CACHE.pt ADAPTER_DIR OUT.json

cache: the base model answers each question once, as the released training does (released prompt,
T=0.6, top-p 0.9, stopping at " Confidence"), for a pool of training questions and the dev split.
Each answer is graded (F1 > 0.5 and exact match) and stored as tokens with the reference model's
log-probabilities of the 11 confidence levels (LevelScheme, adapter-free base model).

train: answers stay frozen at the cached ones, so the exact objective (released reward, -30 for
the leftover mass, exact KL to the cached reference; core.baseline_matched_objective, the same
objective as --objective exact in patches/exact_confidence.patch) is one batched forward and
backward per update: no generation, no reference pass. Logits are computed only at the positions
the levels read (score_rows). Default schedule: one Adam update per 32 length-bucketed questions,
lr 3e-4, 300 steps (~5 min on an L40S; most of the gain by step 150). In a 3-seed comparison it
matched or beat Muon, Scaled AdamW and PoLoRA, and 1e-3 collapses some seeds; see
docs/fast_loop_log.md, which describes those optimizers (not kept in the code). The released
schedule (batches of 8, 4 passes in minibatches of 4, lr 1e-5) is available by flags.

eval: the confidence on each cached dev answer comes from one forward pass: the level
distribution q. Reported: ECE / AUROC / Brier of the expected confidence sum_k k q_k ("expected")
and of a confidence sampled from q at T=0.6 ("sampled", what generation would state), with the
released metrics (torchmetrics 11-bin ECE).

Approximations versus the released protocol: answers are the base model's (training cannot change
them), the stop after the number is not modelled, and one cached answer per question.
"""
import argparse
import collections
import json
import math
import os
import random
import statistics
import time

import torch
from xformers.ops.fmha import attn_bias
from unsloth import FastLanguageModel  # must precede transformers imports

from rewarding_doubt import attn_bias as attention_biases
from rewarding_doubt.core import baseline_matched_objective
from rewarding_doubt.paper_ppo import AdaptiveKLController, evaluation_metrics, is_correct_exact, is_correct_f1
from shared_prefix import LevelScheme, end_of_turn
from subset import subset_loader
from util.ResponseHandling import parse_answer_confidence


def load(model_name, adapter=None):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=adapter or model_name, max_seq_length=1048,
                                                         dtype=None, load_in_4bit=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_cache(args):
    ids = json.load(open(args.ids))
    model, tok = load(args.model)
    eot = end_of_turn(tok)
    confidence = tok.convert_tokens_to_ids("ĠConfidence")
    scheme = LevelScheme(tok, eot)
    rng = random.Random(args.seed)
    cache = dict(model=args.model, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), splits={})
    for split, hf_split, limit in (("train", "train", args.n_train), ("dev", "validation", 0)):
        data = subset_loader(ids)("triviaqa", hf_split, "verbalize", tok)
        order = list(range(len(data)))
        if limit:
            rng.shuffle(order)
            order = order[:int(limit * 1.05) + 32]  # a few answers miss " Confidence"
        rows, t0 = [], time.time()
        FastLanguageModel.for_inference(model)
        for start in range(0, len(order), args.batch):
            part = [data[i] for i in order[start:start + args.batch]]
            prompts = [d["query"] for d in part]
            width = max(map(len, prompts))
            input_ids = torch.tensor([[tok.eos_token_id] * (width - len(p)) + p for p in prompts]).cuda()
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts]).cuda()
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=96, do_sample=True,
                                     temperature=0.6, top_p=0.9, eos_token_id=[tok.eos_token_id, eot, confidence],
                                     pad_token_id=tok.eos_token_id)
            for d, p, row in zip(part, prompts, out[:, width:].tolist()):
                while row and row[-1] == tok.eos_token_id:
                    row.pop()
                if not row or row[-1] != confidence:
                    continue
                prediction, _ = parse_answer_confidence(tok.decode(row, skip_special_tokens=True) + ": 0", False)
                if prediction is None:
                    continue
                rows.append(dict(ids=p + row, qid=d.get("question_id"), answer=prediction,
                                 f1=bool(is_correct_f1(prediction, d["gt_candidates"])),
                                 em=bool(is_correct_exact(prediction, d["gt_candidates"]))))
            if limit and len(rows) >= limit:
                rows = rows[:limit]
                break
        FastLanguageModel.for_training(model)
        with torch.no_grad():
            for start in range(0, len(rows), 32):
                chunk = rows[start:start + 32]
                for r, (levels, _) in zip(chunk, scheme.batch(model, [r["ids"] for r in chunk], [None] * len(chunk))):
                    r["ref"] = levels.float().cpu()
        cache["splits"][split] = rows
        print(json.dumps(dict(split=split, answers=len(rows), accuracy_f1=statistics.fmean(r["f1"] for r in rows),
                              accuracy_em=statistics.fmean(r["em"] for r in rows), seconds=round(time.time() - t0))), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)
    print(json.dumps(dict(saved=args.out, base_dev=dev_metrics_from_levels(
        [r["ref"] for r in cache["splits"]["dev"]], [r["f1"] for r in cache["splits"]["dev"]]))), flush=True)


def dev_metrics_from_levels(levels, labels, temperature=0.6, seed=0):
    """Metrics of the expected and of a T=0.6-sampled confidence, from per-answer level log-probs."""
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


def score_rows(model, scheme, queries, answer_starts):
    """One forward pass over each query + the confidence prefix, with logits only at the positions read:
    (levels [B, 11], answers [B][T_b, V]). levels is log q over the confidence levels (LevelScheme.batch
    with k* = None); answers[b] is the log-softmax at the positions that predict query b's answer
    tokens, answer_starts[b] through the final " Confidence" (empty when answer_starts[b] = len(query)).
    Full-vocabulary logits at every position dominated the backward pass, hence the gathering."""
    device = next(model.parameters()).device
    base = model.get_base_model()
    tail = [] if scheme.single else [scheme.one]
    seqs = [q + scheme.common + tail for q in queries]
    width = max(map(len, seqs))
    # As Unsloth's CausalLM forward does before calling the inner model: no labels, and its causal mask
    # (without the mask, attention is bidirectional).
    base.model._has_no_labels = True
    hidden = base.model(input_ids=torch.tensor([s + [scheme.stop] * (width - len(s)) for s in seqs], device=device),
                        causal_mask=attn_bias.LowerTriangularMask())[0]
    c = len(scheme.common)
    rows = torch.arange(len(queries), device=device)[:, None]
    pos = torch.tensor([[len(q) - 1 + i for i in range(c + 1 + len(tail))] for q in queries], device=device)
    logp = base.lm_head(hidden[rows, pos].to(base.lm_head.weight.dtype)).float().log_softmax(-1)
    common = torch.tensor(scheme.common, device=device)
    common_logp = logp[:, :c].gather(-1, common[None, :, None].expand(len(queries), c, 1)).sum((1, 2))
    if scheme.single:
        levels = common_logp[:, None] + logp[:, c, scheme.numbers]
    else:
        levels = common_logp[:, None] + logp[:, c, scheme.digits]
        log_zero = logp[:, c + 1, scheme.zero]
        log_not_zero = torch.log1p(-log_zero.exp().clamp(max=1 - 1e-7))
        levels = torch.cat([levels[:, :1], levels[:, 1:2] + log_not_zero[:, None], levels[:, 2:], levels[:, 1:2] + log_zero[:, None]], 1)
    spans = [range(start - 1, len(q) - 1) for q, start in zip(queries, answer_starts)]
    flat_rows = torch.tensor([b for b, span in enumerate(spans) for _ in span], device=device, dtype=torch.long)
    flat_pos = torch.tensor([t for span in spans for t in span], device=device, dtype=torch.long)
    flat = base.lm_head(hidden[flat_rows, flat_pos].to(base.lm_head.weight.dtype)).float().log_softmax(-1)
    return levels, list(flat.split([len(span) for span in spans]))


def answer_start(ids, tokenizer):
    """Index of the first answer token in a cached row: right after the last "<|im_start|>assistant\n"
    (rows are the chat prompt, ending with that header, followed by the generated answer)."""
    start = len(ids) - 1 - ids[::-1].index(tokenizer.convert_tokens_to_ids("<|im_start|>")) + 3
    assert tokenizer.decode(ids[start - 3:start]) == "<|im_start|>assistant\n", tokenizer.decode(ids[start - 3:start])
    return start


def answer_kl(policy, reference):
    """KL(policy || reference) over the vocabulary, summed over each answer's positions: [B]."""
    return torch.stack([(p.exp() * (p - r)).sum() for p, r in zip(policy, reference)])


def reference_answers(model, scheme, rows):
    """The reference model's answer log-probs: LoRA and attention biases off (disable_adapter; see
    rewarding_doubt.attn_bias). Gate and MLP-output biases do not switch off, so they are not supported."""
    with torch.no_grad(), model.disable_adapter():
        return score_rows(model, scheme, [r["ids"] for r in rows], [r["answer_start"] for r in rows])[1]


def set_gradient_checkpointing(model, on):
    """Unsloth turns checkpointing on even with use_gradient_checkpointing=False; off is ~1.4x faster
    per update at ~3x the activation memory (10 GB at minibatch 16, 164-token rows)."""
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = on


def trainable_model(model_name):
    model, tok = load(model_name)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing=False, random_state=3407)
    import trl
    trl.trainer.peft_module_casting_to_bf16(model)
    FastLanguageModel.for_training(model)
    set_gradient_checkpointing(model, False)
    return model, tok


LORA_MODULES = ("q", "k", "v", "o", "gate", "up", "down")


def restrict_lora(model, modules, layers):
    """Train only the LoRA adapters on `modules` (subset of LORA_MODULES) in decoder layers `layers`
    (a range of indices); freeze the rest. Frozen adapters keep B = 0, so they leave the model unchanged.
    Returns the number of trainable parameters."""
    import re
    for name, p in model.named_parameters():
        m = re.search(r"\.layers\.(\d+)\..*\.(\w+)_proj\.lora_[AB]\.", name)
        if m:
            p.requires_grad_(m.group(2) in modules and int(m.group(1)) in layers)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_gate_biases(model):
    """A trainable bias on every MLP gate's pre-activation, zero-initialized (the model is unchanged
    at the start): down(act(gate(x) + b) * up(x)). Qwen-2.5's MLP has no biases, and Unsloth's fused
    LoRA MLP kernel ignores them, so these MLPs get a plain forward. Returns the number of biases."""
    import types

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x) + self.gate_bias.to(x.dtype)) * self.up_proj(x))

    device = next(model.parameters()).device
    for layer in model.get_base_model().model.layers:
        layer.mlp.gate_bias = torch.nn.Parameter(torch.zeros(model.config.intermediate_size, device=device))
        layer.mlp.forward = types.MethodType(forward, layer.mlp)
    return len(model.get_base_model().model.layers) * model.config.intermediate_size


def add_residual_biases(model, layers):
    """A trainable vector added to the MLP output (so to the residual stream) in each of `layers`,
    zero-initialized: h + mlp(x) + b. Wraps whatever forward the MLP has (Unsloth's fused kernel).
    Returns the number of parameters."""
    device = next(model.parameters()).device
    decoder = model.get_base_model().model.layers
    for i in layers:
        mlp = decoder[i].mlp
        mlp.residual_bias = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=device))
        inner = mlp.forward

        def forward(x, inner=inner, mlp=mlp):
            out = inner(x)
            return out + mlp.residual_bias.to(out.dtype)
        mlp.forward = forward
    return len(layers) * model.config.hidden_size


def cut_backward_below(model, first_trained_layer):
    """Make backward stop at the first decoder layer whose adapters train: its input is replaced by a
    detached copy. Exact (nothing below trains) and saves the backward through the frozen layers.
    The copy still requires grad: Unsloth's attention picks a grouped-query layout with no backward
    kernel when its input does not, and the adapters above need gradients. Returns the hook handle."""
    def cut(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0].detach().requires_grad_(True), *output[1:])
        return output.detach().requires_grad_(True)
    return model.get_base_model().model.layers[first_trained_layer - 1].register_forward_hook(cut)


def score_dev(model, scheme, rows, label_key="f1", with_answer_kl=True):
    """Dev metrics of the stated confidence on the cached answers, plus (with_answer_kl) how far the
    policy has moved on the answers themselves: KL(policy || reference) over the vocabulary at each
    answer token, per answer (answer_kl) and per token (answer_kl_token)."""
    FastLanguageModel.for_training(model)
    set_gradient_checkpointing(model, False)  # for_training turns it back on, which roughly doubled backward time
    levels, kls, tokens = [], [], 0
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            chunk = rows[start:start + 32]
            lv, answers = score_rows(model, scheme, [r["ids"] for r in chunk], [r["answer_start"] for r in chunk])
            levels += list(lv.float().cpu())
            if with_answer_kl:
                kls += answer_kl(answers, reference_answers(model, scheme, chunk)).tolist()
                tokens += sum(len(a) for a in answers)
    metrics = dev_metrics_from_levels(levels, [r[label_key] for r in rows])
    if with_answer_kl:
        metrics.update(answer_kl=statistics.fmean(kls), answer_kl_token=sum(kls) / tokens)
    return metrics


def regenerate_dev(model, tokenizer, rows, gt_candidates, batch=64):
    """Answer the dev questions again with the current policy (released sampling: T=0.6, top-p 0.9, stop
    at " Confidence"; the sampler seeded per batch, so policies are compared on the same draws) and grade
    them with F1 > 0.5: regen_accuracy over all questions, regen_malformed for answers that never
    reached " Confidence" or did not parse."""
    eot, confidence = end_of_turn(tokenizer), tokenizer.convert_tokens_to_ids("ĠConfidence")
    pad = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)
    correct = malformed = 0
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        prompts = [r["ids"][:r["answer_start"]] for r in part]
        width = max(map(len, prompts))
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts]).cuda()
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts]).cuda()
        torch.manual_seed(start)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=96, do_sample=True, temperature=0.6,
                                 top_p=0.9, eos_token_id=[pad, eot, confidence], pad_token_id=pad)
        for r, row in zip(part, out[:, width:].tolist()):
            while row and row[-1] == pad:
                row.pop()
            prediction = None
            if row and row[-1] == confidence:
                prediction, _ = parse_answer_confidence(tokenizer.decode(row, skip_special_tokens=True) + ": 0", False)
            if prediction is None:
                malformed += 1
            else:
                correct += bool(is_correct_f1(prediction, gt_candidates[r["qid"]]))
    FastLanguageModel.for_training(model)
    set_gradient_checkpointing(model, False)
    return dict(regen_accuracy=correct / len(rows), regen_malformed=malformed / len(rows))


def epoch_batches(lengths, batchsize, window, rng):
    """One epoch of batches (every row once, ragged tail dropped). With window > 0, length bucketing:
    the shuffled epoch is cut into windows of `window` batches, each window is sorted by length before
    being cut into batches, and the batch order is shuffled. Rows are right-padded to the longest in a
    batch; random batches of 32 are 17% padding, bucketed ones 0.2%. Which rows train is unchanged;
    only which rows share a batch (shorter answers are correct more often, so batches are less mixed)."""
    order = rng.sample(range(len(lengths)), len(lengths))
    if window > 0:
        span = window * batchsize
        order = [i for s in range(0, len(order), span) for i in sorted(order[s:s + span], key=lengths.__getitem__)]
    batches = [order[s:s + batchsize] for s in range(0, len(order) - batchsize + 1, batchsize)]
    rng.shuffle(batches)
    return batches


def train(args):
    cache = torch.load(args.cache, weights_only=False)
    model, tok = trainable_model(cache["model"])
    n_layers = model.config.num_hidden_layers
    lo, hi = (0, n_layers - 1) if args.lora_layers == "all" else map(int, args.lora_layers.split("-"))
    modules = [] if args.lora_modules == "none" else args.lora_modules.split(",")
    assert set(modules) <= set(LORA_MODULES), modules
    restrict_lora(model, modules, range(lo, hi + 1))
    n_gate_biases = add_gate_biases(model) if args.gate_bias else 0
    def layer_range(spec):
        return range(0) if spec == "none" else range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1)
    rb, ab = layer_range(args.residual_bias), layer_range(args.attn_bias)
    # attention biases: rewarding_doubt.attn_bias, which also acts in Unsloth's decoding loop (--regen-every)
    # and switches off under disable_adapter() (the reference for --answer-kl)
    n_residual_biases = add_residual_biases(model, rb) + sum(b.numel() for b in attention_biases.install(model, ab).values())
    # backward stops at the first layer where anything trains (section 7-8 of the log: ~28% per step for layers 18-35)
    first = min([lo] * bool(modules) + [0] * args.gate_bias + [rb.start] * bool(rb) + [ab.start] * bool(ab), default=0)
    if first > 0:
        cut_backward_below(model, first)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(dict(lora_modules=modules, lora_layers=[lo, hi], n_layers=n_layers, gate_biases=n_gate_biases,
                          residual_biases=n_residual_biases, backward_cut_at=first, trainable_params=n_trainable)), flush=True)
    scheme = LevelScheme(tok, end_of_turn(tok))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_rows, dev_rows = cache["splits"]["train"], cache["splits"]["dev"]
    for r in train_rows + dev_rows:
        r["answer_start"] = answer_start(r["ids"], tok)
    print("first dev answer:", repr(tok.decode(dev_rows[0]["ids"][dev_rows[0]["answer_start"]:])), flush=True)
    if args.regen_every:
        gt_candidates = {d["question_id"]: d["gt_candidates"] for d in
                         subset_loader({"validation": [r["qid"] for r in dev_rows]})("triviaqa", "validation", "verbalize", tok)}

    # the reference (disable_adapter) switches LoRA and attention biases off, but not gate or MLP-output biases
    reference_ok = not (args.gate_bias or rb)
    if args.answer_kl and not reference_ok:
        raise SystemExit("--answer-kl needs a reference without the trained parameters: not with --gate-bias / --residual-bias")

    def evaluate_dev(step):
        metrics = score_dev(model, scheme, dev_rows, with_answer_kl=reference_ok)
        if args.regen_every and step % args.regen_every == 0:
            metrics.update(regenerate_dev(model, tok, dev_rows, gt_candidates))
        return metrics
    # Adam as in the released code (TRL 0.8.6 PPOTrainer: torch.optim.Adam, default betas, no weight decay).
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    kl_ctl = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon) if args.adaptive_kl else None
    beta = args.kl_coef
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    curve = {0: evaluate_dev(0)}
    print(json.dumps(dict(step=0, **curve[0])), flush=True)
    batches, t0 = [], time.time()
    phase_seconds = collections.Counter()
    last = [time.time()]

    def lap(phase=None):
        """--time-steps: accumulate seconds per phase of an update (synchronizes the GPU, so slightly slower)."""
        if not args.time_steps:
            return
        torch.cuda.synchronize()
        now = time.time()
        if phase:
            phase_seconds[phase] += now - last[0]
        last[0] = now
    for step in range(1, args.steps + 1):
        if not batches:
            batches = epoch_batches([len(r["ids"]) for r in train_rows], args.batchsize, args.bucket_window, rng)
        batch = [train_rows[i] for i in batches.pop()]
        stats = []
        for p in range(args.passes):
            idx = list(range(len(batch)))
            rng.shuffle(idx)
            for mb in range(0, len(idx), args.minibatch):
                chunk = [batch[i] for i in idx[mb:mb + args.minibatch]]
                lap()
                scored, answers = score_rows(model, scheme, [r["ids"] for r in chunk], [r["answer_start"] for r in chunk])
                kls = answer_kl(answers, reference_answers(model, scheme, chunk)) if args.answer_kl else torch.zeros(len(chunk), device=scored.device)
                lap("forward")
                losses = []
                for r, levels, a_kl in zip(chunk, scored, kls):
                    J, kl = baseline_matched_objective(levels.double(), float(r[args.grading]), "discrete-exact", "released",
                                                       -30.0, ref_logq=r["ref"].to(levels.device).double(),
                                                       brier_mix=args.brier_mix)
                    losses.append(-(J - beta * kl) + args.answer_kl * a_kl)
                    if p == 0:
                        stats.append((J.item(), kl.item(), float((levels.double().softmax(-1) * torch.arange(11, device=levels.device)).sum()),
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
        record = dict(step=step, expected_reward=statistics.fmean(s[0] for s in stats), kl=statistics.fmean(s[1] for s in stats),
                      mean_confidence=statistics.fmean(s[2] for s in stats), answer_kl=statistics.fmean(s[3] for s in stats),
                      beta=beta, seconds=time.time() - t0,
                      **({"phase_seconds": dict(phase_seconds)} if args.time_steps else {}))
        log.write(json.dumps(record) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            curve[step] = evaluate_dev(step)
            print(json.dumps(dict(step=step, minutes=round((time.time() - t0) / 60, 1), beta=round(beta, 4), **curve[step])), flush=True)
            log.flush()
    json.dump({str(k): v for k, v in curve.items()}, open(os.path.join(args.out_dir, "curve.json"), "w"), indent=1)
    if args.save:
        model.save_pretrained(os.path.join(args.out_dir, "adapter"))


def evaluate(args):
    cache = torch.load(args.cache, weights_only=False)
    policy_dir = args.adapter
    if os.path.exists(os.path.join(policy_dir, "config.json")):  # a TRL snapshot: config + value head next to the adapter
        import shutil
        policy_dir = "/tmp/fast_eval_adapter"
        shutil.rmtree(policy_dir, ignore_errors=True)
        shutil.copytree(args.adapter, policy_dir, ignore=shutil.ignore_patterns("config.json", "pytorch_model.bin"))
    # The base by the cache's name, then the adapter: an adapter saved from an on-load-quantized model
    # records the pre-quantized checkpoint name, which the pinned Unsloth refuses to load.
    from peft import PeftModel
    model, tok = load(cache["model"])
    model = PeftModel.from_pretrained(model, policy_dir)
    attention_biases.load(model, args.adapter)
    rows = cache["splits"]["dev"]
    for r in rows:
        r["answer_start"] = answer_start(r["ids"], tok)
    metrics = score_dev(model, LevelScheme(tok, end_of_turn(tok)), rows)
    json.dump(metrics, open(args.out, "w"), indent=1)
    print(json.dumps(metrics), flush=True)


def profile(args):
    """Time one training update's parts on the cached rows: forward, loss, backward, optimizer."""
    cache = torch.load(args.cache, weights_only=False)
    model, tok = trainable_model(cache["model"])
    scheme = LevelScheme(tok, end_of_turn(tok))
    rows = cache["splits"]["train"][:256]
    lens = [len(r["ids"]) for r in rows]
    print(json.dumps(dict(tokens_mean=statistics.fmean(lens), tokens_max=max(lens), tokens_min=min(lens),
                          shared_prefix=len(os.path.commonprefix([r["ids"] for r in rows])),
                          logits_dtype=str(model(input_ids=torch.tensor([rows[0]["ids"]], device="cuda")).logits.dtype))), flush=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    def loss_of(chunk, levels_list):
        losses = []
        for r, levels in zip(chunk, levels_list):
            J, kl = baseline_matched_objective(levels.double(), float(r["em"]), "discrete-exact", "released", -30.0,
                                               ref_logq=r["ref"].to(levels.device).double())
            losses.append(-(J - 0.05 * kl))
        return torch.stack(losses).mean()

    def timed(name, mb, scorer, backward=True, n=64):
        times = dict(forward=0., loss=0., backward=0., step=0.)
        sync = torch.cuda.synchronize
        torch.cuda.reset_peak_memory_stats()
        for rep in range(2):  # the first pass is warmup
            for k in times: times[k] = 0.
            for start in range(0, n, mb):
                chunk = rows[start:start + mb]
                sync(); t = time.time()
                with torch.set_grad_enabled(backward):
                    levels = scorer([r["ids"] for r in chunk])
                sync(); times["forward"] += time.time() - t; t = time.time()
                if not backward:
                    continue
                loss = loss_of(chunk, levels)
                sync(); times["loss"] += time.time() - t; t = time.time()
                optimizer.zero_grad(); loss.backward()
                sync(); times["backward"] += time.time() - t; t = time.time()
                optimizer.step()
                sync(); times["step"] += time.time() - t
        per_q = {k: round(1000 * v / n, 1) for k, v in times.items()}
        print(json.dumps(dict(variant=name, minibatch=mb, ms_per_question=per_q, total=round(sum(per_q.values()), 1),
                              peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)

    current = lambda qs: [lv for lv, _ in scheme.batch(model, qs, [None] * len(qs))]
    selective = lambda qs: list(score_rows(model, scheme, qs, [len(q) for q in qs])[0])
    with torch.no_grad():  # the selective path must give the same levels
        a = torch.stack(current([r["ids"] for r in rows[:8]])).float(); b = torch.stack(selective([r["ids"] for r in rows[:8]])).float()
        print(json.dumps(dict(selective_max_abs_diff=float((a - b).abs().max()))), flush=True)
    timed("current", 4, current)
    for gc in (True, False):
        set_gradient_checkpointing(model, gc)
        for mb in (8, 16, 32, 64):
            timed(f"selective_gc_{gc}", mb, selective)
    set_gradient_checkpointing(model, True)
    from torch.profiler import profile as torch_profile, ProfilerActivity
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for start in range(0, 16, 4):
            chunk = rows[start:start + 4]
            loss = loss_of(chunk, current([r["ids"] for r in chunk]))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), flush=True)


def profile_lora(args):
    """Time and peak memory of a training update when only some layers' adapters train, with and
    without gradients forced onto the input embeddings (Unsloth's setting for checkpointing). If the
    inputs do not require grad, backward can stop at the first layer whose adapters train."""
    cache = torch.load(args.cache, weights_only=False)
    model, tok = trainable_model(cache["model"])
    scheme = LevelScheme(tok, end_of_turn(tok))
    rows = sorted(cache["splits"]["train"][:512], key=lambda r: len(r["ids"]))[:256]  # bucketed-like batches
    n_layers = model.config.num_hidden_layers
    probe = model.get_input_embeddings()(torch.tensor([[1, 2]], device="cuda"))
    print(json.dumps(dict(embedding_output_requires_grad=probe.requires_grad,
                          require_grads_hook=hasattr(model.get_base_model(), "_require_grads_hook"))), flush=True)
    for layers in ("all", f"{n_layers // 2}-{n_layers - 1}", f"{3 * n_layers // 4}-{n_layers - 1}"):
        lo, hi = (0, n_layers - 1) if layers == "all" else map(int, layers.split("-"))
        n = restrict_lora(model, LORA_MODULES, range(lo, hi + 1))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)
        decoder = model.get_base_model().model.layers
        for variant in ("input_grads", "no_input_grads", "detach_below"):
            base = model.get_base_model()
            if variant == "input_grads":
                base.enable_input_require_grads()
            elif hasattr(base, "_require_grads_hook"):
                base.disable_input_require_grads()
            seen = {}
            def probe(module, inputs, output):  # returns None: a forward hook's return value replaces the output
                seen.setdefault("layer0_out_requires_grad", (output[0] if isinstance(output, tuple) else output).requires_grad)
            probe_hook = decoder[0].register_forward_hook(probe)
            cut = None
            if variant == "detach_below" and lo > 0:
                cut = cut_backward_below(model, lo)
            fwd = bwd = 0.
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for rep in range(2):  # first pass is warmup
                fwd = bwd = 0.
                for start in range(0, len(rows), 32):
                    chunk = rows[start:start + 32]
                    torch.cuda.synchronize(); t = time.time()
                    levels = score_rows(model, scheme, [r["ids"] for r in chunk], [len(r["ids"]) for r in chunk])[0]
                    loss = levels.logsumexp(-1).mean()
                    torch.cuda.synchronize(); fwd += time.time() - t; t = time.time()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    torch.cuda.synchronize(); bwd += time.time() - t
            probe_hook.remove()
            if cut is not None:
                cut.remove()
            print(json.dumps(dict(layers=layers, trainable_params=n, variant=variant, **seen,
                                  forward_ms_per_q=round(1000 * fwd / len(rows), 1), backward_ms_per_q=round(1000 * bwd / len(rows), 1),
                                  peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)


def dev_gt(args):
    """{question_id: gt_candidates} for the cache's dev rows, from the released data loader, so trainers
    without the released code can grade regenerated answers (minimal_trainer.py --dev-gt)."""
    cache = torch.load(args.cache, weights_only=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cache["model"])
    qids = [r["qid"] for r in cache["splits"]["dev"]]
    data = subset_loader({"validation": qids})("triviaqa", "validation", "verbalize", tok)
    gt = {str(d["question_id"]): d["gt_candidates"] for d in data}
    missing = [q for q in qids if str(q) not in gt]
    json.dump(gt, open(args.out, "w"))
    print(json.dumps(dict(dev_rows=len(qids), written=len(gt), missing=len(missing))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache")
    c.add_argument("ids"); c.add_argument("out")
    c.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    c.add_argument("--n-train", type=int, default=8000)
    c.add_argument("--batch", type=int, default=64)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("cache"); t.add_argument("out_dir")
    # Defaults: the fast schedule (runs/fast/seed_sweep_configs.json, 3 seeds, 4.7 min on an L40S). The released
    # schedule is --batchsize 8 --passes 4 --minibatch 4 --lr 1e-5 (overhead-bound: 48 min per 1,000 steps).
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
    t.add_argument("--brier-mix", type=float, default=0.0)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--lora-modules", default=",".join(LORA_MODULES), help='projections whose adapters train, or "none"')
    t.add_argument("--gate-bias", action="store_true", help="also train a bias on every MLP gate pre-activation (all layers)")
    t.add_argument("--residual-bias", default="none", help='"none" or a layer range, e.g. "18-35": train a vector added to each MLP output')
    t.add_argument("--attn-bias", default="none", help='"none" or a layer range: train a vector added to each attention output')
    t.add_argument("--lora-layers", default="all", help='"all" or an inclusive range of decoder layers, e.g. "18-35"')
    t.add_argument("--answer-kl", type=float, default=0.0,
                   help="weight of KL(policy || reference) summed over each answer's tokens (reference: LoRA and attention biases off)")
    t.add_argument("--regen-every", type=int, default=0,
                   help="also regenerate and grade the dev answers with the policy every N steps (and at step 0)")
    t.add_argument("--time-steps", action="store_true", help="log cumulative seconds per update phase (adds GPU syncs)")
    t.add_argument("--bucket-window", type=int, default=64, help="batches per length-sorting window; 0 = random batches")
    t.add_argument("--save", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("cache"); e.add_argument("adapter"); e.add_argument("out")
    pr = sub.add_parser("profile")
    pr.add_argument("cache")
    g = sub.add_parser("dev-gt")
    g.add_argument("cache"); g.add_argument("out")
    pl = sub.add_parser("profile-lora")
    pl.add_argument("cache")
    args = parser.parse_args()
    {"cache": build_cache, "train": train, "eval": evaluate, "profile": profile, "profile-lora": profile_lora, "dev-gt": dev_gt}[args.command](args)


if __name__ == "__main__":
    main()
