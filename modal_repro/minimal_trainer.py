"""A minimal trainer for the exact confidence objective: Hugging Face transformers, PEFT and PyTorch only.

    python minimal_trainer.py ref CACHE.pt OUT.pt          # bf16 reference levels, untrained dev metrics
    python minimal_trainer.py diagnose CACHE.pt            # shared prefix vs full forward, bf16 and fp32
    python minimal_trainer.py train CACHE.pt OUT_DIR [--steps 300 --lr 3e-4 --attn-bias 18-35 ...]

The data, objective, schedule and metrics are fast_loop.py's (docs/fast_loop_log.md) without Unsloth or TRL:
the model is bf16 with PyTorch SDPA attention, gradient checkpointing is off, and inputs never require grad,
so backward stops by itself at the first layer with anything to train.

Scoring: the 11 levels ": k" from one forward pass, with logits only at the positions they read (as
fast_loop.level_logps). Qwen-2.5 splits "10" into "1" "0", so one extra position gives P("0" | "1"):
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
"""
import argparse
import collections
import json
import os
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from rewarding_doubt.core import baseline_matched_objective
from rewarding_doubt.paper_ppo import AdaptiveKLController, evaluation_metrics

LORA_MODULES = ("q", "k", "v", "o", "gate", "up", "down")


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


def level_logps(lm, levels, queries, pad, shared_prefix=True):
    """[B, 11] log q over the levels after each query. With shared_prefix, the queries' common prefix runs
    once and its key/value cache is expanded over the batch; otherwise one full forward."""
    device = lm.lm_head.weight.device
    seqs = [q + levels.common + levels.tail for q in queries]
    # The prefix stops before the first read position of the shortest query.
    n = min(len(os.path.commonprefix(queries)), min(map(len, queries)) - 1) if shared_prefix else 0
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
    return levels.from_logp(lm.lm_head(hidden[rows, pos]).float().log_softmax(-1))


def load(model_name, device="cuda"):
    """bf16, or 4-bit for a pre-quantized checkpoint such as unsloth/Qwen2.5-3B-Instruct-bnb-4bit (the weights
    Unsloth loads for unsloth/Qwen2.5-3B-Instruct with load_in_4bit: bitsandbytes NF4, double quantization)."""
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                              device_map={"": device})
    lm.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name)
    return lm, tok


def add_lora(lm, modules, layers, dtype=torch.float32):
    """Rank-8 LoRA (alpha 8, no dropout, no bias) on `modules` in decoder `layers`; adapters elsewhere would
    be frozen at B = 0 and change nothing, so they are not created. PEFT keeps adapter weights in fp32;
    the fast loop's were bf16 (TRL's peft_module_casting_to_bf16), so Adam's updates were rounded to bf16."""
    from peft import LoraConfig, get_peft_model
    if not modules or not layers:
        return []
    config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=[f"{m}_proj" for m in modules],
                        layers_to_transform=list(layers), layers_pattern="layers")
    get_peft_model(lm, config)  # injects the adapters into lm's modules in place
    params = [p for n, p in lm.named_parameters() if "lora_" in n]
    for p in params:
        p.data = p.data.to(dtype)
    return params


def add_output_biases(lm, layers, part):
    """A trainable fp32 vector added to the output of `part` ("self_attn" or "mlp"), that is to its write to
    the residual stream, in each of `layers`; zero-initialized, by forward hooks. Returns the parameters."""
    params = []
    for i in layers:
        module = getattr(lm.model.layers[i], part)
        bias = torch.nn.Parameter(torch.zeros(lm.config.hidden_size, device=lm.lm_head.weight.device))

        def hook(module, inputs, out, bias=bias):
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
    return [lv for s in range(0, len(rows), chunk) for lv in level_logps(lm, levels, [r["ids"] for r in rows[s:s + chunk]], pad).float().cpu()]


def check(lm, levels, rows, pad, params=()):
    """Shared-prefix against full-forward scoring on `rows`: max |difference| of the level log-probs, and of
    the gradients of their sum with respect to `params` (relative to the largest gradient entry)."""
    out = {}
    grads = []
    for shared in (True, False):
        for p in params:
            p.grad = None
        lv = level_logps(lm, levels, [r["ids"] for r in rows], pad, shared_prefix=shared)
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
            full = level_logps(lm, levels, queries, pad, shared_prefix=False)
            shared = level_logps(lm, levels, queries, pad)
            single = torch.cat([level_logps(lm, levels, [q], pad, shared_prefix=False) for q in queries])
        print(json.dumps(dict(dtype=str(dtype), shared_vs_full=compare(shared, full), batched_vs_single=compare(full, single),
                              shared_vs_single=compare(shared, single),
                              cache_ref_vs_single=compare(torch.stack([r["ref"] for r in rows]), single))), flush=True)
        del lm
        torch.cuda.empty_cache()


def memory(args):
    """Peak GPU memory of one training update on a bucketed batch, after forward and after backward, for all-layer
    LoRA with and without the shared prefix."""
    cache = torch.load(args.cache, weights_only=False)
    lm, tok = load(args.model or cache["model"])
    params = add_lora(lm, list(LORA_MODULES), range(lm.config.num_hidden_layers))
    levels, pad = Levels(tok), tok.eos_token_id
    rows = cache["splits"]["train"]
    batch = [rows[i] for i in epoch_batches([len(r["ids"]) for r in rows], 32, 64, random.Random(1))[0]]
    queries = [r["ids"] for r in batch]
    print(json.dumps(dict(weights_gb=round(torch.cuda.memory_allocated() / 2**30, 2), prefix=len(os.path.commonprefix(queries)),
                          lengths=[min(map(len, queries)), max(map(len, queries))])), flush=True)
    for shared in (True, False):
        torch.cuda.reset_peak_memory_stats()
        lv = level_logps(lm, levels, queries, pad, shared_prefix=shared)
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
    lm, tok = load(args.model or cache["model"])
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
    cache.update(ref_model=f"{args.model or cache['model']} bf16 (minimal_trainer.py ref)", ref_created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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
    lm, tok = load(args.model or cache["model"])
    pad = tok.eos_token_id
    n_layers = lm.config.num_hidden_layers
    modules = [] if args.lora_modules == "none" else args.lora_modules.split(",")
    assert set(modules) <= set(LORA_MODULES), modules
    torch.manual_seed(args.seed)
    lora_layers, attn_layers, mlp_layers = (layer_range(s, n_layers) for s in (args.lora_layers, args.attn_bias, args.residual_bias))
    trained = dict(lora=add_lora(lm, modules, lora_layers, getattr(torch, args.adapter_dtype)), attn_bias=add_output_biases(lm, attn_layers, "self_attn"),
                   residual_bias=add_output_biases(lm, mlp_layers, "mlp"))
    params = [p for group in trained.values() for p in group]
    print(json.dumps(dict(lora_modules=modules, lora_layers=[lora_layers.start, lora_layers.stop - 1] if modules else None,
                          attn_bias=args.attn_bias, residual_bias=args.residual_bias, trainable_params=sum(p.numel() for p in params),
                          shared_prefix=not args.no_shared_prefix)), flush=True)
    levels = Levels(tok)
    rng = random.Random(args.seed)
    train_rows, dev_rows = cache["splits"]["train"], cache["splits"]["dev"]
    # Adam as in the released code (TRL 0.8.6 PPOTrainer: torch.optim.Adam, default betas, no weight decay).
    optimizer = torch.optim.Adam(params, lr=args.lr)
    kl_ctl = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon) if args.adaptive_kl else None
    beta = args.kl_coef

    def dev_metrics():
        return dev_metrics_from_levels(score(lm, levels, dev_rows, pad), [r["f1"] for r in dev_rows])
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    curve = {0: dev_metrics()}
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
                scored = level_logps(lm, levels, [r["ids"] for r in chunk], pad, shared_prefix=not args.no_shared_prefix)
                lap("forward")
                losses = []
                for r, lv in zip(chunk, scored):
                    J, kl = baseline_matched_objective(lv.double(), float(r[args.grading]), "discrete-exact", "released",
                                                       -30.0, ref_logq=r["ref"].to(lv.device).double())
                    losses.append(-(J - beta * kl))
                    if p == 0:
                        stats.append((J.item(), kl.item(), float((lv.double().softmax(-1) * torch.arange(11, device=lv.device)).sum())))
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
                      mean_confidence=statistics.fmean(s[2] for s in stats), beta=beta, seconds=time.time() - t0,
                      **({"phase_seconds": dict(phase_seconds)} if args.time_steps else {}))
        log.write(json.dumps(record) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            curve[step] = dev_metrics()
            print(json.dumps(dict(step=step, minutes=round((time.time() - t0) / 60, 1), beta=round(beta, 4),
                                  peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), **curve[step])), flush=True)
            log.flush()
    json.dump({str(k): v for k, v in curve.items()}, open(os.path.join(args.out_dir, "curve.json"), "w"), indent=1)
    if args.save:
        torch.save(dict(args=vars(args), lora={n: p.detach().cpu() for n, p in lm.named_parameters() if "lora_" in n},
                        attn_bias=[p.detach().cpu() for p in trained["attn_bias"]],
                        residual_bias=[p.detach().cpu() for p in trained["residual_bias"]]), os.path.join(args.out_dir, "trained.pt"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("ref")
    r.add_argument("cache"); r.add_argument("out")
    r.add_argument("--model", default=None, help="default: the cache's model")
    d = sub.add_parser("diagnose")
    d.add_argument("cache")
    d.add_argument("--rows", type=int, default=64)
    m = sub.add_parser("memory")
    m.add_argument("cache")
    m.add_argument("--model", default=None)
    t = sub.add_parser("train")
    t.add_argument("cache"); t.add_argument("out_dir")
    t.add_argument("--model", default=None, help="default: the cache's model; unsloth/Qwen2.5-3B-Instruct-bnb-4bit for "
                                                 "the 4-bit weights the fast loop trained (with the 4-bit-reference cache)")
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
    t.add_argument("--adapter-dtype", choices=["float32", "bfloat16"], default="float32",
                   help="LoRA weight dtype; the fast loop's were bfloat16")
    t.add_argument("--attn-bias", default="none", help='"none", "all" or a layer range: train a vector added to each attention output')
    t.add_argument("--residual-bias", default="none", help='"none", "all" or a layer range: train a vector added to each MLP output')
    t.add_argument("--bucket-window", type=int, default=64, help="batches per length-sorting window; 0 = random batches")
    t.add_argument("--no-shared-prefix", action="store_true", help="full forward per row instead of a shared prefix cache")
    t.add_argument("--time-steps", action="store_true", help="log cumulative seconds per update phase (adds GPU syncs)")
    t.add_argument("--save", action="store_true", help="save the trained parameters to OUT_DIR/trained.pt")
    args = parser.parse_args()
    {"ref": make_ref, "diagnose": diagnose, "memory": memory, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
