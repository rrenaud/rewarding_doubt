"""A fast, approximate loop for iterating on confidence objectives (no generation after caching).

    python fast_loop.py cache IDS_JSON OUT.pt --model M --n-train 8000
    python fast_loop.py train CACHE.pt OUT_DIR [--steps 1000 --kl-coef 0.05 --adaptive-kl ...]
    python fast_loop.py eval CACHE.pt ADAPTER_DIR OUT.json

cache: the base model answers each question once, as the released training does (released prompt,
T=0.6, top-p 0.9, stopping at " Confidence"), for a pool of training questions and the dev split.
Each answer is graded (F1 > 0.5 and exact match) and stored as tokens with the reference model's
log-probabilities of the 11 confidence levels (LevelScheme, adapter-free base model).

train: answers stay frozen at the cached ones, so the exact objective (released reward, -30 for
the leftover mass, exact KL to the cached reference; core.baseline_matched_objective, the same
objective as --objective exact in patches/exact_confidence.patch) is one batched forward and
backward per update: no generation, no reference pass. Same schedule as the released PPO config:
batches of 8, 4 passes in minibatches of 4. The dev split is scored every --eval-every steps.

eval: the confidence on each cached dev answer comes from one forward pass: the level
distribution q. Reported: ECE / AUROC / Brier of the expected confidence sum_k k q_k ("expected")
and of a confidence sampled from q at T=0.6 ("sampled", what generation would state), with the
released metrics (torchmetrics 11-bin ECE).

Approximations versus the released protocol: answers are the base model's (training cannot change
them), the stop after the number is not modelled, and one cached answer per question.
"""
import argparse
import json
import math
import os
import random
import statistics
import time

import torch
from unsloth import FastLanguageModel  # must precede transformers imports

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


def score_dev(model, scheme, rows, label_key="f1"):
    FastLanguageModel.for_training(model)
    levels = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            chunk = rows[start:start + 32]
            levels += [lv.float().cpu() for lv, _ in scheme.batch(model, [r["ids"] for r in chunk], [None] * len(chunk))]
    return dev_metrics_from_levels(levels, [r[label_key] for r in rows])


def train(args):
    cache = torch.load(args.cache, weights_only=False)
    model, tok = load(cache["model"])
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing=False, random_state=3407)
    import trl
    trl.trainer.peft_module_casting_to_bf16(model)
    FastLanguageModel.for_training(model)
    scheme = LevelScheme(tok, end_of_turn(tok))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_rows, dev_rows = cache["splits"]["train"], cache["splits"]["dev"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    kl_ctl = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon) if args.adaptive_kl else None
    beta = args.kl_coef
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    curve = {0: score_dev(model, scheme, dev_rows)}
    print(json.dumps(dict(step=0, **curve[0])), flush=True)
    order, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        if len(order) < args.batchsize:
            order += rng.sample(range(len(train_rows)), len(train_rows))
        batch = [train_rows[i] for i in order[:args.batchsize]]
        del order[:args.batchsize]
        stats = []
        for p in range(args.passes):
            idx = list(range(len(batch)))
            rng.shuffle(idx)
            for mb in range(0, len(idx), args.minibatch):
                chunk = [batch[i] for i in idx[mb:mb + args.minibatch]]
                scored = scheme.batch(model, [r["ids"] for r in chunk], [None] * len(chunk))
                losses = []
                for r, (levels, _) in zip(chunk, scored):
                    J, kl = baseline_matched_objective(levels.double(), float(r[args.grading]), "discrete-exact", "released",
                                                       -30.0, ref_logq=r["ref"].to(levels.device).double(),
                                                       brier_mix=args.brier_mix)
                    losses.append(-(J - beta * kl))
                    if p == 0:
                        stats.append((J.item(), kl.item(), float((levels.double().softmax(-1) * torch.arange(11, device=levels.device)).sum())))
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                optimizer.step()
        if kl_ctl is not None:
            kl_ctl.update(statistics.fmean(s[1] for s in stats), args.batchsize)
            beta = kl_ctl.value
        record = dict(step=step, expected_reward=statistics.fmean(s[0] for s in stats), kl=statistics.fmean(s[1] for s in stats),
                      mean_confidence=statistics.fmean(s[2] for s in stats), beta=beta, seconds=time.time() - t0)
        log.write(json.dumps(record) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            curve[step] = score_dev(model, scheme, dev_rows)
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
    metrics = score_dev(model, LevelScheme(tok, end_of_turn(tok)), cache["splits"]["dev"])
    json.dump(metrics, open(args.out, "w"), indent=1)
    print(json.dumps(metrics), flush=True)


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
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batchsize", type=int, default=8)
    t.add_argument("--passes", type=int, default=4)
    t.add_argument("--minibatch", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--kl-coef", type=float, default=0.05)
    t.add_argument("--adaptive-kl", action="store_true", help="the released controller: target 6, horizon 10000")
    t.add_argument("--kl-target", type=float, default=6.0)
    t.add_argument("--kl-horizon", type=float, default=10000.0)
    t.add_argument("--grading", choices=["em", "f1"], default="em", help="training label (released default: exact match)")
    t.add_argument("--brier-mix", type=float, default=0.0)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--save", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("cache"); e.add_argument("adapter"); e.add_argument("out")
    args = parser.parse_args()
    {"cache": build_cache, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
