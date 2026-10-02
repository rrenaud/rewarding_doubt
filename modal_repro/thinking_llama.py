"""Unscored thinking before the confidence (docs/thinking_experiment.md).

The released system prompt is changed to ask for
    Answer: <answer>
    Check: <a brief check of the answer>
    Confidence: <0-10>
and each response is built in three stages by the harness:
1. answer: sampled at T=0.6, top-p 0.9 (the released settings), cut at the first newline;
   graded with F1 > 0.5; never trained.
2. check: up to --check-tokens tokens after "\nCheck:", sampled at T=1, kept through the first
   token that contains a newline (a newline is appended if the check ran out of tokens or reached
   the word "Confidence"). Source by arm: the policy ("trained"), the base model with the adapter
   off ("frozen"), a fixed filler text ("filler"), or nothing ("none": no "Check:" line).
   Arm "none" uses a prompt that asks for 'Answer: <answer>\nConfidence: <confidence>'.
3. confidence: "Confidence: " is appended and scored exactly as in discrete-exact single-pass:
   one forward gives the 11 number log-probs (including the forced "Confidence: ", so the mass
   hinge is unchanged) and, after a sampled k*, the stop probability.

Training (per question, G checks): loss = mean over checks of core.objective (discrete-exact,
hinges) and, for "trained", a REINFORCE term on the sampled check tokens with reward
r = J - beta * KL(check), J = -objective (detached), KL = sum over the sampled check tokens of
log pi - log pi_base, and a leave-one-out baseline over the question's other checks.

At the end the adapter is scored on dev (see `evaluate`) and results.json is written.

    python thinking_llama.py IDS_JSON OUT_DIR --arm trained|frozen|filler|none [--kl-beta B] [--eval-only]
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import time

import torch
from unsloth import FastLanguageModel  # must precede transformers imports
import trl

from rewarding_doubt.core import objective
from rewarding_doubt.paper_ppo import evaluation_metrics, is_correct_f1
from shared_prefix import split_candidates
from subset import subset_loader
from util.Prompts import get_prompt

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
RELEASED_FORMAT = "The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else."
THINK_PROMPT = get_prompt("open").replace(
    RELEASED_FORMAT,
    "Before giving the confidence, briefly check whether your answer is correct. The output should have the "
    "format 'Answer: <answer>\nCheck: <a brief check of the answer>\nConfidence: <confidence>' and nothing else.")
# Arm "none" is asked for the same layout without the check line. (With THINK_PROMPT the base model
# puts almost no mass on "Confidence" right after the answer, and the mass hinge wrecks the answers.)
NONE_PROMPT = get_prompt("open").replace(
    RELEASED_FORMAT, "The output should have the format 'Answer: <answer>\nConfidence: <confidence>' and nothing else.")
assert THINK_PROMPT != get_prompt("open") != NONE_PROMPT
FILLER = " Let me check the answer." * 6
FREE_FORMAT = re.compile(r"^\s*Answer:\s*(.+?)\s*\n+\s*Check:(.+?)\n+\s*Confidence:\s*(\d+)\s*$", re.DOTALL)


class Harness:
    """Tokenizer-level pieces shared by training and evaluation."""

    def __init__(self, tokenizer, check_tokens):
        self.tok = tokenizer
        self.eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.newline = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.check_prefix = tokenizer.encode("\nCheck:", add_special_tokens=False)
        candidates = [tokenizer.encode(f"Confidence: {k}", add_special_tokens=False) + [self.eot] for k in range(11)]
        self.common, self.numbers, self.stop = split_candidates(candidates)
        self.filler = tokenizer.encode(FILLER, add_special_tokens=False) + [self.newline]
        self.check_tokens = check_tokens
        # Generation stops at any token containing a newline.
        self.newline_ids = [i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])]
        self.stops = [tokenizer.eos_token_id, self.eot] + self.newline_ids

    def generate(self, model, queries, max_new_tokens, temperature, top_p, n=1):
        width = max(map(len, queries))
        pad = self.tok.eos_token_id
        input_ids = torch.tensor([[pad] * (width - len(q)) + q for q in queries]).cuda()
        mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in queries]).cuda()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                                 num_return_sequences=n, eos_token_id=self.stops, pad_token_id=pad)
        return [row[width:] for row in out.tolist()]

    def answers(self, model, prompts):
        """Stage 1: (tokens of "Answer: X", re-encoded; X) per prompt, or None if malformed."""
        out = []
        for row in self.generate(model, prompts, 48, 0.6, 0.9):
            text = self.tok.decode(row, skip_special_tokens=True).split("\n")[0]
            text = re.split(r",?\s*(?:Check|Confidence)\b", text)[0].strip()
            if not text.startswith("Answer:") or not text[len("Answer:"):].strip():
                out.append(None)
                continue
            out.append((self.tok.encode(text, add_special_tokens=False), text[len("Answer:"):].strip()))
        return out

    def cut_check(self, row):
        """(check tokens ending in a newline, number of them that were sampled)."""
        kept, text = [], ""
        for t in row:
            piece = self.tok.decode([t])
            if t in (self.tok.eos_token_id, self.eot) or "Confidence" in text + piece:
                break
            kept.append(t)
            text += piece
            if "\n" in piece:
                return kept, len(kept)
        return kept + [self.newline], len(kept)

    def checks(self, model, prefixes, arm, n, temperature=1.0):
        """Stage 2: n (tokens, sampled count) checks per prefix."""
        if arm == "filler":
            return [[(list(self.filler), 0) for _ in range(n)] for _ in prefixes]
        if arm == "none":
            return [[([self.newline], 0) for _ in range(n)] for _ in prefixes]
        if arm == "frozen":
            with model.disable_adapter():
                rows = self.generate(model, prefixes, self.check_tokens, temperature, 1.0, n)
        else:
            rows = self.generate(model, prefixes, self.check_tokens, temperature, 1.0, n)
        return [[self.cut_check(rows[i * n + g]) for g in range(n)] for i in range(len(prefixes))]

    def score(self, model, items, with_k=True):
        """One forward over right-padded item sequences.

        Each item has `seq` (prefix + check + common), `span` (the sampled check positions) and
        `k` (sampled number index or None). Returns per item (number log-probs [11], stop
        log-prob or None, sampled-check token log-probs). Unsloth's training forward uses no
        attention mask; attention is causal and every position read precedes the padding.
        """
        seqs = [it["seq"] + ([self.numbers[it["k"]]] if with_k and it["k"] is not None else []) for it in items]
        width = max(map(len, seqs))
        ids = torch.tensor([s + [self.stop] * (width - len(s)) for s in seqs]).cuda()
        logits = model(input_ids=ids).logits
        numbers = torch.tensor(self.numbers, device=logits.device)
        out = []
        for b, (it, s) in enumerate(zip(items, seqs)):
            n = len(it["seq"])
            c = n - len(self.common)
            common_logp = sum(logits[b, c - 1 + i].float().log_softmax(-1)[t] for i, t in enumerate(self.common))
            number_logps = common_logp + logits[b, n - 1].float().log_softmax(-1)[numbers]
            stop_logp = logits[b, n].float().log_softmax(-1)[self.stop] if with_k and it["k"] is not None else None
            c0, c1 = it["span"]
            if c1 > c0:
                targets = torch.tensor(s[c0:c1], device=logits.device)[:, None]
                check = logits[b, c0 - 1:c1 - 1].float().log_softmax(-1).gather(-1, targets).squeeze(-1)
            else:
                check = logits.new_zeros(0, dtype=torch.float32)
            out.append((number_logps, stop_logp, check))
        return out

    def next_token_logits(self, model, seqs, chunk=16):
        """Logits at the last position of each sequence (no grad), in chunks."""
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), chunk):
                part = seqs[i:i + chunk]
                width = max(map(len, part))
                ids = torch.tensor([s + [self.stop] * (width - len(s)) for s in part]).cuda()
                logits = model(input_ids=ids).logits
                out += [logits[b, len(s) - 1].float() for b, s in enumerate(part)]
        return out


def item_for(prefix, check, h):
    tokens, sampled = check
    return dict(seq=prefix + tokens + h.common, span=(len(prefix), len(prefix) + sampled))


def prefix_for(prompt, answer_tokens, arm, h):
    return prompt + answer_tokens + ([] if arm == "none" else h.check_prefix)


def sample_top_p(logits, temperature, top_p, gen):
    probs = (logits / temperature).softmax(-1)
    sorted_probs, index = probs.sort(descending=True)
    sorted_probs = sorted_probs * (sorted_probs.cumsum(-1) - sorted_probs < top_p)
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=gen)
    return int(index[choice])


def evaluate(model, h, data, arm, seed=0, batch=32):
    """Dev evaluation, one response per question.

    Answer (T=0.6, top-p 0.9), check (T=1, as trained; arm "frozen" with the adapter off), then the
    confidence sampled at T=0.6, top-p 0.9 from the logits after "Confidence: " (a non-number is a
    format failure). From the same logits, the unsampled confidence E_pi[k] (pi renormalized over
    the 11 numbers), and two re-scorings of E_pi[k]: with another question's check ("swapped") and
    with the check line removed ("nocheck").
    """
    FastLanguageModel.for_inference(model)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for start in range(0, len(data), batch):
        part = [data[i] for i in range(start, min(start + batch, len(data)))]
        prompts = [d["query"] for d in part]
        answers = h.answers(model, prompts)
        ok = [i for i, a in enumerate(answers) if a is not None]
        prefixes = [prefix_for(prompts[i], answers[i][0], arm, h) for i in ok]
        checks = h.checks(model, prefixes, arm, 1) if prefixes else []
        for j, i in enumerate(ok):
            rows.append(dict(prompt=prompts[i], answer_tokens=answers[i][0], answer=answers[i][1],
                             check=checks[j][0][0], correct=is_correct_f1(answers[i][1], part[i]["gt_candidates"])))
        rows += [dict(answer=None, correct=False) for a in answers if a is None]
    FastLanguageModel.for_training(model)  # plain forward passes below
    valid = [r for r in rows if r["answer"] is not None]
    numbers = torch.tensor(h.numbers).cuda()
    levels = torch.arange(11, device="cuda", dtype=torch.float32)

    def unsampled(logits):
        return float((logits[numbers].softmax(-1) * levels).sum()), float(logits.log_softmax(-1)[numbers].logsumexp(-1).exp())

    own = h.next_token_logits(model, [prefix_for(r["prompt"], r["answer_tokens"], arm, h) + r["check"] + h.common
                                      for r in valid])
    for r, lg in zip(valid, own):
        k = sample_top_p(lg, 0.6, 0.9, gen)
        r["confidence"] = h.numbers.index(k) if k in h.numbers else None
        r["pi_confidence"], r["number_mass"] = unsampled(lg)
    others = list(range(len(valid)))
    random.Random(seed).shuffle(others)
    if arm in ("trained", "frozen"):
        swapped = h.next_token_logits(model, [prefix_for(r["prompt"], r["answer_tokens"], arm, h) + valid[o]["check"]
                                              + h.common for r, o in zip(valid, others)])
        for r, lg in zip(valid, swapped):
            r["pi_swapped"] = unsampled(lg)[0]
    if arm != "none":
        nocheck = h.next_token_logits(model, [r["prompt"] + r["answer_tokens"] + [h.newline] + h.common for r in valid])
        for r, lg in zip(valid, nocheck):
            r["pi_nocheck"] = unsampled(lg)[0]
    metrics = evaluation_metrics([dict(confidence=r.get("confidence"), correct=r["correct"]) for r in rows])
    for key, name in (("pi_confidence", "unsampled"), ("pi_swapped", "swapped"), ("pi_nocheck", "nocheck")):
        if valid and key in valid[0]:
            m = evaluation_metrics([dict(confidence=r[key], correct=r["correct"]) for r in valid])
            metrics.update({f"{k}_{name}": m[k] for k in ("ece", "auroc", "brier")})
    metrics["answer_format_rate"] = len(valid) / len(rows)
    metrics["number_mass"] = statistics.fmean(r["number_mass"] for r in valid)
    metrics["check_tokens_mean"] = statistics.fmean(len(r["check"]) - 1 for r in valid)
    metrics["short_check_rate"] = statistics.fmean(len(r["check"]) <= 2 for r in valid)
    out_rows = [dict(answer=r["answer"], correct=r["correct"], check=h.tok.decode(r["check"]) if r["answer"] else None,
                     confidence=r.get("confidence"), pi_confidence=r.get("pi_confidence"), pi_swapped=r.get("pi_swapped"),
                     pi_nocheck=r.get("pi_nocheck")) for r in rows]
    return metrics, out_rows


def free_format_compliance(model, h, data, n=128):
    """Does the model follow the requested three-line format unaided? (base-model diagnostic)"""
    FastLanguageModel.for_inference(model)
    texts = []
    pad = h.tok.eos_token_id
    for start in range(0, min(n, len(data)), 32):
        prompts = [data[i]["query"] for i in range(start, min(start + 32, n, len(data)))]
        width = max(map(len, prompts))
        ids = torch.tensor([[pad] * (width - len(q)) + q for q in prompts]).cuda()
        mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in prompts]).cuda()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=200, do_sample=True, temperature=0.6,
                                 top_p=0.9, eos_token_id=[pad, h.eot], pad_token_id=pad)
        texts += [h.tok.decode(row[width:], skip_special_tokens=True) for row in out.tolist()]
    return dict(free_format_rate=sum(bool(FREE_FORMAT.match(t)) for t in texts) / len(texts), free_examples=texts[:12])


def train(model, h, data, args, log, samples):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng, row_rng = random.Random(args.seed), random.Random(args.seed + 1)
    group = args.group if args.arm in ("trained", "frozen") else 1
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batchsize):
            t0 = time.time()
            part = [data[i] for i in order[start:start + args.batchsize]]
            prompts = [d["query"] for d in part]
            FastLanguageModel.for_inference(model)
            answers = h.answers(model, prompts)
            questions = [dict(prefix=prefix_for(prompts[i], a[0], args.arm, h), answer=a[1],
                              label=float(is_correct_f1(a[1], part[i]["gt_candidates"])))
                         for i, a in enumerate(answers) if a is not None]
            if not questions:
                continue
            t_answer = time.time()
            checks = h.checks(model, [q["prefix"] for q in questions], args.arm, group)
            t_check = time.time()
            items = [dict(q=qi, label=q["label"], **item_for(q["prefix"], c, h))
                     for qi, q in enumerate(questions) for c in checks[qi]]
            FastLanguageModel.for_training(model)
            # Sampled confidence k* at T=1 from the current policy, for the stop check.
            for it, lg in zip(items, h.next_token_logits(model, [it["seq"] for it in items])):
                k = int(torch.multinomial(lg.softmax(-1), 1))
                it["k"] = h.numbers.index(k) if k in h.numbers else None
            if args.arm == "trained" and args.kl_beta > 0:
                with torch.no_grad(), model.disable_adapter():
                    for i in range(0, len(items), 8):
                        for it, (_, _, check) in zip(items[i:i + 8], h.score(model, items[i:i + 8], with_k=False)):
                            it["ref_check"] = check
            t_ref = time.time()
            stats = dict(J=[], kl=[], check_tokens=[], mass=[], stop=[], hinge_active=[], adv_abs=[])
            q_order = list(range(len(questions)))
            for p in range(args.passes):
                row_rng.shuffle(q_order)
                for mb_start in range(0, len(q_order), args.minibatch):
                    mb_q = q_order[mb_start:mb_start + args.minibatch]
                    optimizer.zero_grad()
                    for qi in mb_q:  # one backward per question bounds memory at G sequences
                        mine = [it for it in items if it["q"] == qi]
                        losses, rewards, check_logps = [], [], []
                        for it, (number_logps, stop_logp, check) in zip(mine, h.score(model, mine)):
                            lp = number_logps[None].double()
                            loss, _ = objective(lp, lp.new_tensor([it["label"]]), "discrete-exact", args.reward,
                                                args.format_weight, args.format_threshold, args.reward_mix)
                            if stop_logp is not None:
                                loss = loss + args.format_weight * (math.log(args.format_threshold) - stop_logp.double()).clamp(min=0)
                            kl = float((check.detach() - it["ref_check"]).sum()) if "ref_check" in it else 0.
                            losses.append(loss)
                            rewards.append(-float(loss) - args.kl_beta * kl)
                            check_logps.append(check.sum())
                            if p == 0:
                                mass = float(lp.detach().logsumexp(-1).exp())
                                stats["J"].append(-float(loss)); stats["kl"].append(kl)
                                stats["check_tokens"].append(it["span"][1] - it["span"][0])
                                stats["mass"].append(mass); stats["hinge_active"].append(float(mass < args.format_threshold))
                                if stop_logp is not None:
                                    stats["stop"].append(float(stop_logp.exp()))
                        total = torch.stack(losses).mean()
                        if args.arm == "trained" and len(mine) > 1:
                            pg = []
                            for j in range(len(mine)):
                                advantage = rewards[j] - statistics.fmean(rewards[:j] + rewards[j + 1:])
                                pg.append(-advantage * check_logps[j].double())
                                if p == 0:
                                    stats["adv_abs"].append(abs(advantage))
                            total = total + torch.stack(pg).mean()
                        (total / len(mb_q)).backward()
                    optimizer.step()
            step += 1
            record = dict(step=step, epoch=epoch, questions=len(questions),
                          answer_accuracy=statistics.fmean(q["label"] for q in questions),
                          seconds=time.time() - t0, answer_seconds=t_answer - t0, check_seconds=t_check - t_answer,
                          ref_seconds=t_ref - t_check, update_seconds=time.time() - t_ref,
                          **{k: (statistics.fmean(v) if v else None) for k, v in stats.items()})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if step % 8 == 1:
                for it in items[:2 * group]:
                    samples.write(json.dumps(dict(step=step, answer=questions[it["q"]]["answer"], label=it["label"],
                                                  check=h.tok.decode(it["seq"][it["span"][0]:it["span"][1]]))) + "\n")
                samples.flush()
            if args.max_steps and step >= args.max_steps:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids")
    parser.add_argument("out_dir")
    parser.add_argument("--arm", choices=["trained", "frozen", "filler", "none"], required=True)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--group", type=int, default=4, help="checks per question (arms trained and frozen)")
    parser.add_argument("--check-tokens", type=int, default=96)
    parser.add_argument("--lr", type=float, default=4.01e-05)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--format-weight", type=float, default=1.01)
    parser.add_argument("--format-threshold", type=float, default=0.95)
    parser.add_argument("--reward", default="paper")
    parser.add_argument("--reward-mix", type=float, default=0.0)
    parser.add_argument("--eval-only", action="store_true", help="score the base model on dev, no training")
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=1048, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing=False, random_state=3407)
    trl.trainer.peft_module_casting_to_bf16(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    h = Harness(tokenizer, args.check_tokens)
    ids = json.load(open(args.ids))
    prompt = NONE_PROMPT if args.arm == "none" else THINK_PROMPT
    dev = subset_loader(ids, prompt)("triviaqa", "validation", "verbalize", tokenizer)
    if args.eval_limit:
        dev = dev.select(range(args.eval_limit))
    extra = {}
    if args.eval_only:
        extra = free_format_compliance(model, h, dev)
        print(json.dumps(extra), flush=True)
    else:
        data = subset_loader(ids, prompt)("triviaqa", "train", "verbalize", tokenizer)
        with open(os.path.join(args.out_dir, "metrics.jsonl"), "w") as log, \
                open(os.path.join(args.out_dir, "check_samples.jsonl"), "w") as samples:
            train(model, h, data, args, log, samples)
        model.save_pretrained(os.path.join(args.out_dir, "adapter"))
    t_eval = time.time()
    metrics, rows = evaluate(model, h, dev, args.arm, args.seed)
    metrics.update(extra, arm=args.arm, kl_beta=args.kl_beta, eval_only=args.eval_only,
                   train_minutes=(t_eval - t_start) / 60, eval_minutes=(time.time() - t_eval) / 60)
    json.dump(metrics, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(args.out_dir, "eval_dev.json"), "w"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "free_examples"}), flush=True)


if __name__ == "__main__":
    main()
