"""Prepare fixed answers, train independent adapters, and compare confidence readouts."""
import argparse
import hashlib
import importlib.metadata
from itertools import islice
import json
from pathlib import Path
import random
import re

import tinker
import torch

from . import paper_ppo
from .backend import (SYSTEM, candidate_datums, confidence_sequences, confidence_loss, datum,
                      prompt_tokens, score_candidates)
from .core import LEVELS, calibration_metrics, grade, normalize, parse_confidence, reward

PROMPTS = {"ours": SYSTEM, "paper": paper_ppo.SYSTEM}


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    from datasets import load_dataset
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Opening exclusively prevents overwriting an expensive answer cache.
    with output.open("x") as stream:
        data = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=args.split,
                            streaming=True)
        sampler = tinker.ServiceClient().create_sampling_client(base_model=args.model)
        tokenizer = sampler.get_tokenizer()
        rows = list(islice(data, args.limit))
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            futures = [sampler.sample(
                tinker.ModelInput.from_ints(prompt_tokens(tokenizer, row["question"])),
                num_samples=1, sampling_params=tinker.SamplingParams(
                    max_tokens=256, temperature=0.6, top_p=0.9, seed=args.seed + start + j))
                for j, row in enumerate(batch)]
            for j, (row, future) in enumerate(zip(batch, futures)):
                i = start + j
                seq = future.result().sequences[0]
                raw = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                match = re.fullmatch(r"\s*Answer:\s*(.+?),\s*Confidence:\s*(?:10|[0-9])\s*",
                                     raw, re.DOTALL)
                answer = match[1].strip() if match else ""
                references = list(dict.fromkeys([row["answer"]["value"],
                                                *row["answer"]["aliases"],
                                                *row["answer"]["normalized_aliases"]]))
                record = dict(id=row["question_id"], question=row["question"], answer=answer,
                              references=references, answer_format_valid=match is not None,
                              raw_answer_generation=raw, model=args.model, split=args.split,
                              source="mandarjoshi/trivia_qa:rc.nocontext", seed=args.seed + i)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"cached {i + 1}/{args.limit}", flush=True)


def is_correct(answer, references, grading, multiple_choice=False):
    if grading == "exact":
        return any(normalize(answer) == normalize(a) for a in references)
    return grade(answer, references, multiple_choice)


def examples(path, grading):
    rows = read_rows(path)
    # Never train on an empty answer fabricated from a failed parse.
    valid = [r for r in rows if r.get("answer_format_valid", True)]
    if not valid:
        raise ValueError("No valid cached answers")
    for r in valid:
        if not r["answer"].strip() or not r["references"]:
            raise ValueError("Every valid row needs a nonempty answer and references")
        r["correct"] = is_correct(r["answer"], r["references"], grading, r.get("multiple_choice", False))
    return valid, len(rows) - len(valid)


def paper_ppo_step(args, client, reference, tokenizer, batch, kl, seed):
    """One batch of the released Rewarding Doubt training loop (Train.py + TRL 0.8.6 PPO).

    The answer is sampled on-policy and then treated as part of the query; only the confidence
    continuation is the PPO response. Differences from TRL: no value head (V = 0, see
    paper_ppo.advantages_without_value_head) and no ratio_threshold batch skipping.
    """
    cfg = paper_ppo.CONFIG
    terminators = [t for t in {tokenizer.eos_token_id, tokenizer.pad_token_id} if t is not None]
    confidence_token = tokenizer.encode(" Confidence", add_special_tokens=False)
    if len(confidence_token) != 1:
        raise ValueError("Answer-stage stop needs ' Confidence' to be a single token")
    sampler = client.save_weights_and_get_sampling_client()
    prompts = [prompt_tokens(tokenizer, r["question"], system=PROMPTS[args.prompt]) for r in batch]
    # Train.py generation_kwargs_prediction: stop after "Answer: ..., Confidence".
    answer_futures = [sampler.sample(tinker.ModelInput.from_ints(p), 1, tinker.SamplingParams(
        max_tokens=256, temperature=0.6, top_p=0.9, stop=terminators + confidence_token, seed=seed + j))
        for j, p in enumerate(prompts)]
    queries, answer_stopped = [], []
    for p, future in zip(prompts, answer_futures):
        answer = list(future.result().sequences[0].tokens)
        answer_stopped.append(answer[-1:] == confidence_token)
        # util.remove_padding strips trailing pad tokens, and pad == eos in Train.py.
        while answer and answer[-1] == tokenizer.eos_token_id:
            answer.pop()
        queries.append(p + answer)
    # Train.py generation_kwargs_ppo: plain sampling (top_k 0, top_p 1, T=1), max_new_tokens 500.
    response_futures = [sampler.sample(tinker.ModelInput.from_ints(q), 1, tinker.SamplingParams(
        max_tokens=500, temperature=1, top_p=1, stop=terminators, seed=seed + len(batch) + j))
        for j, q in enumerate(queries)]
    responses = [f.result().sequences[0] for f in response_futures]
    ref_futures = [reference.compute_logprobs(tinker.ModelInput.from_ints(q + list(s.tokens)))
                   for q, s in zip(queries, responses)]
    scores, confidences, correct, rollouts = [], [], [], []
    for row, p, q, s in zip(batch, prompts, queries, responses):
        text = tokenizer.decode(q[len(p):] + list(s.tokens), skip_special_tokens=True)
        rollouts.append(dict(id=row.get("id"), text=text))
        answer, confidence = paper_ppo.parse_response(text)
        ok = answer is not None and is_correct(answer, row["references"], args.grading)
        correct.append(ok if answer is not None else None)
        confidences.append(confidence)
        scores.append((-30. if args.reward == "released" else -3.) if confidence is None
                      else float(reward(confidence / 10, ok, args.reward)))
        rollouts[-1].update(correct=correct[-1], confidence=confidence, score=scores[-1])
    logps = [list(s.logprobs) for s in responses]
    ref_logps = [list(f.result()[-len(s.tokens):]) for f, s in zip(ref_futures, responses)]
    if any(v is None for lps in ref_logps for v in lps):
        raise ValueError("Reference model omitted a response log probability")
    rewards = [paper_ppo.token_rewards(score, lp, ref, kl.value)
               for score, lp, ref in zip(scores, logps, ref_logps)]
    advantages = paper_ppo.whiten([paper_ppo.advantages_without_value_head(r, cfg["gamma"], cfg["lam"])
                                   for r in rewards])
    # torch.optim.Adam defaults, which TRL uses.
    adam = tinker.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=1e-8)
    rng = random.Random(seed)
    mini = max(1, len(batch) // 2)  # Train.py: mini_batch_size = batchsize / 2
    futures = []
    for _ in range(cfg["ppo_epochs"]):
        order = list(range(len(batch)))
        rng.shuffle(order)
        for start in range(0, len(order), mini):
            idx = order[start:start + mini]
            # Tinker's ppo loss sums over tokens; TRL takes a masked mean over the minibatch.
            n_tokens = sum(len(responses[i].tokens) for i in idx)
            datums = [datum(queries[i], list(responses[i].tokens), old_logps=logps[i],
                            advantages=[a / n_tokens for a in advantages[i]]) for i in idx]
            # Submit without waiting so forward/backward and optimizer steps pipeline.
            futures.append(client.forward_backward(datums, loss_fn="ppo"))
            futures.append(client.optim_step(adam))
    losses = [f.result().metrics.get("loss:sum", 0.) for f in futures[::2]]
    for f in futures[1::2]:
        f.result()
    mean_kl = sum(sum(lp) - sum(ref) for lp, ref in zip(logps, ref_logps)) / len(batch)
    kl_coef = kl.value
    kl.update(mean_kl, len(batch))
    valid = [c for c in confidences if c is not None]
    graded = [c for c in correct if c is not None]
    return dict(score=sum(scores) / len(scores), invalid_rate=1 - len(valid) / len(batch),
                answer_stage_stopped=sum(answer_stopped) / len(batch),
                answer_accuracy=sum(graded) / len(graded) if graded else None,
                confidence_mean=sum(valid) / 10 / len(valid) if valid else None,
                response_tokens=sum(len(s.tokens) for s in responses) / len(batch),
                kl=mean_kl, kl_coef=kl_coef, ppo_loss_mean=sum(losses) / len(losses)), rollouts


def train(args):
    if args.mode == "paper-ppo":
        # Answers are generated on-policy, so every question is usable regardless of its cache.
        rows, excluded = read_rows(args.data), 0
    else:
        rows, excluded = examples(args.data, args.grading)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.pop("func", None)
    config.update(data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  excluded_invalid_answers=excluded, tinker_version=importlib.metadata.version("tinker"))
    write_json(out / "config.json", config)
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=args.model, rank=args.rank, seed=args.seed)
    tokenizer = client.get_tokenizer()
    if args.mode in ("discrete-exact", "fractional"):
        sequences = confidence_sequences(tokenizer)
        lengths = [len(s) for s in sequences]
    rng = random.Random(args.seed)
    step = 0
    snapshots = []
    if args.mode == "paper-ppo":
        reference = service.create_sampling_client(base_model=args.model)
        kl = paper_ppo.AdaptiveKLController(paper_ppo.CONFIG["init_kl_coef"], paper_ppo.CONFIG["kl_target"],
                                            paper_ppo.CONFIG["kl_horizon"])
    with (out / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            order = rows.copy()
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                if args.mode == "paper-ppo":
                    metrics, rollouts = paper_ppo_step(args, client, reference, tokenizer, batch, kl,
                                                       args.seed + 1000 * step)
                    step += 1
                    with (out / "rollouts.jsonl").open("a") as stream:
                        stream.writelines(json.dumps(dict(step=step, **r)) + "\n" for r in rollouts)
                    record = dict(step=step, epoch=epoch, **metrics)
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                    if args.save_every and step % args.save_every == 0:
                        path = client.save_weights_for_sampler(f"step-{step:05d}").result().path
                        snapshots.append(dict(step=step, sampler_path=path))
                        write_json(out / "snapshots.json", snapshots)
                    if args.max_steps and step >= args.max_steps:
                        break
                    continue
                prefixes = [prompt_tokens(tokenizer, r["question"], r["answer"], PROMPTS[args.prompt])
                            for r in batch]
                labels = [float(r["correct"]) for r in batch]
                if args.mode == "discrete-ppo":
                    sampler = client.save_weights_and_get_sampling_client()
                    futures = [sampler.sample(tinker.ModelInput.from_ints(p), args.group_size,
                               tinker.SamplingParams(max_tokens=8, temperature=1, top_p=1,
                                                     seed=args.seed + step * args.batch_size + j))
                               for j, p in enumerate(prefixes)]
                    datums, rewards, invalid = [], [], 0
                    for prefix, label, future in zip(prefixes, labels, futures):
                        rollouts = future.result().sequences
                        group_rewards = []
                        for seq in rollouts:
                            p = parse_confidence(tokenizer.decode(seq.tokens, skip_special_tokens=True))
                            invalid += p is None
                            group_rewards.append((-30. if args.reward == "released" else -3.) if p is None
                                                 else float(reward(p, label, args.reward)))
                        baseline = sum(group_rewards) / len(group_rewards)
                        for seq, r in zip(rollouts, group_rewards):
                            if seq.logprobs is None:
                                raise ValueError("Sampling must return behavior logprobs")
                            # Leave-one-out baseline avoids the (G-1)/G shrinkage.
                            advantage = (r - baseline) * len(rollouts) / (len(rollouts) - 1)
                            datums.append(datum(prefix, seq.tokens, old_logps=seq.logprobs,
                                                advantages=[advantage] * len(seq.tokens)))
                        rewards.extend(group_rewards)
                    result = client.forward_backward(datums, loss_fn="ppo").result()
                    metrics = dict(reward=sum(rewards) / len(rewards), invalid_rate=invalid / len(rewards))
                else:
                    datums = candidate_datums(prefixes, sequences)
                    result = client.forward_backward_custom(
                        datums, confidence_loss(labels, args.mode, args.reward, lengths,
                                                args.format_weight, args.format_threshold)).result()
                    metrics = {}
                client.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
                step += 1
                record = dict(step=step, epoch=epoch, **metrics, tinker=result.metrics)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if args.save_every and step % args.save_every == 0:
                    # Sampler-only weights: enough to evaluate or deploy an intermediate step.
                    path = client.save_weights_for_sampler(f"step-{step:05d}").result().path
                    snapshots.append(dict(step=step, sampler_path=path))
                    write_json(out / "snapshots.json", snapshots)
                if args.max_steps and step >= args.max_steps:
                    break
            if args.max_steps and step >= args.max_steps:
                break
    state = client.save_state("final").result()
    weights = client.save_weights_for_sampler("final").result()
    write_json(out / "checkpoint.json", dict(state_path=state.path, sampler_path=weights.path,
                                            steps=step, model=args.model, snapshots=snapshots))


def evaluate(args):
    rows, excluded = examples(args.data, args.grading)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    service = tinker.ServiceClient()
    sampler = (service.create_sampling_client(model_path=args.checkpoint) if args.checkpoint else
               service.create_sampling_client(base_model=args.model))
    tokenizer = sampler.get_tokenizer()
    sequences = confidence_sequences(tokenizer)
    records = []
    with (output / "predictions.jsonl").open("w") as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prefixes = [prompt_tokens(tokenizer, r["question"], r["answer"], PROMPTS[args.prompt])
                        for r in batch]
            logps = score_candidates(sampler, prefixes, sequences)
            probs = logps.softmax(-1)
            means = probs @ probs.new_tensor(LEVELS)
            futures = [sampler.sample(tinker.ModelInput.from_ints(p), 1,
                       tinker.SamplingParams(max_tokens=8, temperature=1, top_p=1,
                                             seed=args.seed + start + j))
                       for j, p in enumerate(prefixes)]
            for j, (row, future) in enumerate(zip(batch, futures)):
                raw = tokenizer.decode(future.result().sequences[0].tokens, skip_special_tokens=True)
                record = dict(id=row.get("id", start + j), correct=row["correct"],
                              fractional=float(means[j]), argmax=float(probs[j].argmax()) / 10,
                              sampled=parse_confidence(raw), raw_confidence=raw,
                              valid_confidence_mass=float(logps[j].logsumexp(-1).exp()),
                              candidate_logprobs=logps[j].tolist())
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    report = dict(excluded_invalid_answers=excluded, total_valid_answers=len(records),
                  invalid_confidence_rate=sum(r["sampled"] is None for r in records) / len(records),
                  mean_valid_confidence_mass=sum(r["valid_confidence_mass"] for r in records) / len(records),
                  data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  model=args.model, checkpoint=args.checkpoint, grading=args.grading, bins=args.bins)
    for name in ["fractional", "argmax", "sampled"]:
        valid = [r for r in records if r[name] is not None]
        report[name] = calibration_metrics([r[name] for r in valid], [r["correct"] for r in valid],
                                            args.bins) if valid else None
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2))


def evaluate_generate(args):
    """The released evaluation protocol (InferenceDatasetSplit.py + Evaluation.get_all_metrics).

    The model generates answer and confidence together from the question alone; there are no
    cached answers. Rows need `question` and `references` (TriviaQA normalized aliases, the
    released code's gt_candidates).
    """
    rows = read_rows(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    service = tinker.ServiceClient()
    sampler = (service.create_sampling_client(model_path=args.checkpoint) if args.checkpoint else
               service.create_sampling_client(base_model=args.model))
    tokenizer = sampler.get_tokenizer()
    stop = [t for t in {tokenizer.eos_token_id, tokenizer.pad_token_id} if t is not None]
    futures = [sampler.sample(tinker.ModelInput.from_ints(
                   prompt_tokens(tokenizer, r["question"], system=PROMPTS[args.prompt])), 1,
               tinker.SamplingParams(max_tokens=32, temperature=0.6, top_p=0.9, stop=stop, seed=args.seed + i))
               for i, r in enumerate(rows)]
    records = []
    with (output / "predictions.jsonl").open("w") as stream:
        for i, (row, future) in enumerate(zip(rows, futures)):
            text = tokenizer.decode(future.result().sequences[0].tokens, skip_special_tokens=True)
            answer, confidence = paper_ppo.parse_response(text)
            record = dict(id=row.get("id", i), response=text, answer=answer, confidence=confidence,
                          correct=paper_ppo.is_correct_f1(answer, row["references"]))
            records.append(record)
            stream.write(json.dumps(record) + "\n")
    report = dict(paper_ppo.evaluation_metrics(records), model=args.model, checkpoint=args.checkpoint,
                  prompt=args.prompt, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Cache base-model answers to TriviaQA")
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--limit", type=int, default=128)
    p.set_defaults(func=prepare)
    p = sub.add_parser("train")
    p.add_argument("--mode", choices=["discrete-ppo", "discrete-exact", "fractional", "paper-ppo"],
                   required=True, help="paper-ppo: released Rewarding Doubt PPO loop (on-policy answers, "
                                       "KL to base, 4 PPO epochs); ignores cached answers")
    p.add_argument("--reward", choices=["paper", "released"], default="paper")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--format-weight", type=float, default=1.0,
                   help="exact modes: weight of the valid-mass hinge penalty (0 disables)")
    p.add_argument("--format-threshold", type=float, default=0.95,
                   help="exact modes: valid confidence mass below which the hinge activates")
    p.add_argument("--save-every", type=int, default=40,
                   help="save sampler weights every N steps (0 disables)")
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", help="tinker:// sampler weights path; omit for base model")
    p.add_argument("--bins", type=int, default=11)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("evaluate-generate", help="Released evaluation protocol: generate answer + confidence")
    p.add_argument("--checkpoint", help="tinker:// sampler weights path; omit for base model")
    p.set_defaults(func=evaluate_generate)
    for name, p in sub.choices.items():
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--seed", type=int, default=2)
        p.add_argument("--output", required=True)
        if name != "prepare":
            p.add_argument("--data", required=True)
            p.add_argument("--prompt", choices=sorted(PROMPTS), default="ours",
                           help="system prompt: ours, or the released code's TriviaQA prompt")
            p.add_argument("--grading", choices=["f1", "exact"], default="f1")
            p.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    for name in ["limit", "batch_size", "epochs", "rank", "bins", "lr"]:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "group_size", 2) < 2 or getattr(args, "max_steps", 0) < 0:
        parser.error("group-size must be >= 2 and max-steps >= 0")
    if getattr(args, "format_weight", 0) < 0 or getattr(args, "save_every", 0) < 0:
        parser.error("format-weight and save-every must be >= 0")
    if not 0 < getattr(args, "format_threshold", .5) < 1:
        parser.error("format-threshold must be in (0, 1)")
    args.func(args)


if __name__ == "__main__":
    main()
