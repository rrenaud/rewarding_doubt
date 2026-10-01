"""Our exact confidence objectives on the released code's model, data, prompt and evaluation.

Runs inside the Modal container from SingleAnswerSetting/. Everything matches the released
Train.py except the update rule:

* model: 4-bit Unsloth Llama-3-8B-Instruct, LoRA as in util/ModelLoader.load_lora_model_tokenizer
  (r=8, alpha=8, dropout 0, Unsloth gradient checkpointing, random_state 3407); no value head
* data: TriviaQA `unfiltered` through util.DataHelper (released system prompt), subset by ID
* each step: the current policy samples "Answer: ..., Confidence" exactly as Train.py does
  (T=0.6, top-p 0.9, 256 tokens, stop at " Confidence"); the answer is graded with the released
  F1 > 0.5 code; then all 11 continuations ": k<eot>" are teacher-forced and the loss is
  rewarding_doubt.core.objective (the same function as the Tinker runs), averaged over rows
* optimizer: torch.optim.Adam(lr), as TRL uses; batch 8; 2 epochs; torch.manual_seed(2)
* --scoring single (default): after the answer, the policy samples its confidence continuation at
  T=1 (as Train.py's second generate does); one forward pass over query + ": " + sampled number
  gives log P(k) for all 11 numbers (one softmax) and log P(<eot> | sampled k). The loss is the
  objective on the 11 numbers, the hinge on their total mass, and a stop hinge
  format_weight * relu(log threshold - log P(<eot> | k)). `shared` and `full` score the 11
  complete strings ": k<eot>" instead.
* --passes P --minibatch M reuse each sampled batch like TRL's ppo_epochs / mini_batch_size:
  P passes over the batch in minibatches of M, one Adam step per minibatch. The exact loss is a
  deterministic function of (question, answer, label), so reuse needs no importance weights.
  --passes 4 --minibatch 4 matches PPO's 8 optimizer steps per batch of 8.

    python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact|fractional [--passes 4 --minibatch 4]
                          [--save-every 32] [--max-steps N]
"""
import argparse
import json
import math
import os
import random
import time

import torch
from unsloth import FastLanguageModel  # must precede transformers imports
import trl

from rewarding_doubt.core import objective
from shared_prefix import candidate_logps, full_sequence_logps, single_pass_logps, split_candidates
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries
from util.EvaluationMetrics import Metric, is_answer_correct
from util.ResponseHandling import parse_answer_confidence

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def row_loss(model, row, candidates, eot, args):
    """Loss for one (query, label, sampled number) row: (loss, confidence, valid mass, stop prob)."""
    query, label, k_star = row
    stop_prob = None
    if args.scoring == "single":
        common, numbers, stop = split_candidates(candidates)
        logps, stop_logp = single_pass_logps(model, query, common, numbers, stop, k_star)
    else:
        logps, stop_logp = (candidate_logps(model, query, candidates) if args.scoring == "shared"
                            else full_sequence_logps(model, query, candidates, eot)), None
    logps = logps[None].double()
    loss, confidence = objective(logps, logps.new_tensor([label]), args.mode, args.reward,
                                 args.format_weight, args.format_threshold)
    if stop_logp is not None:
        loss = loss + args.format_weight * (math.log(args.format_threshold) - stop_logp.double()).clamp(min=0)
        stop_prob = stop_logp.exp().item()
    return loss, confidence, logps.detach().logsumexp(-1).exp().item(), stop_prob


def sample_numbers(model, tokenizer, queries, candidates, eot):
    """Sample each query's confidence continuation at T=1 and return the sampled number index.

    None when the continuation is not ": <number>" (the mass hinge covers that case).
    """
    common, numbers, _ = split_candidates(candidates)
    width = max(map(len, queries))
    input_ids = torch.tensor([[tokenizer.eos_token_id] * (width - len(q)) + q for q in queries]).cuda()
    mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in queries]).cuda()
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=len(common) + 2,
                             do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             eos_token_id=[tokenizer.eos_token_id, eot], pad_token_id=tokenizer.eos_token_id)
    picks = []
    for row in out[:, width:].tolist():
        ok = row[:len(common)] == common and len(row) > len(common) and row[len(common)] in numbers
        picks.append(numbers.index(row[len(common)]) if ok else None)
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids")
    parser.add_argument("out_dir")
    parser.add_argument("--mode", choices=["discrete-exact", "fractional"], required=True)
    parser.add_argument("--reward", default="paper")
    parser.add_argument("--format-weight", type=float, default=1.0)
    parser.add_argument("--format-threshold", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=0, help="default: the whole batch")
    parser.add_argument("--save-every", type=int, default=0, help="also save adapters every N steps")
    parser.add_argument("--scoring", choices=["single", "shared", "full"], default="single",
                        help="single: one pass, 11 number probabilities + stop check on a sampled number; "
                             "shared: prefix pass + 11 cached continuations; full: 11 full sequences "
                             "(the runs before 2026-10-01T20Z)")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # util/ModelLoader.load_lora_model_tokenizer (is_unsloth=True), minus the value head.
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=1048,
                                                         dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing="unsloth", random_state=3407,
                                             use_rslora=False, loftq_config=None)
    # ModelLoader applies this (via the value-head wrapper) so LoRA weights are bf16, which
    # Unsloth's fused LoRA backward requires.
    trl.trainer.peft_module_casting_to_bf16(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    confidence_token = tokenizer.convert_tokens_to_ids("ĠConfidence")
    # The continuation PPO trains on after " Confidence": ": k" then end of turn.
    candidates = [tokenizer.encode(f": {k}", add_special_tokens=False) + [eot] for k in range(11)]
    assert len({tuple(c) for c in candidates}) == 11
    assert not any(a != b and b[:len(a)] == a for a in candidates for b in candidates)

    ids = json.load(open(args.ids))
    data = subset_loader(ids)("triviaqa", "train", "verbalize", tokenizer)
    collate = DataCollatorForTokenizedQueries(tokenizer)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    generation = dict(max_new_tokens=256, eos_token_id=[tokenizer.eos_token_id, eot, confidence_token],
                      do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    rng = random.Random(args.seed)  # question order: identical for every --passes setting
    row_rng = random.Random(args.seed + 1)
    step = 0
    log = open(os.path.join(args.out_dir, "metrics.jsonl"), "w")
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batchsize):
            t0 = time.time()
            batch = collate([data[i] for i in order[start:start + args.batchsize]])
            FastLanguageModel.for_inference(model)
            with torch.no_grad():
                out = model.generate(input_ids=batch["input_ids"].cuda(),
                                     attention_mask=batch["attention_mask"].cuda(), **generation)
            rows = []
            for i in range(len(out)):
                prompt = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
                answer = out[i][batch["input_ids"].shape[1]:].tolist()
                while answer and answer[-1] == tokenizer.eos_token_id:  # util.remove_padding
                    answer.pop()
                if not answer or answer[-1] != confidence_token:
                    continue  # answer never reached " Confidence"; PPO would score it invalid
                text = tokenizer.decode(answer, skip_special_tokens=True) + ": 0"
                prediction, _ = parse_answer_confidence(text, False)
                if prediction is None:
                    continue
                correct = is_answer_correct(prediction, batch["gt_candidates"][i], Metric.F1, 0.5)
                rows.append((prompt + answer, float(correct), None))
            if args.scoring == "single" and rows:
                picks = sample_numbers(model, tokenizer, [q for q, _, _ in rows], candidates, eot)
                rows = [(q, label, k) for (q, label, _), k in zip(rows, picks)]
            FastLanguageModel.for_training(model)
            stats = dict(loss=0., confidence=0., mass=0., hinge_active=0., stop_prob=0., sampled_number_rate=0.)
            stop_checks = 0
            minibatch = args.minibatch or args.batchsize
            updates = 0
            for epoch_pass in range(args.passes):
                order_rows = list(range(len(rows)))
                row_rng.shuffle(order_rows)
                for mb_start in range(0, len(order_rows), minibatch):
                    mb = [rows[i] for i in order_rows[mb_start:mb_start + minibatch]]
                    optimizer.zero_grad()
                    for row in mb:
                        loss, confidence, mass, stop_prob = row_loss(model, row, candidates, eot, args)
                        (loss / len(mb)).backward()
                        if epoch_pass == 0:  # log pre-update statistics, once per row
                            stats["loss"] += loss.item() / len(rows)
                            stats["confidence"] += confidence.item() / len(rows)
                            stats["mass"] += mass / len(rows)
                            stats["hinge_active"] += (mass < args.format_threshold) / len(rows)
                            stats["sampled_number_rate"] += (row[2] is not None) / len(rows)
                            if stop_prob is not None:
                                stats["stop_prob"] += stop_prob
                                stop_checks += 1
                    optimizer.step()
                    updates += 1
            stats["stop_prob"] = stats["stop_prob"] / stop_checks if stop_checks else None
            step += 1
            record = dict(step=step, epoch=epoch, rows=len(rows), skipped=len(out) - len(rows),
                          answer_accuracy=sum(r[1] for r in rows) / max(1, len(rows)), updates=updates,
                          seconds=time.time() - t0, time=time.time(), **stats)
            if args.save_every and step % args.save_every == 0:
                t_save = time.time()
                model.save_pretrained(os.path.join(args.out_dir, f"snapshot-step{step:05d}"))
                tokenizer.save_pretrained(os.path.join(args.out_dir, f"snapshot-step{step:05d}"))
                record["save_seconds"] = time.time() - t_save
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        # Train.py saves model_finetuned after every epoch; save the adapter per epoch.
        model.save_pretrained(os.path.join(args.out_dir, f"model_finetuned_epoch{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"model_finetuned_epoch{epoch + 1}"))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
