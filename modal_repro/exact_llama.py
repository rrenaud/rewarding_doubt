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
* --passes P --minibatch M reuse each sampled batch like TRL's ppo_epochs / mini_batch_size:
  P passes over the batch in minibatches of M, one Adam step per minibatch. The exact loss is a
  deterministic function of (question, answer, label), so reuse needs no importance weights.
  --passes 4 --minibatch 4 matches PPO's 8 optimizer steps per batch of 8.

    python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact|fractional [--passes 4 --minibatch 4]
                          [--save-every 32] [--max-steps N]
"""
import argparse
import json
import os
import random
import time

import torch
from unsloth import FastLanguageModel  # must precede transformers imports
import trl

from rewarding_doubt.core import objective
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries
from util.EvaluationMetrics import Metric, is_answer_correct
from util.ResponseHandling import parse_answer_confidence

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def row_loss(model, query, label, candidates, eot, args):
    """Teacher-force the 11 continuations after one query; return (loss, confidence, valid mass)."""
    seqs = [query + c for c in candidates]
    width = max(map(len, seqs))
    input_ids = torch.tensor([s + [eot] * (width - len(s)) for s in seqs]).cuda()
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs]).cuda()
    logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
    logps = []
    for k, c in enumerate(candidates):
        positions = torch.arange(len(query) - 1, len(query) - 1 + len(c), device=logits.device)
        token_logps = logits[k, positions].log_softmax(-1).gather(-1, torch.tensor(c, device=logits.device)[:, None])
        logps.append(token_logps.sum())
    logps = torch.stack(logps)[None].double()
    loss, confidence = objective(logps, logps.new_tensor([label]), args.mode, args.reward,
                                 args.format_weight, args.format_threshold)
    return loss, confidence, logps.detach().logsumexp(-1).exp().item()


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
                rows.append((prompt + answer, float(correct)))
            FastLanguageModel.for_training(model)
            stats = dict(loss=0., confidence=0., mass=0., hinge_active=0.)
            minibatch = args.minibatch or args.batchsize
            updates = 0
            for epoch_pass in range(args.passes):
                order_rows = list(range(len(rows)))
                row_rng.shuffle(order_rows)
                for mb_start in range(0, len(order_rows), minibatch):
                    mb = [rows[i] for i in order_rows[mb_start:mb_start + minibatch]]
                    optimizer.zero_grad()
                    for query, label in mb:
                        loss, confidence, mass = row_loss(model, query, label, candidates, eot, args)
                        (loss / len(mb)).backward()
                        if epoch_pass == 0:  # log pre-update statistics, once per row
                            stats["loss"] += loss.item() / len(rows)
                            stats["confidence"] += confidence.item() / len(rows)
                            stats["mass"] += mass / len(rows)
                            stats["hinge_active"] += (mass < args.format_threshold) / len(rows)
                    optimizer.step()
                    updates += 1
            step += 1
            record = dict(step=step, epoch=epoch, rows=len(rows), skipped=len(out) - len(rows),
                          answer_accuracy=sum(l for _, l in rows) / max(1, len(rows)), updates=updates,
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
