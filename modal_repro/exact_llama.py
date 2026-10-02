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
* --regularization baseline (with --scoring single) replaces the hinges with the released PPO's
  own terms, computed from the same pass: the expected reward without renormalizing, invalid
  outcomes scored --invalid-reward (-30 with --reward released), and beta * KL to the base
  model (adapter disabled, one no-grad pass per question per step) with TRL's adaptive
  controller (beta 0.05, target 6, horizon 10000). See core.baseline_matched_objective.
* --value-head adds TRL 0.8.6's ValueHead (dropout 0.1 + Linear(hidden, 1) on the final hidden
  states) as an auxiliary task: at each response position (after " Confidence", ":", " " and the
  sampled number) it regresses the exact expected reward J (detached), loss
  vf_coef * 0.5 * mean((V - J)^2), vf_coef 0.1, trained by the same Adam. Exact scoring needs no
  baseline, so its only effect is the gradient it sends into the shared LoRA weights.
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

from rewarding_doubt.core import baseline_matched_objective, objective
from rewarding_doubt.paper_ppo import AdaptiveKLController
from shared_prefix import candidate_logps, full_sequence_logps, single_pass_logps, split_candidates
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries
from util.EvaluationMetrics import Metric, is_answer_correct
from util.ResponseHandling import parse_answer_confidence

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def reference_scores(model, query, k_star, candidates):
    """The base model's number log-probs and stop log-prob for this row (adapter disabled)."""
    common, numbers, stop = split_candidates(candidates)
    with torch.no_grad(), model.disable_adapter():
        logq, stop_logp = single_pass_logps(model, query, common, numbers, stop, k_star)
    return logq.double(), (stop_logp.double() if stop_logp is not None else None)


class ValueHead(torch.nn.Module):
    """trl.models.modeling_value_head.ValueHead (0.8.6): dropout then a linear map to a scalar."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.summary = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        return self.summary(self.dropout(hidden.to(self.summary.weight.dtype))).squeeze(-1)


def capture_final_hidden(model, store):
    """Keep the input to lm_head (the final hidden states) from every forward pass."""
    lm_head = model.base_model.model.lm_head
    return lm_head.register_forward_hook(lambda module, inputs, output: store.__setitem__("hidden", inputs[0]))


def row_loss(model, row, candidates, eot, args, beta=0., value_head=None, store=None):
    """Loss for one row: (loss, confidence, valid mass, stop prob, KL, value loss)."""
    query, label, k_star, ref = row
    stop_prob, kl = None, None
    if args.scoring == "single":
        common, numbers, stop = split_candidates(candidates)
        logps, stop_logp = single_pass_logps(model, query, common, numbers, stop, k_star)
        if stop_logp is not None:
            stop_prob = stop_logp.exp().item()
    else:
        logps, stop_logp = (candidate_logps(model, query, candidates) if args.scoring == "shared"
                            else full_sequence_logps(model, query, candidates, eot)), None
    if args.regularization == "baseline":
        logq = logps.double()
        J, kl_term = baseline_matched_objective(
            logq, label, args.mode, args.reward, args.invalid_reward,
            stop_logp.double() if stop_logp is not None else None, k_star, ref[0],
            ref[1] if stop_logp is not None else None)
        mass = logq.detach().exp().sum()
        confidence = (logq.detach().exp() / mass * logq.new_tensor([k / 10 for k in range(11)])).sum()
        loss, value_loss = -J + beta * kl_term, None
        if value_head is not None:
            hidden = store["hidden"][0]  # [sequence, hidden] from the forward pass above
            positions = torch.arange(len(query) - 1, hidden.shape[0], device=hidden.device)
            values = value_head(hidden[positions])
            value_loss = 0.5 * ((values.double() - J.detach()) ** 2).mean()
            loss = loss + args.vf_coef * value_loss
            value_loss = value_loss.item()
        return loss, confidence, mass.item(), stop_prob, kl_term.item(), value_loss
    logps = logps[None].double()
    loss, confidence = objective(logps, logps.new_tensor([label]), args.mode, args.reward,
                                 args.format_weight, args.format_threshold)
    if stop_logp is not None:
        loss = loss + args.format_weight * (math.log(args.format_threshold) - stop_logp.double()).clamp(min=0)
    return loss, confidence, logps.detach().logsumexp(-1).exp().item(), stop_prob, kl, None


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
    parser.add_argument("--regularization", choices=["hinge", "baseline"], default="hinge")
    parser.add_argument("--invalid-reward", type=float, default=None, help="default: -30 released, -3 paper")
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--kl-target", type=float, default=6.)
    parser.add_argument("--kl-horizon", type=float, default=10000.)
    parser.add_argument("--value-head", action="store_true")
    parser.add_argument("--vf-coef", type=float, default=0.1)
    parser.add_argument("--scoring", choices=["single", "shared", "full"], default="single",
                        help="single: one pass, 11 number probabilities + stop check on a sampled number; "
                             "shared: prefix pass + 11 cached continuations; full: 11 full sequences "
                             "(the runs before 2026-10-01T20Z)")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.regularization == "baseline" and args.scoring != "single":
        parser.error("--regularization baseline needs --scoring single")
    if args.invalid_reward is None:
        args.invalid_reward = -30. if args.reward == "released" else -3.
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
    value_head, store = None, {}
    if args.value_head:
        if args.regularization != "baseline":
            parser.error("--value-head regresses the baseline-matched reward; use --regularization baseline")
        value_head = ValueHead(model.config.hidden_size).cuda()
        capture_final_hidden(model, store)
    params = [p for p in model.parameters() if p.requires_grad] + (list(value_head.parameters()) if value_head else [])
    optimizer = torch.optim.Adam(params, lr=args.lr)
    generation = dict(max_new_tokens=256, eos_token_id=[tokenizer.eos_token_id, eot, confidence_token],
                      do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    kl_controller = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon)
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
                rows.append((prompt + answer, float(correct), None, None))
            if args.scoring == "single" and rows:
                picks = sample_numbers(model, tokenizer, [r[0] for r in rows], candidates, eot)
                rows = [(q, label, k, None) for (q, label, _, _), k in zip(rows, picks)]
            FastLanguageModel.for_training(model)
            if args.regularization == "baseline":  # reference scores are fixed for the whole step
                rows = [(q, label, k, reference_scores(model, q, k, candidates)) for q, label, k, _ in rows]
            stats = dict(loss=0., confidence=0., mass=0., hinge_active=0., stop_prob=0., sampled_number_rate=0.,
                         kl=0., kl_coef=kl_controller.value, value_loss=0.)
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
                        loss, confidence, mass, stop_prob, kl, value_loss = row_loss(
                            model, row, candidates, eot, args, kl_controller.value, value_head, store)
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
                            if kl is not None:
                                stats["kl"] += kl / len(rows)
                            if value_loss is not None:
                                stats["value_loss"] += value_loss / len(rows)
                    optimizer.step()
                    updates += 1
            stats["stop_prob"] = stats["stop_prob"] / stop_checks if stop_checks else None
            if args.regularization == "baseline":  # TRL updates after each batch, n_steps = batch size
                kl_controller.update(stats["kl"], args.batchsize)
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
