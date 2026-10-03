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

* --frozen-answers: answers are generated with the adapter disabled (the base model), so training
  the confidence cannot change them; only the confidence continuation comes from the adapter.
  Evaluate such adapters with frozen_eval.py (base answer, then adapted confidence). Without it,
  the long runs drifted: answers that never reach " Confidence" are dropped from the loss, so
  nothing resisted "Answer: Answer: Answer: ..." repetition or hedges like "None, I made a
  mistake!" written into the answer, and answer accuracy fell 3-5 points over 4,000 steps.
* stability: every step feeds rewarding_doubt.stability.StabilityMonitor (sampled confidences,
  answer correctness, loss, gradient norm, entropy of pi) and appends to stability.jsonl; flags
  that rise go to stability_events.jsonl. With --stop-on FLAGS the run checkpoints and exits
  (code 3) when one of them rises.
* resume: every --checkpoint-every steps (and on SIGTERM, e.g. spot preemption) the LoRA weights,
  value head, Adam state, KL controller, RNG streams, monitor and loop position are saved to
  OUT_DIR/checkpoint (rewarding_doubt.checkpoint). Rerunning the same command resumes from it;
  logs are truncated to the checkpoint step and continue. Generation runs on the GPU, so a resumed
  run follows the same question order but is not bit-identical to an uninterrupted one.

    python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact|fractional [--passes 4 --minibatch 4]
                          [--save-every 32] [--max-steps N] [--checkpoint-every 32] [--stop-on collapsed]
"""
import argparse
import contextlib
import json
import math
import os
import random
import signal
import sys
import time
import zlib

import torch
from unsloth import FastLanguageModel  # must precede transformers imports
import trl

from rewarding_doubt.checkpoint import (load_checkpoint, load_trainable_state, rng_state, save_rotating_checkpoint,
                                        set_rng_state, trainable_state, truncate_jsonl, write_status)
from rewarding_doubt.core import baseline_matched_objective, objective
from rewarding_doubt.generations import GenerationLog
from rewarding_doubt.stability import FLAGS, StabilityMonitor
from rewarding_doubt.tracking import Tracker
from rewarding_doubt.paper_ppo import AdaptiveKLController
from shared_prefix import LevelScheme, candidate_logps, end_of_turn, full_sequence_logps
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries
from util.EvaluationMetrics import Metric, is_answer_correct
from util.ResponseHandling import parse_answer_confidence

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def reference_scores(model, rows, scheme):
    """The base model's level log-probs and stop log-prob per row (adapter disabled, one batch)."""
    with torch.no_grad(), model.disable_adapter():
        scored = scheme.batch(model, [r[0] for r in rows], [r[2] for r in rows])
    return [(logq.double(), stop_logp.double() if stop_logp is not None else None) for logq, stop_logp in scored]


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


def score_rows(model, rows, candidates, eot, args, scheme):
    """(level log-probs, stop log-prob or None) per row; single-pass rows share one forward call."""
    if args.scoring == "single":
        return scheme.batch(model, [r[0] for r in rows], [r[2] for r in rows])
    score = candidate_logps if args.scoring == "shared" else lambda m, q, c: full_sequence_logps(m, q, c, eot)
    return [(score(model, r[0], candidates), None) for r in rows]


def row_objective(row, logps, stop_logp, args, beta=0., value_head=None, hidden=None):
    """Loss for one scored row: (loss, confidence, valid mass, stop prob, KL, value loss).

    `hidden` is this row's final hidden states [sequence, hidden], needed for the value head.
    """
    query, label, k_star, ref = row
    stop_prob = stop_logp.exp().item() if stop_logp is not None else None
    if args.regularization == "baseline":
        logq = logps.double()
        J, kl_term = baseline_matched_objective(
            logq, label, args.mode, args.reward, args.invalid_reward,
            stop_logp.double() if stop_logp is not None else None, k_star, ref[0],
            ref[1] if stop_logp is not None else None, args.reward_mix)
        mass = logq.detach().exp().sum()
        confidence = (logq.detach().exp() / mass * logq.new_tensor([k / 10 for k in range(11)])).sum()
        loss, value_loss = -J + beta * kl_term, None
        if value_head is not None:
            # States after " Confidence", each common token (": "), and the sampled number if any.
            positions = torch.arange(len(query) - 1, len(query) + args.n_common + (k_star is not None),
                                     device=hidden.device)
            values = value_head(hidden[positions])
            value_loss = 0.5 * ((values.double() - J.detach()) ** 2).mean()
            loss = loss + args.vf_coef * value_loss
            value_loss = value_loss.item()
        return loss, confidence, mass.item(), stop_prob, kl_term.item(), value_loss
    logps = logps[None].double()
    loss, confidence = objective(logps, logps.new_tensor([label]), args.mode, args.reward,
                                 args.format_weight, args.format_threshold, args.reward_mix)
    if stop_logp is not None:
        loss = loss + args.format_weight * (math.log(args.format_threshold) - stop_logp.double()).clamp(min=0)
    return loss, confidence, logps.detach().logsumexp(-1).exp().item(), stop_prob, None, None


def sample_numbers(model, tokenizer, queries, scheme, eot):
    """Sample each query's confidence continuation at T=1 and return the sampled level, or None
    when the continuation is not ": <level>" (the mass hinge covers that case)."""
    width = max(map(len, queries))
    input_ids = torch.tensor([[tokenizer.eos_token_id] * (width - len(q)) + q for q in queries]).cuda()
    mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in queries]).cuda()
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=len(scheme.common) + 3,
                             do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             eos_token_id=[tokenizer.eos_token_id, eot], pad_token_id=tokenizer.eos_token_id)
    return [scheme.parse(row) for row in out[:, width:].tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids")
    parser.add_argument("out_dir")
    parser.add_argument("--mode", choices=["discrete-exact", "fractional"], required=True)
    parser.add_argument("--reward", default="paper")
    parser.add_argument("--reward-mix", type=float, default=0.0,
                        help="weight of the Brier score mixed into the log-score reward (core.reward brier_mix)")
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
    parser.add_argument("--forward-batch", type=int, default=0,
                        help="rows per forward call within a minibatch (0: the whole minibatch); same math")
    parser.add_argument("--grad-ckpt", choices=["unsloth", "off"], default="off",
                        help="off is 1.4x faster at batch 4 and needs ~9 GB (bench_batching.py)")
    parser.add_argument("--vf-coef", type=float, default=0.1)
    parser.add_argument("--scoring", choices=["single", "shared", "full"], default="single",
                        help="single: one pass, 11 number probabilities + stop check on a sampled number; "
                             "shared: prefix pass + 11 cached continuations; full: 11 full sequences "
                             "(the runs before 2026-10-01T20Z)")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model", default=MODEL, help="model loaded in 4-bit, e.g. unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint-every", type=int, default=32, help="resumable checkpoint interval (0: off)")
    parser.add_argument("--stop-on", default="nonfinite",
                        help=f"comma-separated stability flags that end the run ({', '.join(FLAGS)}; empty: none)")
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing OUT_DIR/checkpoint")
    parser.add_argument("--stop-after", type=int, default=0, help="testing: behave as if preempted after N steps")
    parser.add_argument("--frozen-answers", action="store_true", help="generate answers with the adapter disabled")
    args = parser.parse_args()
    if args.regularization == "baseline" and args.scoring != "single":
        parser.error("--regularization baseline needs --scoring single")
    if args.invalid_reward is None:
        args.invalid_reward = -30. if args.reward == "released" else -3.
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # util/ModelLoader.load_lora_model_tokenizer (is_unsloth=True), minus the value head.
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=1048,
                                                         dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing="unsloth" if args.grad_ckpt == "unsloth" else False,
                                             random_state=3407,
                                             use_rslora=False, loftq_config=None)
    # ModelLoader applies this (via the value-head wrapper) so LoRA weights are bf16, which
    # Unsloth's fused LoRA backward requires.
    trl.trainer.peft_module_casting_to_bf16(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eot = end_of_turn(tokenizer)
    confidence_token = tokenizer.convert_tokens_to_ids("ĠConfidence")
    # The continuation PPO trains on after " Confidence": ": k" then end of turn.
    candidates = [tokenizer.encode(f": {k}", add_special_tokens=False) + [eot] for k in range(11)]
    scheme = LevelScheme(tokenizer, eot)
    args.n_common = len(scheme.common) if args.scoring == "single" else 0
    if args.value_head and not scheme.single:
        parser.error("--value-head reads one position per level token; it supports one-token levels (Llama-3) only")
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
    monitor = StabilityMonitor()
    stop_on = {f for f in args.stop_on.split(",") if f}
    if stop_on - set(FLAGS):
        parser.error(f"unknown --stop-on flags: {sorted(stop_on - set(FLAGS))}")
    checkpoint_dir = os.path.join(args.out_dir, "checkpoint")
    paths = {name: os.path.join(args.out_dir, f"{name}.jsonl") for name in ("metrics", "stability", "stability_events")}
    step, start_epoch, start_index, saved_order = 0, 0, 0, None
    last_checkpoint = [0]  # step of the latest checkpoint (promoted to -healthy if no flag follows it)
    state = None if args.no_resume else load_checkpoint(checkpoint_dir)
    if state is not None:
        load_trainable_state(state["weights"], model, *([value_head] if value_head else []))
        optimizer.load_state_dict(state["optimizer"])
        kl_controller.value = state["kl_coef"]
        rng.setstate(state["order_rng"])
        row_rng.setstate(state["row_rng"])
        set_rng_state(state["rng"])
        monitor.load_state_dict(state["monitor"])
        step, start_epoch, start_index, saved_order = state["step"], state["epoch"], state["index"], state["order"]
        last_checkpoint[0] = step
        for path in paths.values():
            truncate_jsonl(path, step)
        print(f"resumed from {checkpoint_dir} at step {step} (epoch {start_epoch}, question {start_index})", flush=True)
    logs = {name: open(path, "a" if state is not None else "w") for name, path in paths.items()}
    tracker = Tracker(args.out_dir, {k: v for k, v in vars(args).items() if k != "ids"})
    generations = GenerationLog(args.out_dir, "exact", vars(args), step if state is not None else None)

    def checkpoint(epoch, index, order):
        if not args.checkpoint_every and not stopping:
            return
        t_save = time.time()
        save_rotating_checkpoint(checkpoint_dir, dict(
            step=step, epoch=epoch, index=index, order=order, kl_coef=kl_controller.value,
            weights=trainable_state(model, *([value_head] if value_head else [])),
            optimizer=optimizer.state_dict(), order_rng=rng.getstate(), row_rng=row_rng.getstate(),
            rng=rng_state(), monitor=monitor.state_dict()), monitor.clean_since(last_checkpoint[0]))
        last_checkpoint[0] = step
        print(f"checkpoint at step {step} ({time.time() - t_save:.1f} s)", flush=True)

    stopping = []  # set by SIGTERM: finish the current step, checkpoint, exit

    def on_sigterm(signum, frame):
        stopping.append("SIGTERM")
    signal.signal(signal.SIGTERM, on_sigterm)

    for epoch in range(start_epoch, args.epochs):
        if saved_order is not None and epoch == start_epoch:
            order = saved_order  # the order rng already advanced past this epoch's shuffle
        else:
            order = list(range(len(data)))
            rng.shuffle(order)
        first = start_index if epoch == start_epoch else 0
        for start in range(first, len(order), args.batchsize):
            t0 = time.time()
            batch = collate([data[i] for i in order[start:start + args.batchsize]])
            FastLanguageModel.for_inference(model)
            with torch.no_grad(), (model.disable_adapter() if args.frozen_answers else contextlib.nullcontext()):
                out = model.generate(input_ids=batch["input_ids"].cuda(),
                                     attention_mask=batch["attention_mask"].cuda(), **generation)
            rows, entries, row_entry = [], [], []  # entries: one generation-log line per answer
            for i in range(len(out)):
                prompt = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
                answer = out[i][batch["input_ids"].shape[1]:].tolist()
                while answer and answer[-1] == tokenizer.eos_token_id:  # util.remove_padding
                    answer.pop()
                entries.append(dict(question_id=batch["question_id"][i], correct=None, confidence=None,
                                    answer=tokenizer.decode(answer, skip_special_tokens=True)))
                if not answer or answer[-1] != confidence_token:
                    entries[-1]["format"] = "no_confidence"
                    continue  # answer never reached " Confidence"; PPO would score it invalid
                text = tokenizer.decode(answer, skip_special_tokens=True) + ": 0"
                prediction, _ = parse_answer_confidence(text, False)
                if prediction is None:
                    entries[-1]["format"] = "unparsed"
                    continue
                correct = is_answer_correct(prediction, batch["gt_candidates"][i], Metric.F1, 0.5)
                entries[-1].update(answer=prediction, correct=bool(correct))
                rows.append((prompt + answer, float(correct), None, None))
                row_entry.append(len(entries) - 1)
            if args.scoring == "single" and rows:
                picks = sample_numbers(model, tokenizer, [r[0] for r in rows], scheme, eot)
                rows = [(q, label, k, None) for (q, label, _, _), k in zip(rows, picks)]
                for j, k in zip(row_entry, picks):
                    entries[j]["confidence"] = k
            FastLanguageModel.for_training(model)
            if args.regularization == "baseline":  # reference scores are fixed for the whole step
                refs = reference_scores(model, rows, scheme)
                rows = [(q, label, k, ref) for (q, label, k, _), ref in zip(rows, refs)]
            stats = dict(loss=0., confidence=0., mass=0., hinge_active=0., stop_prob=0., sampled_number_rate=0.,
                         kl=0., kl_coef=kl_controller.value, value_loss=0.)
            stop_checks, stats_rows, entropies, max_probs, grad_norms = 0, [], [], [], []
            entry_of_row = {id(r): j for r, j in zip(rows, row_entry)}
            minibatch = args.minibatch or args.batchsize
            updates = 0
            for epoch_pass in range(args.passes):
                order_rows = list(range(len(rows)))
                row_rng.shuffle(order_rows)
                for mb_start in range(0, len(order_rows), minibatch):
                    mb = [rows[i] for i in order_rows[mb_start:mb_start + minibatch]]
                    optimizer.zero_grad()
                    chunk = args.forward_batch or len(mb)
                    for c_start in range(0, len(mb), chunk):
                        rows_in = mb[c_start:c_start + chunk]
                        scored = score_rows(model, rows_in, candidates, eot, args, scheme)
                        hidden = store.get("hidden")
                        total = 0.
                        for b, (row, (logps, stop_logp)) in enumerate(zip(rows_in, scored)):
                            loss, confidence, mass, stop_prob, kl, value_loss = row_objective(
                                row, logps, stop_logp, args, kl_controller.value, value_head,
                                hidden[b] if value_head is not None else None)
                            total = total + loss / len(mb)
                            if epoch_pass == 0:
                                stats_rows.append((row, loss.item(), confidence.item(), mass, stop_prob, kl, value_loss))
                                pi = logps.detach().double().softmax(-1)
                                entropies.append(float(-(pi * pi.clamp_min(1e-30).log()).sum()))
                                max_probs.append(float(pi.max()))
                                entries[entry_of_row[id(row)]].update(
                                    pi=[round(float(x), 4) for x in pi], mass=round(mass, 5), loss=round(loss.item(), 5),
                                    stop_prob=None if stop_prob is None else round(stop_prob, 5))
                        total.backward()
                    grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, float("inf"))))
                    optimizer.step()
                    updates += 1
            for row, loss, confidence, mass, stop_prob, kl, value_loss in stats_rows:
                stats["loss"] += loss / len(rows)
                stats["confidence"] += confidence / len(rows)
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
            stats["stop_prob"] = stats["stop_prob"] / stop_checks if stop_checks else None
            if args.regularization == "baseline":  # TRL updates after each batch, n_steps = batch size
                kl_controller.update(stats["kl"], args.batchsize)
            step += 1
            record = dict(step=step, epoch=epoch, batch_hash=zlib.crc32(json.dumps(order[start:start + args.batchsize]).encode()),
                          rows=len(rows), skipped=len(out) - len(rows),
                          answer_accuracy=sum(r[1] for r in rows) / max(1, len(rows)), updates=updates,
                          seconds=time.time() - t0, time=time.time(), **stats)
            if args.save_every and step % args.save_every == 0:
                t_save = time.time()
                model.save_pretrained(os.path.join(args.out_dir, f"snapshot-step{step:05d}"))
                tokenizer.save_pretrained(os.path.join(args.out_dir, f"snapshot-step{step:05d}"))
                record["save_seconds"] = time.time() - t_save
            generations.write(step, entries)
            logs["metrics"].write(json.dumps(record) + "\n")
            logs["metrics"].flush()
            print(json.dumps(record), flush=True)
            # Stability: sampled confidence levels (None: answer or confidence malformed), correctness.
            mean = lambda xs: sum(xs) / len(xs) if xs else None
            stability, events = monitor.update(
                step, [r[2] for r in rows] + [None] * (len(out) - len(rows)), [r[1] for r in rows],
                loss=float(stats["loss"]), grad_norm=max(grad_norms) if grad_norms else None,
                pi_entropy=mean(entropies), pi_max=mean(max_probs))
            logs["stability"].write(json.dumps(stability) + "\n")
            logs["stability"].flush()
            tracker.log({**{k: v for k, v in record.items() if k not in ("time", "batch_hash")},
                         **{f"stability/{k}": v for k, v in stability.items() if k != "step"}}, step=step)
            for event in events:
                logs["stability_events"].write(json.dumps(event) + "\n")
                logs["stability_events"].flush()
                print("STABILITY", json.dumps(event), flush=True)
                tracker.alert(f"{os.path.basename(args.out_dir)}: {event['flag']}", json.dumps(event))
                if event["flag"] in stop_on:
                    stopping.append(event["flag"])
            if args.stop_after and step == args.stop_after:
                stopping.append("stop-after")
            done_with_epoch = start + args.batchsize >= len(order)
            position = (epoch + 1, 0, None) if done_with_epoch else (epoch, start + args.batchsize, order)
            if stopping or (args.checkpoint_every and step % args.checkpoint_every == 0):
                checkpoint(*position)
            if stopping:
                print(f"stopping at step {step}: {', '.join(stopping)}", flush=True)
                preempted = stopping[0] in ("SIGTERM", "stop-after")
                write_status(args.out_dir, "preempted" if preempted else "diverged", step=step, reason=stopping,
                             flags=sorted(monitor.raised))
                tracker.finish()
                sys.exit(143 if preempted else 3)
            if args.max_steps and step >= args.max_steps:
                break
        # Train.py saves model_finetuned after every epoch; save the adapter per epoch.
        model.save_pretrained(os.path.join(args.out_dir, f"model_finetuned_epoch{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, f"model_finetuned_epoch{epoch + 1}"))
        if args.max_steps and step >= args.max_steps:
            break
    write_status(args.out_dir, "completed", step=step, flags=sorted(monitor.raised))
    tracker.finish()


if __name__ == "__main__":
    main()
