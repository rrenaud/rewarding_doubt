"""Find the fastest equivalent batching for exact_llama.py's training step, and check equivalence.

On real prompts with on-policy answers, sampled confidences and LoRA perturbed away from zero:
  * equivalence: one minibatch's loss and LoRA+value-head gradient with 1 row per forward call vs
    2 and 4 rows per call (cosine, relative error);
  * timing of the update part of one step (4 passes x 2 minibatches of 4 = 8 Adam updates) for
    rows-per-call in {1, 2, 4} x gradient checkpointing {on, off}, with peak memory;
  * timing of the reference pass (per row vs one batch) and of answer + confidence sampling.
    python bench_batching.py IDS_JSON
"""
import argparse
import json
import sys
import time

import torch
from unsloth import FastLanguageModel
import trl

from exact_llama import (MODEL, ValueHead, capture_final_hidden, reference_scores, row_objective,
                         sample_numbers, score_rows)
from shared_prefix import split_candidates
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries


def main():
    torch.manual_seed(0)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=1048, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                             use_gradient_checkpointing="unsloth", random_state=3407)
    trl.trainer.peft_module_casting_to_bf16(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    confidence = tokenizer.convert_tokens_to_ids("ĠConfidence")
    candidates = [tokenizer.encode(f": {k}", add_special_tokens=False) + [eot] for k in range(11)]
    args = argparse.Namespace(scoring="single", regularization="baseline", mode="discrete-exact", reward="released",
                              invalid_reward=-30., format_weight=1., format_threshold=0.95, vf_coef=0.1,
                              n_common=len(split_candidates(candidates)[0]), forward_batch=0)
    ids = json.load(open(sys.argv[1]))
    data = subset_loader({"train": ids["train"][:8]})("triviaqa", "train", "verbalize", tokenizer)
    batch = DataCollatorForTokenizedQueries(tokenizer)([data[i] for i in range(len(data))])
    report = {}
    FastLanguageModel.for_inference(model)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                             max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9,
                             eos_token_id=[tokenizer.eos_token_id, eot, confidence], pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); report["answer_sampling_s"] = time.time() - t0
    queries = []
    for i in range(len(out)):
        prompt = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
        answer = out[i][batch["input_ids"].shape[1]:].tolist()
        while answer and answer[-1] == tokenizer.eos_token_id:
            answer.pop()
        if answer and answer[-1] == confidence:
            queries.append(prompt + answer)
    t0 = time.time()
    picks = sample_numbers(model, tokenizer, queries, candidates, eot)
    torch.cuda.synchronize(); report["confidence_sampling_s"] = time.time() - t0
    rows = [(q, float(i % 2), k, None) for i, (q, k) in enumerate(zip(queries, picks))]
    FastLanguageModel.for_training(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.normal_(0, 0.01)
    # Reference pass: per row vs one batch.
    torch.cuda.synchronize(); t0 = time.time()
    for r in rows:
        reference_scores(model, [r], candidates)
    torch.cuda.synchronize(); report["reference_per_row_s"] = time.time() - t0
    t0 = time.time()
    refs = reference_scores(model, rows, candidates)
    torch.cuda.synchronize(); report["reference_batched_s"] = time.time() - t0
    rows = [(q, label, k, ref) for (q, label, k, _), ref in zip(rows, refs)]
    report["rows"] = len(rows)

    value_head, store = ValueHead(model.config.hidden_size).cuda(), {}
    value_head.eval()  # no dropout, so gradients are comparable across settings
    capture_final_hidden(model, store)
    params = [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters())

    def minibatch_grad(mb, chunk):
        for p in params:
            p.grad = None
        losses = []
        for c in range(0, len(mb), chunk):
            rows_in = mb[c:c + chunk]
            scored = score_rows(model, rows_in, candidates, eot, args)
            total = 0.
            for b, (row, (logps, stop_logp)) in enumerate(zip(rows_in, scored)):
                loss = row_objective(row, logps, stop_logp, args, 0.05, value_head, store["hidden"][b])[0]
                total = total + loss / len(mb)
                losses.append(loss.item())
            total.backward()
        return torch.tensor(losses), torch.cat([p.grad.float().flatten() for p in params])

    mb = rows[:4]
    l1, g1 = minibatch_grad(mb, 1)
    l1b, g1b = minibatch_grad(mb, 1)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    report["equivalence"] = {"repeat_1": dict(cos=cos(g1b, g1), rel=rel(g1b, g1), max_loss_diff=float((l1b - l1).abs().max()))}
    for chunk in (2, 4):
        lc, gc = minibatch_grad(mb, chunk)
        report["equivalence"][f"rows_per_call_{chunk}"] = dict(cos=cos(gc, g1), rel=rel(gc, g1),
                                                               max_loss_diff=float((lc - l1).abs().max()))
    print(json.dumps(report), flush=True)

    inner = model.base_model.model.model
    optimizer = torch.optim.Adam(params, lr=1e-5)
    timings = {}
    for ckpt in (True, False):
        inner.gradient_checkpointing = ckpt
        for chunk in (1, 2, 4):
            def step():
                for _ in range(4):  # passes
                    for start in (0, 4):  # two minibatches of 4
                        optimizer.zero_grad()
                        minibatch = rows[start:start + 4]
                        for c in range(0, len(minibatch), chunk):
                            rows_in = minibatch[c:c + chunk]
                            scored = score_rows(model, rows_in, candidates, eot, args)
                            total = 0.
                            for b, (row, (logps, stop_logp)) in enumerate(zip(rows_in, scored)):
                                total = total + row_objective(row, logps, stop_logp, args, 0.05, value_head,
                                                              store["hidden"][b])[0] / len(minibatch)
                            total.backward()
                        optimizer.step()
            try:
                step()  # warm-up
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                for _ in range(3):
                    step()
                torch.cuda.synchronize()
                timings[f"ckpt_{'on' if ckpt else 'off'}_rows_per_call_{chunk}"] = dict(
                    update_part_s=(time.time() - t0) / 3, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            except torch.cuda.OutOfMemoryError:
                timings[f"ckpt_{'on' if ckpt else 'off'}_rows_per_call_{chunk}"] = "OOM"
                torch.cuda.empty_cache()
            print(json.dumps(timings), flush=True)
    report["update_part_timings"] = timings
    report["gpu"] = torch.cuda.get_device_name()
    json.dump(report, open("/tmp/bench_batching.json", "w"), indent=1)


if __name__ == "__main__":
    main()
