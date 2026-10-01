"""Check shared-prefix scoring against 11 full sequences on the real Llama setup.

For a few real training prompts with on-policy answers, and LoRA weights perturbed away from
zero so gradients are non-trivial, compare:
  * the 11 log-probabilities: shared-prefix vs batched full sequences, against the bf16 noise
    floor (batched full sequences vs the same sequences one at a time);
  * the LoRA gradient of both objectives (cosine similarity, relative L2 error), against the same
    noise floor;
  * forward+backward time per question.
    python verify_shared_prefix.py IDS_JSON
"""
import json
import sys
import time

import torch
from unsloth import FastLanguageModel
import trl

from rewarding_doubt.core import objective
from shared_prefix import candidate_logps, full_sequence_logps
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


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
    ids = json.load(open(sys.argv[1]))
    data = subset_loader({"train": ids["train"][:6]})("triviaqa", "train", "verbalize", tokenizer)
    batch = DataCollatorForTokenizedQueries(tokenizer)([data[i] for i in range(len(data))])
    FastLanguageModel.for_inference(model)
    with torch.no_grad():
        out = model.generate(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                             max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9,
                             eos_token_id=[tokenizer.eos_token_id, eot, confidence], pad_token_id=tokenizer.eos_token_id)
    queries = []
    for i in range(len(out)):
        prompt = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
        answer = out[i][batch["input_ids"].shape[1]:].tolist()
        while answer and answer[-1] == tokenizer.eos_token_id:
            answer.pop()
        if answer and answer[-1] == confidence:
            queries.append(prompt + answer)
    FastLanguageModel.for_training(model)
    lora = [p for n, p in model.named_parameters() if p.requires_grad]
    with torch.no_grad():  # move LoRA away from its zero-initialized B so gradients are non-trivial
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.normal_(0, 0.01)

    def grads(fn, query, mode, label):
        for p in lora:
            p.grad = None
        logps = fn(query)
        loss, _ = objective(logps[None].double(), torch.tensor([label], dtype=torch.float64, device=logps.device), mode)
        loss.backward()
        return logps.detach(), torch.cat([p.grad.float().flatten() for p in lora])

    full = lambda q: full_sequence_logps(model, q, candidates, eot)
    one_by_one = lambda q: torch.cat([full_sequence_logps(model, q, [c], eot) for c in candidates])
    shared = lambda q: candidate_logps(model, q, candidates)
    report = dict(queries=len(queries), query_tokens=[len(q) for q in queries], cases=[])
    for qi, query in enumerate(queries):
        for mode in ["discrete-exact", "fractional"]:
            label = float(qi % 2)
            lf, gf = grads(full, query, mode, label)
            lo, go = grads(one_by_one, query, mode, label)
            ls, gs = grads(shared, query, mode, label)
            rel = lambda a, b: ((a - b).norm() / b.norm()).item()
            cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            report["cases"].append(dict(
                query=qi, mode=mode, label=label,
                logp_full=[round(v, 4) for v in lf.tolist()],
                max_logp_diff_shared_vs_full=(ls - lf).abs().max().item(),
                max_logp_diff_floor=(lo - lf).abs().max().item(),
                grad_rel_err_shared_vs_full=rel(gs, gf), grad_rel_err_floor=rel(go, gf),
                grad_cos_shared_vs_full=cos(gs, gf), grad_cos_floor=cos(go, gf)))
            print(json.dumps(report["cases"][-1]), flush=True)
    timings = {}
    for name, fn in [("full", full), ("shared", shared)]:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(3):
            for query in queries:
                grads(fn, query, "fractional", 1.0)
        torch.cuda.synchronize()
        timings[name] = (time.time() - start) / (3 * len(queries))
    report["seconds_per_question_fwd_bwd"] = timings
    print(json.dumps(dict(report, cases=None)), flush=True)
    json.dump(report, open("/tmp/verify_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
