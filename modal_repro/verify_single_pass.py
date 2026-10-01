"""Check single-pass scoring (11 numbers from one softmax + a stop check) on the real Llama setup.

Settings: the base model, a trained discrete-exact adapter, and LoRA perturbed away from zero.
  * equivalence: single-pass number log-probs and their gradient vs 11 full sequences ": k"
    (no stop token), against the bf16 noise floor (batched vs one at a time);
  * the assumption: P(<eot> | k) from full sequences, how much ignoring it moves pi (KL), and
    whether the single-pass stop check reproduces P(<eot> | k*) at the sampled-style k*;
  * forward+backward time per question for single, shared and full scoring.
    python verify_single_pass.py IDS_JSON TRAINED_ADAPTER_DIR
"""
import json
import sys
import time

import torch
from unsloth import FastLanguageModel
import trl

from rewarding_doubt.core import objective
from shared_prefix import candidate_logps, full_sequence_logps, single_pass_logps, split_candidates
from subset import subset_loader
from util.DataHelper import DataCollatorForTokenizedQueries

MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def load(name, adapter):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=1048, dtype=None, load_in_4bit=True)
    if not adapter:
        model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
                                                 use_gradient_checkpointing="unsloth", random_state=3407)
    trl.trainer.peft_module_casting_to_bf16(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def queries_for(model, tokenizer, ids, eot, confidence):
    data = subset_loader({"train": ids})("triviaqa", "train", "verbalize", tokenizer)
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
    return queries


def assumption_stats(model, queries, candidates, eot):
    """How far is P(<eot> | k) from 1, and does single-pass reproduce it?"""
    common, numbers, stop = split_candidates(candidates)
    no_stop = [c[:-1] for c in candidates]
    rows = []
    with torch.no_grad():
        for q in queries:
            with_stop = full_sequence_logps(model, q, candidates, eot)
            without = full_sequence_logps(model, q, no_stop, eot)
            p_stop = (with_stop - without).exp()
            pi_full, pi_single_ref = with_stop.softmax(-1), without.softmax(-1)
            single, _ = single_pass_logps(model, q, common, numbers, stop)
            pi_single = single.softmax(-1)
            k = int(pi_single.argmax())
            _, stop_logp = single_pass_logps(model, q, common, numbers, stop, k_star=k)
            rows.append(dict(
                pi_weighted_stop=float((pi_full * p_stop).sum()),
                min_stop_where_pi_gt_1pct=float(p_stop[pi_full > 0.01].min()),
                kl_full_vs_single=float((pi_full * (pi_full.log() - pi_single.log())).sum()),
                max_abs_dpi=float((pi_full - pi_single).abs().max()),
                max_logp_diff_single_vs_full_numbers=float((single - without).abs().max()),
                max_dpi_single_vs_full_numbers=float((pi_single - pi_single_ref).abs().max()),
                stop_single=float(stop_logp.exp()), stop_full=float(p_stop[k]), k_star=k))
    keys = rows[0].keys()
    summary = {k: dict(mean=sum(r[k] for r in rows) / len(rows), worst=(min if "stop" in k and "kl" not in k else max)(r[k] for r in rows))
               for k in keys if k != "k_star"}
    return dict(per_query=rows, summary=summary)


def main():
    torch.manual_seed(0)
    ids = json.load(open(sys.argv[1]))["train"][:12]
    report = {}
    # 1) Trained adapter: does formatting stay easy after training? (forward only)
    model, tokenizer = load(sys.argv[2], adapter=True)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    confidence = tokenizer.convert_tokens_to_ids("ĠConfidence")
    candidates = [tokenizer.encode(f": {k}", add_special_tokens=False) + [eot] for k in range(11)]
    report["trained_adapter"] = assumption_stats(model, queries_for(model, tokenizer, ids, eot, confidence), candidates, eot)
    print("trained", json.dumps(report["trained_adapter"]["summary"]), flush=True)
    del model
    torch.cuda.empty_cache()
    # 2) Base model, then 3) perturbed LoRA for equivalence and gradients.
    model, tokenizer = load(MODEL, adapter=False)
    queries = queries_for(model, tokenizer, ids, eot, confidence)
    report["base"] = assumption_stats(model, queries, candidates, eot)
    print("base", json.dumps(report["base"]["summary"]), flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.normal_(0, 0.01)
    report["perturbed"] = assumption_stats(model, queries, candidates, eot)
    lora = [p for p in model.parameters() if p.requires_grad]
    common, numbers, stop = split_candidates(candidates)
    no_stop = [c[:-1] for c in candidates]

    def grad(fn, q, mode, label):
        for p in lora:
            p.grad = None
        logps = fn(q)
        loss, _ = objective(logps[None].double(), torch.tensor([label], dtype=torch.float64, device=logps.device), mode)
        loss.backward()
        return torch.cat([p.grad.float().flatten() for p in lora])

    single = lambda q: single_pass_logps(model, q, common, numbers, stop)[0]
    full = lambda q: full_sequence_logps(model, q, no_stop, eot)
    one_by_one = lambda q: torch.cat([full_sequence_logps(model, q, [c], eot) for c in no_stop])
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    grads = []
    for i, q in enumerate(queries[:6]):
        for mode in ["discrete-exact", "fractional"]:
            gs, gf, go = grad(single, q, mode, float(i % 2)), grad(full, q, mode, float(i % 2)), grad(one_by_one, q, mode, float(i % 2))
            grads.append(dict(query=i, mode=mode, cos_single=cos(gs, gf), cos_floor=cos(go, gf),
                              relerr_single=rel(gs, gf), relerr_floor=rel(go, gf)))
    report["gradients"] = grads
    timings = {}
    scorers = {"single": lambda q: single_pass_logps(model, q, common, numbers, stop, k_star=10)[0],
               "shared": lambda q: candidate_logps(model, q, candidates),
               "full": lambda q: full_sequence_logps(model, q, candidates, eot)}
    for name, fn in scorers.items():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(3):
            for q in queries[:6]:
                grad(fn, q, "fractional", 1.0)
        torch.cuda.synchronize()
        timings[name] = (time.time() - start) / 18
    report["seconds_per_question_fwd_bwd"] = timings
    report["queries"] = len(queries)
    print(json.dumps(dict(gradients=grads, timings=timings)), flush=True)
    json.dump(report, open("/tmp/verify_single_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
