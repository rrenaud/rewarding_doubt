"""Evaluate an adapter trained with --frozen-answers: base-model answer, then adapted confidence.

    python frozen_eval.py IDS_JSON MODEL_DIR OUT_JSON      # from SingleAnswerSetting/

Same settings as the released evaluation (InferenceDatasetSplit + util.EvaluationMetrics: the
16-bit base model under the adapter, as ModelLoader loads it) except
that generation is split where training split it:
1. answer: the adapter disabled, the released prompt, T=0.6, top-p 0.9, up to 256 tokens, stopping
   at " Confidence" (Train.py's prediction terminators);
2. confidence: the adapter enabled, continuing from prompt + answer at T=0.6, top-p 0.9.
The text is parsed with the released parse_answer_confidence and graded with F1 > 0.5;
metrics come from rewarding_doubt.paper_ppo.evaluation_metrics (torchmetrics ECE, sklearn AUROC
and Brier), plus unsampled columns from pi at the number position. Writes OUT_JSON (one row per
question) and OUT_JSON with _metrics.json (as subset.py evaluate does).
"""
import json
import os
import shutil
import sys

import torch
from unsloth import FastLanguageModel  # must precede transformers imports

from rewarding_doubt.paper_ppo import evaluation_metrics
from subset import subset_loader
from util.EvaluationMetrics import Metric, is_answer_correct
from util.ResponseHandling import parse_answer_confidence


def left_pad(rows, pad):
    width = max(map(len, rows))
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows]).cuda()
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows]).cuda()
    return ids, mask, width


def main(ids_path, model_dir, out_path, batch=32):
    if os.path.exists(os.path.join(model_dir, "config.json")):  # a PPO snapshot: TRL config + value head
        policy_dir = "/tmp/frozen_eval_adapter"
        shutil.rmtree(policy_dir, ignore_errors=True)
        shutil.copytree(model_dir, policy_dir, ignore=shutil.ignore_patterns("config.json", "pytorch_model.bin"))
        model_dir = policy_dir
    # As util.ModelLoader.load_model_tokenizer (the released evaluation): load_in_4bit=False, so Unsloth
    # puts the adapter on the 16-bit base model, although training used the 4-bit one.
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_dir, max_seq_length=1048, dtype=None,
                                                         load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    pad, eot = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")
    confidence_token = tokenizer.convert_tokens_to_ids("ĠConfidence")
    numbers = [tokenizer.encode(f": {k}", add_special_tokens=False)[-1] for k in range(11)]
    data = subset_loader(json.load(open(ids_path)))("triviaqa", "validation", "verbalize", tokenizer)
    rows = []
    for start in range(0, len(data), batch):
        part = [data[i] for i in range(start, min(start + batch, len(data)))]
        prompts = [d["query"] for d in part]
        ids, mask, width = left_pad(prompts, pad)
        with torch.no_grad(), model.disable_adapter():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=256, do_sample=True,
                                 temperature=0.6, top_p=0.9, eos_token_id=[pad, eot, confidence_token], pad_token_id=pad)
        answers = []
        for row in out[:, width:].tolist():
            while row and row[-1] == pad:
                row.pop()
            answers.append(row)
        ready = [i for i, a in enumerate(answers) if a and a[-1] == confidence_token]
        continuations, pis = {}, {}
        if ready:
            ids2, mask2, width2 = left_pad([prompts[i] + answers[i] for i in ready], pad)
            with torch.no_grad():
                out2 = model.generate(input_ids=ids2, attention_mask=mask2, max_new_tokens=8, do_sample=True,
                                      temperature=0.6, top_p=0.9, eos_token_id=[pad, eot], pad_token_id=pad,
                                      output_logits=True, return_dict_in_generate=True)
            logits = torch.stack(out2.logits, 1).float()
            for j, i in enumerate(ready):
                gen = out2.sequences[j, width2:].tolist()
                continuations[i] = gen
                # pi at the number: ": " then the number token, as in the training scorer.
                if len(gen) > 2:
                    p = logits[j, 2].softmax(-1)[torch.tensor(numbers, device=logits.device)]
                    pis[i] = (p / p.sum()).tolist()
        for i, d in enumerate(part):
            text = tokenizer.decode(answers[i] + continuations.get(i, []), skip_special_tokens=True)
            prediction, confidence = parse_answer_confidence(text, False)
            ok = prediction is not None and confidence is not None and 0 <= confidence <= 10
            correct = bool(ok and is_answer_correct(prediction, d["gt_candidates"], Metric.F1, 0.5))
            rows.append(dict(question=d["question"], question_id=d.get("question_id"), response=text,
                             prediction=prediction, confidence=confidence if ok else None, correct=correct,
                             pi=pis.get(i), gt_candidates=d["gt_candidates"]))
    metrics = evaluation_metrics([dict(confidence=r["confidence"], correct=r["correct"]) for r in rows])
    with_pi = [r for r in rows if r["confidence"] is not None and r["pi"] is not None]
    if with_pi:
        unsampled = evaluation_metrics([dict(confidence=sum(k * p for k, p in enumerate(r["pi"])), correct=r["correct"])
                                        for r in with_pi])
        metrics.update({f"{k}_unsampled": unsampled[k] for k in ("ece", "auroc", "brier")})
    metrics["protocol"] = "frozen-answers"
    json.dump(rows, open(out_path, "w"))
    json.dump(metrics, open(out_path.replace(".json", "_metrics.json"), "w"), indent=1)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
