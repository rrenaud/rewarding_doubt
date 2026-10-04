"""exact_fast level scoring in the patched util/ExactConfidence.py, against teacher forcing.

One prompt at a time (no batching effects), compares each fast level with its definition computed
from plain single-sequence forward passes:
  one-token levels (Llama):  p(k) = P(": k")
  Qwen ("10" = "1" "0"):     p(k) = P(": k") for k != 1, 10;  p(1) = P(": 1") (1 - P("0" | ": 1"));
                             p(10) = P(": 1") P("0" | ": 1") = P(": 10")
and reports the largest absolute difference in probability.
"""
import sys
import torch
from unsloth import FastLanguageModel
from util.ExactConfidence import _shared_level_logps, end_of_turn
from util.Prompts import get_prompt


def seq_logp(model, prefix, tokens):
    logp = model(input_ids=torch.tensor([prefix + tokens], device="cuda")).logits[0].float().log_softmax(-1)
    return sum(logp[len(prefix) - 1 + i, t] for i, t in enumerate(tokens)), logp


for name in sys.argv[1:]:
    model, tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=1048, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_training(model)
    levels = [tok.encode(f": {k}", add_special_tokens=False) for k in range(11)]
    worst = 0.
    with torch.no_grad():
        for q, a in [("What is the capital of Australia?", "Sydney"), ("Who wrote Middlemarch?", "George Eliot"),
                     ("How many moons does Mars have?", "Three")]:
            prefix = tok.apply_chat_template([{"role": "system", "content": get_prompt("open")}, {"role": "user", "content": q}],
                                             tokenize=True, add_generation_prompt=True) + tok.encode(f"Answer: {a}, Confidence", add_special_tokens=False)
            fast = _shared_level_logps(model, [prefix], levels, tok.pad_token_id)[0].exp()
            ref = []
            for k in range(11):
                lp, _ = seq_logp(model, prefix, levels[k])
                ref.append(lp.exp())
            if len(levels[10]) != len(levels[1]):  # Qwen: split level 1 from the "1" prefix of "10"
                ref[1] = ref[1] - ref[10]
            diff = max(abs(float(f - r)) for f, r in zip(fast, ref))
            worst = max(worst, diff)
            print(name, a, "fast", [round(float(x), 3) for x in fast], "ref", [round(float(x), 3) for x in ref], flush=True)
    print("RESULT", name, "max |p_fast - p_teacher_forced| =", worst, flush=True)
