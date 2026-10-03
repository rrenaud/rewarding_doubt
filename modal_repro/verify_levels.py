"""Check LevelScheme against teacher-forced scoring of every level, for Llama-3 and Qwen-2.5.

    python verify_levels.py MODEL [MODEL ...]

For each query (released prompt + a dev question + "Answer: X, Confidence"), compares:
  levels[k] (k != 1)  vs  log P(common + tokens(k)) teacher-forced
  levels[1]           vs  log P(common + "1") + log(1 - P("0" | ..."1"))   (Qwen; Llama: plain)
  stop(k*)            vs  log P(<eot> | common + tokens(k*)) teacher-forced
with the released system prompt, so the model puts its mass on the levels. It reports the largest
difference in probability over all levels and in log-probability over levels with p > 1e-3 (far
less likely levels only show bf16 rounding of tiny logits), and the total level mass.
"""
import json
import sys

import torch
from unsloth import FastLanguageModel  # must precede transformers imports

from shared_prefix import LevelScheme, end_of_turn
from util.Prompts import get_prompt


def teacher_forced(model, prefix, tokens):
    ids = torch.tensor([prefix + tokens], device="cuda")
    logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
    return sum(logp[len(prefix) - 1 + i, t] for i, t in enumerate(tokens)), logp


def check(name, one_at_a_time):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=1048, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_training(model)
    eot = end_of_turn(tokenizer)
    scheme = LevelScheme(tokenizer, eot)
    questions = ["Which planet is known as the Red Planet?", "Who wrote the novel Middlemarch?",
                 "What is the capital of Australia?", "In which year did the Berlin Wall fall?"]
    answers = ["Mars", "George Eliot", "Sydney", "1989"]
    queries = []
    for q, a in zip(questions, answers):
        prompt = tokenizer.apply_chat_template([{"role": "system", "content": get_prompt("open")}, {"role": "user", "content": q}],
                                               tokenize=True, add_generation_prompt=True)
        queries.append(prompt + tokenizer.encode(f"Answer: {a}, Confidence", add_special_tokens=False))
    worst, worst_p, masses = 0., 0., []
    with torch.no_grad():
        for k_star in (1, 10, 7, None):
            scored = ([scheme.batch(model, [q], [k_star])[0] for q in queries] if one_at_a_time
                      else scheme.batch(model, queries, [k_star] * len(queries)))
            for (levels, stop), q in zip(scored, queries):
                for k in range(11):
                    ref, logp = teacher_forced(model, q, scheme.common + scheme.tokens(k))
                    if k == 1 and not scheme.single:
                        _, after = teacher_forced(model, q, scheme.common + scheme.tokens(1))
                        n = len(q) + len(scheme.common)
                        ref = ref + torch.log1p(-after[n, scheme.zero].exp())
                    worst_p = max(worst_p, abs(float(levels[k].exp() - ref.exp())))
                    if float(ref.exp()) > 1e-3:
                        worst = max(worst, abs(float(levels[k] - ref)))
                if k_star is not None:
                    toks = scheme.common + scheme.tokens(k_star)
                    _, logp = teacher_forced(model, q, toks + [eot])
                    worst = max(worst, abs(float(stop - logp[len(q) + len(toks) - 1, eot])))
                masses.append(float(levels.logsumexp(-1).exp()))
    print(json.dumps(dict(model=name, batch="one query" if one_at_a_time else "4 queries", one_token_levels=scheme.single, common=tokenizer.decode(scheme.common),
                          max_abs_prob_diff=worst_p, max_abs_logp_diff_likely_levels=worst, level_mass=[round(m, 4) for m in masses[:4]])), flush=True)
    return max(worst, worst_p)


if __name__ == "__main__":
    results = [check(name, one) for name in sys.argv[1:] for one in (True, False)]
    sys.exit(0 if max(results) < 5e-2 else 1)
