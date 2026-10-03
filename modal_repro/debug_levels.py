"""Per-level comparison for one query (debugging LevelScheme on Qwen)."""
import sys
import torch
from unsloth import FastLanguageModel
from shared_prefix import LevelScheme, end_of_turn
from util.Prompts import get_prompt

model, tok = FastLanguageModel.from_pretrained(model_name=sys.argv[1], max_seq_length=1048, dtype=None, load_in_4bit=True)
FastLanguageModel.for_training(model)
eot = end_of_turn(tok); s = LevelScheme(tok, eot)
q = tok.apply_chat_template([{"role": "system", "content": get_prompt("open")}, {"role": "user", "content": "Who wrote the novel Middlemarch?"}],
                            tokenize=True, add_generation_prompt=True) + tok.encode("Answer: George Eliot, Confidence", add_special_tokens=False)
n = len(q) + len(s.common)
def run(seq):
    return model(input_ids=torch.tensor([seq], device="cuda")).logits[0].float().log_softmax(-1)
with torch.no_grad():
    a = run(q + s.common + [s.one]); b = run(q + s.common + [s.digits[7]]); c = run(q + s.common)
    print("first-token logits same across endings:", float((a[n - 1] - b[n - 1]).abs().max()), float((a[n - 1] - c[n - 1]).abs().max()))
    levels, _ = s.batch(model, [q], [None])[0]
    for k in range(11):
        toks = s.common + s.tokens(k)
        lp = run(q + toks); ref = sum(lp[len(q) - 1 + i, t] for i, t in enumerate(toks))
        print(k, [tok.decode([t]) for t in s.tokens(k)], "scheme %.4f" % float(levels[k].exp()), "teacher %.4f" % float(ref.exp()))
    print("P(0|1) %.4f  P(eot|1) %.4f" % (float(a[n, s.zero].exp()), float(a[n, eot].exp())))
with torch.no_grad():
    alone = run(q + s.common + [s.one])
    pair = model(input_ids=torch.tensor([q + s.common + [s.one], q + s.common + [s.digits[7]]], device="cuda")).logits[0].float().log_softmax(-1)
    print("batching effect: max |logp diff| at the number position", float((alone[n - 1] - pair[n - 1]).abs().max()),
          "| max |prob diff|", float((alone[n - 1].exp() - pair[n - 1].exp()).abs().max()))
    levels_alone, _ = s.batch(model, [q], [None])[0]; levels_pair, _ = s.batch(model, [q], [7])[0]
    print("batching effect on the 11 levels: max |prob diff|", float((levels_alone.exp() - levels_pair.exp()).abs().max()))
