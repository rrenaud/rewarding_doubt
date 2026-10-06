"""Check rewarding_doubt.attn_bias inside Unsloth's training forward, its decoding loop and PEFT's
disable_adapter, on Qwen-2.5-3B with a LoRA adapter as the released code builds it.

    python attn_bias_check.py

1. Zero biases leave the logits unchanged; random biases change them; disable_adapter() gives the
   base logits again (the reference model).
2. Greedy decoding agrees with a full forward pass over the decoded tokens: at every decoded
   position the decoded token is the forward's argmax, or within 1 logit of it: decoding and
   training kernels differ slightly, bf16 resolves logits of this size to 0.5, and large random
   biases produce gibberish full of near-ties. Checked with zero biases and
   random biases of scale 0.1 and 1. With the decoding-loop patch removed, they disagree, so the
   check can see a bias that decoding misses.
3. save / load round-trips the values.
"""
import json
import tempfile

import torch
from unsloth import FastLanguageModel

from rewarding_doubt import attn_bias


def main():
    model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-3B-Instruct", max_seq_length=1048, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8, lora_dropout=0, bias="none", random_state=3407)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Which country is Mount Kilimanjaro in? Answer briefly."}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    report = {}

    def logits(x):
        FastLanguageModel.for_training(model)
        with torch.no_grad():
            return model(input_ids=x).logits.float()

    base = logits(ids)
    biases = attn_bias.install(model, range(18, 36))
    report["zero_bias_max_diff"] = float((logits(ids) - base).abs().max())
    torch.manual_seed(0)
    with torch.no_grad():
        for b in biases.values():
            b.normal_(0, 1.0)
    biased = logits(ids)
    report["random_bias_max_diff"] = float((biased - base).abs().max())
    with torch.no_grad(), model.disable_adapter():
        report["disable_adapter_vs_base_max_diff"] = float((logits(ids) - base).abs().max())

    def greedy_agreement(n=24):
        """Decoded tokens that are also the full forward's argmax, and the largest logit gap where not."""
        FastLanguageModel.for_inference(model)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        full = logits(out)[0, ids.shape[1] - 1:-1]
        decoded = out[0, ids.shape[1]:]
        full = full[:len(decoded)]
        gaps = (full.max(-1).values - full.gather(-1, decoded[:, None])[:, 0])
        agree = int((gaps == 0).sum())
        return dict(agree=f"{agree}/{len(decoded)}", max_gap=round(float(gaps.max()), 3), text=tok.decode(decoded)), agree == len(decoded) or float(gaps.max()) <= 1.0

    def set_biases(scale):
        torch.manual_seed(0)
        with torch.no_grad():
            for b in biases.values():
                b.normal_(0, scale) if scale else b.zero_()

    import unsloth.models.llama as unsloth_llama
    patched = unsloth_llama.LlamaAttention_fast_forward_inference
    decode_ok = []
    for scale in (0.0, 0.1, 1.0):
        set_biases(scale)
        report[f"decode_vs_forward_scale{scale}"], ok = greedy_agreement()
        decode_ok.append(ok)
    unsloth_llama.LlamaAttention_fast_forward_inference = attn_bias._patched[0]  # Unsloth's original
    report["unpatched_decode_vs_forward_scale1.0"], unpatched_ok = greedy_agreement()
    unsloth_llama.LlamaAttention_fast_forward_inference = patched

    with tempfile.TemporaryDirectory() as d:
        attn_bias.save(model, d)
        saved = torch.load(f"{d}/{attn_bias.FILE}")
    report["save_roundtrip_ok"] = all(torch.equal(saved[i], b.detach().cpu()) for i, b in biases.items())
    print(json.dumps(report, indent=1), flush=True)
    ok = (report["zero_bias_max_diff"] == 0 and report["random_bias_max_diff"] > 1 and report["disable_adapter_vs_base_max_diff"] == 0
          and all(decode_ok) and not unpatched_ok and report["save_roundtrip_ok"])
    print("ATTN_BIAS_CHECK", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
