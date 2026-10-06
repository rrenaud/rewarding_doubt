"""Trainable vectors added to attention outputs: each chosen layer's write to the residual stream
becomes attn(x) + b, with b zero-initialized, so an installed model starts identical to the base.

For the released pipeline (Unsloth + PEFT + TRL) the vector has to act everywhere the policy runs:

- training and prompt prefill go through each attention module's forward, which is wrapped;
- Unsloth's decoding loop calls LlamaAttention_fast_forward_inference(layer.self_attn, ...) by
  name instead of the module, so that function is patched in unsloth.models.llama;
- the reference model is the policy under PEFT's disable_adapter() (TRL and
  patches/exact_confidence.patch), which does not know about these vectors, so disable_adapter is
  patched to switch them off as well;
- save_pretrained writes only the LoRA adapter, so the vectors go to attn_biases.pt beside it.
"""
import contextlib
import os

import torch

FILE = "attn_biases.pt"
_enabled = [True]
_patched = []  # holds Unsloth's original decoding-loop attention once patched


def decoder_layers(model):
    """The decoder's ModuleList of layers inside a PEFT, value-head or plain Hugging Face model."""
    for name, module in model.named_modules():
        if name.endswith("model.layers") and isinstance(module, torch.nn.ModuleList):
            return module
    raise ValueError("no decoder layers found")


def _patch_once():
    if _patched:
        return
    import peft
    import unsloth.models.llama as unsloth_llama

    inference = unsloth_llama.LlamaAttention_fast_forward_inference

    def attention_inference(self, *args, **kwargs):
        out, present = inference(self, *args, **kwargs)
        bias = getattr(self, "residual_bias", None)
        if bias is not None and _enabled[0]:
            out = out + bias.to(out.dtype)
        return out, present
    unsloth_llama.LlamaAttention_fast_forward_inference = attention_inference

    disable_adapter = peft.PeftModel.disable_adapter

    @contextlib.contextmanager
    def disable_adapter_and_biases(self):
        with disable_adapter(self):
            previous, _enabled[0] = _enabled[0], False
            try:
                yield
            finally:
                _enabled[0] = previous
    peft.PeftModel.disable_adapter = disable_adapter_and_biases
    _patched.append(inference)


def install(model, layers):
    """Add a zero vector to the attention output of each layer index in `layers`; returns them by layer."""
    _patch_once()
    decoder = decoder_layers(model)
    device = next(model.parameters()).device
    hidden = next(m.config.hidden_size for m in model.modules() if hasattr(getattr(m, "config", None), "hidden_size"))
    biases = {}
    for i in layers:
        attn = decoder[i].self_attn
        attn.residual_bias = torch.nn.Parameter(torch.zeros(hidden, device=device))
        inner = attn.forward

        def forward(*args, inner=inner, attn=attn, **kwargs):
            out = inner(*args, **kwargs)
            if not _enabled[0]:
                return out
            if isinstance(out, tuple):
                return (out[0] + attn.residual_bias.to(out[0].dtype), *out[1:])
            return out + attn.residual_bias.to(out.dtype)
        attn.forward = forward
        biases[i] = attn.residual_bias
    return biases


def state(model):
    return {i: layer.self_attn.residual_bias.detach().cpu() for i, layer in enumerate(decoder_layers(model))
            if hasattr(layer.self_attn, "residual_bias")}


def save(model, directory):
    torch.save(state(model), os.path.join(directory, FILE))


def load(model, directory):
    """Install and fill the vectors saved in `directory`, if any; returns the number of layers."""
    path = os.path.join(directory, FILE)
    if not os.path.exists(path):
        return 0
    saved = torch.load(path)
    biases = install(model, sorted(saved))
    with torch.no_grad():
        for i, value in saved.items():
            biases[i].copy_(value)
    return len(saved)
