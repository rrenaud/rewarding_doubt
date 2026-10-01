"""Tinker datums: shifted targets, fixed answer prefixes, confidence-only loss."""
import tinker
import torch

from .core import objective

SYSTEM = (
    "Answer the question and report your confidence as an integer from 0 to 10. "
    "0 means certain the answer is wrong; 10 means certain it is correct. "
    "Use exactly this format: Answer: <answer>, Confidence: <confidence>"
)


def prompt_tokens(tokenizer, question, answer=None, system=SYSTEM):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
    if answer is not None:
        tokens += tokenizer.encode(f"Answer: {answer}, Confidence: ", add_special_tokens=False)
    return tokens


def confidence_sequences(tokenizer):
    """Score complete strings, including EOS so '1' and '10' are disjoint events."""
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer needs an end-of-turn/EOS token")
    sequences = [tokenizer.encode(str(i), add_special_tokens=False) + [tokenizer.eos_token_id]
                 for i in range(11)]
    if len(set(map(tuple, sequences))) != 11:
        raise ValueError("Confidence sequences must be distinct")
    for i, a in enumerate(sequences):
        if any(b[:len(a)] == a for j, b in enumerate(sequences) if j != i):
            raise ValueError("Confidence sequences must not be prefixes of one another")
    return sequences


def tensor(values, dtype):
    return tinker.TensorData(data=values, dtype=dtype, shape=[len(values)])


def datum(prefix, completion, *, old_logps=None, advantages=None):
    """advantages: one value per completion token (RL losses), or None for cross-entropy."""
    if not prefix or not completion:
        raise ValueError("Both prefix and completion must be nonempty")
    full = prefix + completion
    inputs = {"target_tokens": tensor(full[1:], "int64")}
    n = len(prefix) - 1
    if old_logps is None:
        inputs["weights"] = tensor([0.] * n + [1.] * len(completion), "float32")
    else:
        if advantages is None or not len(old_logps) == len(advantages) == len(completion):
            raise ValueError("Rollout logprobs and advantages are required for all completion tokens")
        inputs["logprobs"] = tensor([0.] * n + list(old_logps), "float32")
        inputs["advantages"] = tensor([0.] * n + list(advantages), "float32")
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs=inputs)


def candidate_datums(prefixes, sequences):
    return [datum(prefix, sequence) for prefix in prefixes for sequence in sequences]


def confidence_loss(labels, mode, variant, lengths, format_weight=0.0, format_threshold=0.95):
    """Only confidence suffix logprobs contribute; autograd crosses candidate branches."""
    def loss_fn(data, logprobs):
        if len(logprobs) != 11 * len(labels):
            raise ValueError("Expected exactly 11 candidate branches per example")
        matrix = torch.stack([lp[-lengths[i % 11]:].sum()
                              for i, lp in enumerate(logprobs)]).reshape(-1, 11)
        loss, confidence = objective(matrix, matrix.new_tensor(labels), mode, variant,
                                     format_weight, format_threshold)
        mass = matrix.detach().logsumexp(-1).exp()
        return loss, {"confidence_mean": confidence.detach().mean().item(),
                      "valid_confidence_mass_mean": mass.mean().item(),
                      "format_hinge_active": (mass < format_threshold).double().mean().item()}
    return loss_fn


def score_candidates(sampler, prefixes, sequences):
    # Teacher-force ALL 11 candidates, including those missing from top-k.
    futures = [(sampler.compute_logprobs(tinker.ModelInput.from_ints(prefix + sequence)), len(sequence))
               for prefix in prefixes for sequence in sequences]
    values = []
    for future, length in futures:
        suffix = future.result()[-length:]
        if any(v is None for v in suffix):
            raise ValueError("Tinker omitted a requested candidate log probability")
        values.append(sum(suffix))
    return torch.tensor(values, dtype=torch.float64).reshape(-1, 11)
