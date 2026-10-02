"""Run the released Train.py / InferenceDatasetSplit.py on a fixed subset of TriviaQA questions.

Runs inside the Modal container from SingleAnswerSetting/. The released training and inference
code is used unmodified; the only intervention is that `load_prepared_dataset`, which those
modules import by name, is replaced by a copy of the original that keeps only the requested
question IDs before normalization (which drops `question_id`). Each epoch's
`model_finetuned` checkpoint is also copied to `model_finetuned_epoch<k>`, because Train.py
overwrites it every epoch.

    python subset.py train IDS_JSON [--save-every N] [--seed S] [--no-kl] [--hinge W] -- <Train.py args>

Options before "--" change PPO from outside Train.py, which stays unmodified:
  --seed S   reseed torch/numpy/random after importing Train.py (which hard-codes manual_seed(2))
  --no-kl    PPOConfig(init_kl_coef=0, adap_kl_ctrl=False): no KL penalty toward the base model
  --hinge W  add discrete-exact's format hinges to PPO's policy loss, from the logits PPO already
             computes for its sampled response ": k<eot>":
             W * [relu(log 0.95 - log M) + relu(log 0.95 - log P(<eot> | k))], with
             M = P(":") P(" ") sum_k P(k) at the confidence position; rows whose sampled response
             does not start with ": " are left to the -30 format reward.
    python subset.py evaluate IDS_JSON MODEL_DIR OUT_JSON

`evaluate` reports the released metrics unchanged (sampled integer confidence) and adds
unsampled columns at no extra cost: generate() also returns the raw logits of every step, and
the softmax at the step that emitted the confidence number is pi over the 11 levels. ECE /
AUROC / Brier are computed from the mean confidence sum_k pi_k k/10 (and ECE from the most
likely level) on the same rows and labels. Requesting logits does not change sampling.
"""
import json
import os
import shutil
import sys
import time

from datasets import load_dataset

from util import DataHelper
from util.Prompts import get_prompt


def add_ppo_hinge(trainer_class, weight, log_path, threshold=0.95):
    """Add discrete-exact's two format hinges to TRL 0.8.6 PPO's policy loss (see module docstring).

    batched_forward_pass(return_logits=True) is the training forward of each minibatch; its
    queries, responses and attention mask are kept so that loss() can locate, per row, the
    confidence position in the logits TRL passes it (logits[:, t] predicts input token t + 1).
    """
    import math
    import torch

    original_forward, original_loss = trainer_class.batched_forward_pass, trainer_class.loss
    stash, log = {}, open(log_path, "w")

    def batched_forward_pass(self, model, queries, responses, model_inputs, return_logits=False, response_masks=None):
        if return_logits:
            stash.update(queries=queries, responses=responses, attention_mask=model_inputs["attention_mask"])
        return original_forward(self, model, queries, responses, model_inputs, return_logits, response_masks)

    def loss(self, old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns):
        pg_loss, vf_loss, stats = original_loss(self, old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns)
        tok = self.tokenizer
        colon, space = tok.encode(": 0", add_special_tokens=False)[:2]
        numbers = torch.tensor([tok.encode(f": {k}", add_special_tokens=False)[-1] for k in range(11)], device=logits.device)
        stop = tok.convert_tokens_to_ids("<|eot_id|>")
        terms, masses, stops = [], [], []
        for j, (query, response) in enumerate(zip(stash["queries"], stash["responses"])):
            response = response.tolist()
            if response[:2] != [colon, space]:
                continue
            o = int(stash["attention_mask"][j].nonzero()[0]) + len(query)  # first response position
            lp = lambda index: logits[j, index].float().log_softmax(-1)
            log_mass = lp(o - 1)[colon] + lp(o)[space] + lp(o + 1)[numbers].logsumexp(-1)
            term = (math.log(threshold) - log_mass).clamp(min=0)
            masses.append(log_mass.exp().item())
            if len(response) > 2 and response[2] in numbers.tolist():
                log_stop = lp(o + 2)[stop]
                term = term + (math.log(threshold) - log_stop).clamp(min=0)
                stops.append(log_stop.exp().item())
            terms.append(term)
        if terms:
            pg_loss = pg_loss + weight * torch.stack(terms).mean()
        log.write(json.dumps(dict(rows=len(stash["queries"]), hinged=len(terms),
                                  mass=sum(masses) / max(1, len(masses)), stop=sum(stops) / max(1, len(stops)))) + "\n")
        log.flush()
        return pg_loss, vf_loss, stats

    trainer_class.batched_forward_pass, trainer_class.loss = batched_forward_pass, loss


def subset_loader(ids_by_split):
    def load_prepared_dataset(dataset, split, method, tokenizer):
        # util/DataHelper.load_prepared_dataset, with one added filter step.
        descriptor = DataHelper.get_dataset_descriptor(dataset)
        data = load_dataset(**descriptor.huggingface_config, split=split)
        keep = set(ids_by_split[split])
        data = data.filter(lambda x: x["question_id"] in keep)
        if len(data) != len(keep):
            raise ValueError(f"{split}: found {len(data)} of {len(keep)} requested questions")
        data = data.map(lambda x: descriptor.normalize_function(x), remove_columns=descriptor.columns_to_remove)
        system_prompt = get_prompt(descriptor.type)
        return data.map(lambda x: DataHelper.prepare_queries(x, tokenizer, system_prompt, tokenize=True))
    return load_prepared_dataset


def unsampled_metrics(tokenizer, captured, results):
    """Confidence read from the 11-level softmax at the step that emitted the number."""
    import torch
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from torchmetrics.classification import BinaryCalibrationError
    from util.EvaluationMetrics import Metric, is_answer_correct

    numbers = [tokenizer.encode(f": {k}", add_special_tokens=False)[-1] for k in range(11)]
    assert len(set(numbers)) == 11, "each level 0..10 must be a single token"
    confidence_token = tokenizer.convert_tokens_to_ids("ĠConfidence")
    rows = [(seq, logits) for tokens, step_logits in captured for seq, logits in zip(tokens, step_logits)]
    if len(rows) != len(results):
        raise ValueError(f"captured {len(rows)} generations for {len(results)} results")
    means, argmaxes, masses, labels, missing = [], [], [], [], 0
    for (seq, logits), result in zip(rows, results):
        if result.is_wrong_format:  # same rows as the released metrics
            continue
        seq = seq.tolist()
        start = max((i for i, t in enumerate(seq) if t == confidence_token), default=None)
        step = next((i for i in range(start + 1, min(start + 4, len(seq))) if seq[i] in numbers), None) if start is not None else None
        if step is None:
            missing += 1
            continue
        probs = logits[step].softmax(-1)[numbers]
        pi = probs / probs.sum()
        means.append(float((pi * torch.arange(11)).sum()) / 10)
        argmaxes.append(int(pi.argmax()) / 10)
        masses.append(float(probs.sum()))
        labels.append(int(is_answer_correct(result.prediction, result.gt_candidates, Metric.F1, 0.5)))
    ece = lambda p: BinaryCalibrationError(n_bins=11, norm="l1")(torch.tensor(p), torch.tensor(labels)).item()
    return dict(ece_unsampled=ece(means), auroc_unsampled=roc_auc_score(labels, means),
                brier_unsampled=brier_score_loss(labels, means), ece_unsampled_argmax=ece(argmaxes),
                number_mass_unsampled=sum(masses) / len(masses), unsampled_rows=len(labels),
                unsampled_missing=missing)


def main():
    command, ids_path = sys.argv[1], sys.argv[2]
    ids_by_split = json.load(open(ids_path))
    loader = subset_loader(ids_by_split)
    if command == "train":
        import Train
        from util.ppo_trainer_no_cache import PPOTrainerNoCache

        Train.load_prepared_dataset = loader
        original_save = PPOTrainerNoCache.save_pretrained
        epochs_saved = []

        def save_pretrained(self, path, *args, **kwargs):
            original_save(self, path, *args, **kwargs)
            if os.path.basename(path) == "model_finetuned":
                epochs_saved.append(path)
                shutil.copytree(path, f"{path}_epoch{len(epochs_saved)}", dirs_exist_ok=True)

        PPOTrainerNoCache.save_pretrained = save_pretrained
        args = Train.setup_parser().parse_args(sys.argv[sys.argv.index("--") + 1:])
        ours = sys.argv[:sys.argv.index("--")]
        option = lambda name, cast, default: cast(ours[ours.index(name) + 1]) if name in ours else default
        save_every = option("--save-every", int, 0)
        seed = option("--seed", int, None)
        hinge_weight = option("--hinge", float, 0.)
        if seed is not None:
            import random
            import numpy as np
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        if "--no-kl" in ours:
            original_config = Train.PPOConfig
            Train.PPOConfig = lambda **kwargs: original_config(**{**kwargs, "init_kl_coef": 0.0, "adap_kl_ctrl": False})
        if hinge_weight:
            add_ppo_hinge(PPOTrainerNoCache, hinge_weight, os.path.join(args.out_dir, "hinge.jsonl"))
        # Timestamp every PPO step (and save periodic snapshots) without touching Train.py.
        os.makedirs(args.out_dir, exist_ok=True)
        timing = open(os.path.join(args.out_dir, "steps.jsonl"), "w")
        original_step = PPOTrainerNoCache.step
        count = [0]

        def step(self, *step_args, **step_kwargs):
            entered = time.time()
            stats = original_step(self, *step_args, **step_kwargs)
            count[0] += 1
            record = dict(step=count[0], enter=entered, exit=time.time())
            if save_every and count[0] % save_every == 0:
                original_save(self, os.path.join(args.out_dir, f"snapshot-step{count[0]:05d}"))
                record["save_seconds"] = time.time() - record["exit"]
            timing.write(json.dumps(record) + "\n")
            timing.flush()
            return stats

        PPOTrainerNoCache.step = step
        Train.train(args.out_dir, lr=args.lr, epochs=args.epochs, batchsize=args.batchsize,
                    model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir, dataset=args.dataset,
                    log_with=args.log_with, is_unsloth=args.is_unsloth)
    elif command == "evaluate":
        import torch
        import InferenceDatasetSplit
        from util.EvaluationMetrics import (Metric, QAResults_to_ECE, QAResults_to_accuracy,
                                            QAResults_to_auroc_score, QAResults_to_brier_score)
        from util.ResponseHandling import load_QAResults

        model_dir, out_path = sys.argv[3], sys.argv[4]
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            # TRL's PPOTrainer.save_pretrained also writes its PPOConfig as config.json and the
            # value head as pytorch_model.bin; Unsloth refuses a directory holding both a
            # config.json and an adapter. Load the policy adapter from a copy without them.
            policy_dir = "/tmp/policy_adapter"
            shutil.rmtree(policy_dir, ignore_errors=True)
            shutil.copytree(model_dir, policy_dir, ignore=shutil.ignore_patterns("config.json", "pytorch_model.bin"))
            model_dir = policy_dir
        InferenceDatasetSplit.load_prepared_dataset = loader
        # Capture generate()'s raw per-step logits; inference_dataset still receives the sequences.
        loaded, captured = {}, []
        original_loader = InferenceDatasetSplit.load_model_tokenizer

        def keep_model(*load_args, **load_kwargs):
            model, tokenizer = original_loader(*load_args, **load_kwargs)
            original_generate = model.generate

            def generate(*gen_args, **gen_kwargs):
                out = original_generate(*gen_args, output_logits=True, return_dict_in_generate=True, **gen_kwargs)
                prompt_len = gen_kwargs["input_ids"].shape[1]
                captured.append((out.sequences[:, prompt_len:].cpu(), torch.stack(out.logits, 1).float().cpu()))
                return out.sequences

            model.generate = generate
            loaded["tokenizer"] = tokenizer
            return model, tokenizer

        InferenceDatasetSplit.load_model_tokenizer = keep_model
        torch.manual_seed(0)
        # Evaluation.py / EvaluateModel.ipynb settings: is_unsloth=True, batch 32, 32 new tokens.
        InferenceDatasetSplit.inference_dataset(model_dir, True, "triviaqa", "validation", out_path)
        results = load_QAResults(out_path)
        settings = dict(metric=Metric.F1, threshold=0.5, max_confidence=10)
        metrics = dict(model=sys.argv[3], n=len(results),
                       wrong_format_rate=sum(r.is_wrong_format for r in results) / len(results),
                       ece=QAResults_to_ECE(results, n_bins=11, **settings),
                       accuracy=QAResults_to_accuracy(results, **settings),
                       auroc=QAResults_to_auroc_score(results, **settings),
                       brier=QAResults_to_brier_score(results, **settings))
        metrics.update(unsampled_metrics(loaded["tokenizer"], captured, results))
        json.dump(metrics, open(out_path.replace(".json", "_metrics.json"), "w"), indent=2)
        print(json.dumps(metrics))
    else:
        raise SystemExit(f"Unknown command {command}")


if __name__ == "__main__":
    main()
