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
  --fast     numerically equivalent speedups (see add_ppo_speedups): log-probs only over response
             positions, no entropy statistic, no gradient checkpointing, no per-step empty_cache
  --ppo KEY=VALUE  override a PPOConfig field (repeatable), e.g. --ppo cliprange=0.1 --ppo ppo_epochs=2
  --hinge-threshold T  threshold for --hinge (default 0.95)
  --reward-mix M  mix the Brier score into Train.py's reward: (1 - M) log + M Brier on the released
             scale (rewarding_doubt.core.reward); parsing and -30 unchanged
  --grading f1  grade answers in Train.py's reward with F1 > 0.5 (as the paper describes and as our
             evaluation does) instead of its default exact match; see docs/grading.md
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


def mixed_reward(brier_mix, grading="exact"):
    """util.RLHelper.QAResult_to_reward with the Brier score mixed in and a choice of grader.

    Same parse and -30 for malformed output; Train.py calls it with the default metric, which is
    exact match in the released code and F1 > 0.5 with grading="f1".
    """
    from rewarding_doubt.core import reward
    from util.EvaluationMetrics import Metric, is_answer_correct
    from util.RLHelper import wrong_format_penalty
    default_metric = Metric.F1 if grading == "f1" else Metric.EXACT

    def QAResult_to_reward(result, metric=default_metric, threshold=0.5):
        if result.confidence is None or not 0 <= result.confidence <= 10:
            return wrong_format_penalty
        correct = is_answer_correct(result.prediction, result.gt_candidates, metric, threshold)
        return float(reward(result.confidence / 10, float(correct), "released", brier_mix))
    return QAResult_to_reward


def add_ppo_speedups(trainer_class):
    """Numerically equivalent speedups for TRL 0.8.6 PPO as used by Train.py.

    1. batched_forward_pass computes log-probs (a log-softmax over the 128k vocabulary) only over
       each row's response span instead of every position; TRL masks the other positions anyway.
       Identical values at every position TRL uses; masked positions hold 0.
    2. entropy_from_logits (a logging statistic, ppo/policy/entropy) returns zeros.
    3. Gradient checkpointing is off (util.ModelLoader asks Unsloth for it); same forward pass.
    4. PPOConfig(optimize_device_cache=False) (set by the caller): no gc/empty_cache every step.
    """
    import math
    import torch
    import trl.trainer.ppo_trainer as ppo_module
    import util.ModelLoader as model_loader

    ppo_module.entropy_from_logits = lambda logits: torch.zeros(logits.shape[:-1], device=logits.device)
    original_peft = model_loader.FastLanguageModel.get_peft_model
    model_loader.FastLanguageModel.get_peft_model = staticmethod(
        lambda *a, **k: original_peft(*a, **{**k, "use_gradient_checkpointing": False}))

    def batched_forward_pass(self, model, queries, responses, model_inputs, return_logits=False, response_masks=None):
        # trl 0.8.6 PPOTrainer.batched_forward_pass, decoder-only branch, with log-probs restricted
        # to the response span.
        bs, fbs = len(queries), self.config.mini_batch_size
        all_logprobs, all_logits, all_masks, all_values = [], [], [], []
        model.eval()
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs:(i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch, response_batch = queries[i * fbs:(i + 1) * fbs], responses[i * fbs:(i + 1) * fbs]
            response_masks_batch = response_masks[i * fbs:(i + 1) * fbs] if response_masks is not None else None
            logits, _, values = model(**input_kwargs)
            input_ids, attention_mask = input_kwargs["input_ids"], input_kwargs["attention_mask"]
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]
            logprobs = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=logits.dtype, device=logits.device)
            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])
                if response_masks is not None:
                    response_masks_batch[j] = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]
                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]
                span = logits[j, start:end].log_softmax(-1)  # trl.core.logprobs_from_logits, same op
                logprobs[j, start:end] = span.gather(-1, input_ids[j, start + 1:end + 1, None]).squeeze(-1)
            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)
        return (torch.cat(all_logprobs), torch.cat(all_logits)[:, :-1] if return_logits else None,
                torch.cat(all_values)[:, :-1], torch.cat(all_masks)[:, :-1])

    trainer_class.batched_forward_pass = batched_forward_pass


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
    if len(set(labels)) < 2:  # nothing (or one class) left to score, e.g. a run that broke the format
        return dict(ece_unsampled=None, auroc_unsampled=None, brier_unsampled=None, ece_unsampled_argmax=None,
                    number_mass_unsampled=None, unsampled_rows=len(labels), unsampled_missing=missing)
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
        hinge_threshold = option("--hinge-threshold", float, 0.95)
        reward_mix = option("--reward-mix", float, 0.0)
        grading = option("--grading", str, "exact")
        if grading not in ("exact", "f1"):
            raise SystemExit(f"--grading must be exact or f1, not {grading}")
        if reward_mix or grading == "f1":
            Train.QAResult_to_reward = mixed_reward(reward_mix, grading)
        if seed is not None:
            import random
            import numpy as np
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        # PPOTrainer.__init__ calls set_seed(config.seed) (default 0), which would override the
        # reseeding above, so the seed must also go into PPOConfig.
        config_overrides = {"seed": seed} if seed is not None else {}
        if "--no-kl" in ours:
            config_overrides.update(init_kl_coef=0.0, adap_kl_ctrl=False)
        if "--fast" in ours:
            config_overrides.update(optimize_device_cache=False)
            add_ppo_speedups(PPOTrainerNoCache)
        for i, arg in enumerate(ours):
            if arg == "--ppo":
                key, value = ours[i + 1].split("=", 1)
                config_overrides[key] = int(value) if value.isdigit() else float(value)
        if config_overrides:
            original_config = Train.PPOConfig
            Train.PPOConfig = lambda **kwargs: original_config(**{**kwargs, **config_overrides})
        if hinge_weight:
            add_ppo_hinge(PPOTrainerNoCache, hinge_weight, os.path.join(args.out_dir, "hinge.jsonl"), hinge_threshold)
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
