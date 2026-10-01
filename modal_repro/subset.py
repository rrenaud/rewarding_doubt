"""Run the released Train.py / InferenceDatasetSplit.py on a fixed subset of TriviaQA questions.

Runs inside the Modal container from SingleAnswerSetting/. The released training and inference
code is used unmodified; the only intervention is that `load_prepared_dataset`, which those
modules import by name, is replaced by a copy of the original that keeps only the requested
question IDs before normalization (which drops `question_id`). Each epoch's
`model_finetuned` checkpoint is also copied to `model_finetuned_epoch<k>`, because Train.py
overwrites it every epoch.

    python subset.py train IDS_JSON -- <Train.py args>
    python subset.py evaluate IDS_JSON MODEL_DIR OUT_JSON
"""
import json
import os
import shutil
import sys

from datasets import load_dataset

from util import DataHelper
from util.Prompts import get_prompt


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
        json.dump(metrics, open(out_path.replace(".json", "_metrics.json"), "w"), indent=2)
        print(json.dumps(metrics))
    else:
        raise SystemExit(f"Unknown command {command}")


if __name__ == "__main__":
    main()
