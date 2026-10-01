"""Run the released Rewarding Doubt code unmodified on Modal GPUs.

    modal run modal_repro/app.py::smoke                   # stage 1: environment check
    modal run modal_repro/app.py::smoke --minutes 20 --gpu L40S
    modal run modal_repro/app.py::stage2 --data-run runs/pilot-20261001T002945Z   # our subset
    modal run modal_repro/app.py::exact --max-steps 3                                # smoke
    modal run modal_repro/app.py::exact                                              # both objectives
    modal run modal_repro/app.py::curves                     # PPO + exact (1 and 8 updates/batch), snapshots

The image clones pasta99/RewardingDoubt at a pinned commit and installs its full pinned
requirements.txt (torch 2.5.1, transformers 4.48.0, trl 0.8.6, unsloth @ d6982c1). The Hugging
Face cache and training outputs live on a Volume, so model and dataset downloads happen once.
Results land in ./runs/modal-<stage>-<timestamp>/.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import modal

REPO = "https://github.com/pasta99/RewardingDoubt"
COMMIT = "416c9e0de22a387c35a51260e7739fa28a37865e"
CODE = "/opt/RewardingDoubt/SingleAnswerSetting"
VOL = "/vol"
MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        f"git clone {REPO} /opt/RewardingDoubt",
        f"git -C /opt/RewardingDoubt checkout {COMMIT}",
        "pip install --no-cache-dir -r /opt/RewardingDoubt/requirements.txt",
    )
    .add_local_file(Path(__file__).parent / "subset.py", f"{CODE}/subset.py", copy=True)
    .add_local_file(Path(__file__).parent / "exact_llama.py", f"{CODE}/exact_llama.py", copy=True)
    .add_local_file(Path(__file__).parent / "shared_prefix.py", f"{CODE}/shared_prefix.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_shared_prefix.py", f"{CODE}/verify_shared_prefix.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_single_pass.py", f"{CODE}/verify_single_pass.py", copy=True)
    # Our package, so the exact objectives use the very same core.objective as the Tinker runs.
    .add_local_dir(Path(__file__).parents[1] / "src" / "rewarding_doubt", "/opt/rd/rewarding_doubt", copy=True,
                   ignore=["__pycache__"])
    .env({"PYTHONPATH": "/opt/rd"})
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled"})
)

volume = modal.Volume.from_name("rewarding-doubt-repro", create_if_missing=True)
app = modal.App("rewarding-doubt-repro", image=image)
HOUR = 3600


def tensorboard_scalars(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Every scalar TRL logged, as {tag: [(step, value), ...]}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalars: dict[str, list[tuple[int, float]]] = {}
    for events in Path(log_dir).rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            scalars.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    return {tag: sorted(values) for tag, values in scalars.items()}


@app.function(volumes={VOL: volume}, timeout=3 * HOUR)
def train_for(minutes: float, args: list[str], run_name: str) -> dict:
    """Run Train.py unmodified, stop it after `minutes`, and return its logs and scalars.

    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess
    import time

    out_dir = f"{VOL}/outputs/{run_name}"
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)  # Train.py uses os.mkdir
    command = ["python", "Train.py", "--out_dir", out_dir, *args]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    start = time.time()
    with open(f"/tmp/{run_name}.log", "w") as log:
        proc = subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
        try:
            exit_code = proc.wait(timeout=minutes * 60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            exit_code = None  # stopped by the time budget, as intended
    elapsed = time.time() - start
    volume.commit()
    return dict(command=command, gpu=gpu, exit_code=exit_code, elapsed_s=elapsed,
                log=Path(f"/tmp/{run_name}.log").read_text(), pip_freeze=freeze,
                scalars=tensorboard_scalars(out_dir) if Path(out_dir).exists() else {})


@app.function(volumes={VOL: volume}, timeout=8 * HOUR)
def train_and_evaluate_subset(ids_by_split: dict, train_args: list[str], run_name: str) -> dict:
    """Train.py on a question subset, then the released inference + metrics on held-out IDs.

    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess
    import time

    out_dir = f"{VOL}/outputs/{run_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ids_path = f"{out_dir}/ids.json"
    Path(ids_path).write_text(json.dumps(ids_by_split))
    logs, timings = {}, {}
    if Path(f"{out_dir}/train.log").exists():
        logs["train"] = Path(f"{out_dir}/train.log").read_text()

    def run(name, command):
        start = time.time()
        with open(f"{out_dir}/{name}.log", "w") as log:
            code = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
        timings[name] = time.time() - start
        logs[name] = Path(f"{out_dir}/{name}.log").read_text()
        volume.commit()
        return code

    train_code = (run("train", ["python", "subset.py", "train", ids_path, "--", "--out_dir", out_dir, *train_args])
                  if train_args else None)  # no train_args: re-evaluate an existing run's checkpoints
    models = {"base": MODEL} if train_args else {}
    for name in ["model_finetuned_epoch1", "model_finetuned_epoch2", "model_finetuned_best"]:
        if Path(f"{out_dir}/{name}").exists():
            models[name] = f"{out_dir}/{name}"
    metrics, results = {}, {}
    for name, model_dir in models.items():
        out_json = f"{out_dir}/eval_{name}.json"
        if run(f"eval_{name}", ["python", "subset.py", "evaluate", ids_path, model_dir, out_json]) == 0:
            metrics[name] = json.loads(Path(out_json.replace(".json", "_metrics.json")).read_text())
            results[name] = json.loads(Path(out_json).read_text())
    return dict(train_exit_code=train_code, timings=timings, metrics=metrics, results=results, logs=logs,
                scalars=tensorboard_scalars(out_dir))


@app.function(volumes={VOL: volume}, timeout=8 * HOUR)
def train_and_evaluate_exact(ids_by_split: dict, mode: str, run_name: str, max_steps: int = 0) -> dict:
    """exact_llama.py on the subset, then the released evaluation on each epoch's adapter.

    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess
    import time

    out_dir = f"{VOL}/outputs/{run_name}/{mode}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ids_path = f"{out_dir}/ids.json"
    Path(ids_path).write_text(json.dumps(ids_by_split))
    logs, timings, metrics, results = {}, {}, {}, {}

    def run(name, command):
        start = time.time()
        with open(f"{out_dir}/{name}.log", "w") as log:
            code = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
        timings[name] = time.time() - start
        logs[name] = Path(f"{out_dir}/{name}.log").read_text()
        volume.commit()
        return code

    train_code = run("train", ["python", "exact_llama.py", ids_path, out_dir, "--mode", mode,
                               "--max-steps", str(max_steps)])
    for epoch in (1, 2):
        model_dir = f"{out_dir}/model_finetuned_epoch{epoch}"
        if not Path(model_dir).exists():
            continue
        name = f"model_finetuned_epoch{epoch}"
        out_json = f"{out_dir}/eval_{name}.json"
        if run(f"eval_{name}", ["python", "subset.py", "evaluate", ids_path, model_dir, out_json]) == 0:
            metrics[name] = json.loads(Path(out_json.replace(".json", "_metrics.json")).read_text())
            results[name] = json.loads(Path(out_json).read_text())
    train_metrics = Path(f"{out_dir}/metrics.jsonl")
    return dict(mode=mode, train_exit_code=train_code, timings=timings, metrics=metrics, results=results,
                logs=logs, train_metrics=train_metrics.read_text() if train_metrics.exists() else "")


@app.function(volumes={VOL: volume}, timeout=8 * HOUR)
def train_with_snapshots(ids_by_split: dict, label: str, command: list[str], run_name: str) -> dict:
    """Train one configuration, saving snapshots; return snapshot dirs and per-step timing.

    `command` is the script invocation after `python`, with OUT_DIR as a placeholder.
    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess

    out_dir = f"{VOL}/outputs/{run_name}/{label}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ids_path = f"{out_dir}/ids.json"
    Path(ids_path).write_text(json.dumps(ids_by_split))
    command = [arg.replace("OUT_DIR", out_dir).replace("IDS_JSON", ids_path) for arg in command]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    with open(f"{out_dir}/train.log", "w") as log:
        code = subprocess.run(["python", *command], cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
    volume.commit()
    timing = next((Path(out_dir) / name for name in ["metrics.jsonl", "steps.jsonl"]
                   if (Path(out_dir) / name).exists()), None)
    snapshots = sorted(str(p) for p in Path(out_dir).iterdir() if p.name.startswith("snapshot-step"))
    return dict(label=label, exit_code=code, gpu=gpu, snapshots=snapshots,
                timing=timing.read_text() if timing else "", log=Path(f"{out_dir}/train.log").read_text())


@app.function(volumes={VOL: volume}, timeout=HOUR)
def evaluate_checkpoint(ids_by_split: dict, model_dir: str) -> dict:
    """The released evaluation (subset.py evaluate) on one checkpoint directory.

    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess

    volume.reload()
    ids_path = "/tmp/ids.json"
    Path(ids_path).write_text(json.dumps(ids_by_split))
    out_json = f"{model_dir}/eval.json"
    proc = subprocess.run(["python", "subset.py", "evaluate", ids_path, model_dir, out_json], cwd=CODE,
                          capture_output=True, text=True)
    volume.commit()
    if proc.returncode:
        return dict(model_dir=model_dir, error=proc.stdout[-3000:] + proc.stderr[-3000:])
    return dict(model_dir=model_dir, metrics=json.loads(Path(out_json.replace(".json", "_metrics.json")).read_text()),
                results=json.loads(Path(out_json).read_text()))


@app.function(timeout=600)
def shell(command: str) -> str:
    """Run a shell command in the pinned image (CPU only); for inspecting installed code."""
    import subprocess
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    return proc.stdout + proc.stderr


@app.local_entrypoint()
def inspect(command: str):
    print(shell.remote(command))


def summarize(scalars: dict[str, list[tuple[int, float]]]) -> list[str]:
    lines = []
    for tag in ["env/reward_mean", "objective/kl", "objective/kl_coef", "ppo/policy/clipfrac",
                "ppo/val/vpred", "ppo/loss/value", "time/ppo/total"]:
        values = scalars.get(tag)
        if values:
            first, last = values[: max(1, len(values) // 5)], values[-max(1, len(values) // 5):]
            mean = lambda xs: sum(v for _, v in xs) / len(xs)
            lines.append(f"{tag}: {len(values)} steps, first-fifth mean {mean(first):.4g}, "
                         f"last-fifth mean {mean(last):.4g}")
    return lines


@app.local_entrypoint()
def smoke(minutes: float = 25, gpu: str = "L40S", batchsize: int = 2):
    """Stage 1: confirm the pinned environment runs Train.py on TriviaQA and PPO steps proceed."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"smoke-{stamp}"
    # Paper (Sec. 4): TriviaQA, two epochs, learning rate 1e-5, Unsloth 4-bit Llama-3-8B-Instruct.
    args = ["--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL, "--tokenizer_dir", MODEL,
            "--epochs", "2", "--lr", "1e-5", "--batchsize", str(batchsize), "--log_with", "tensorboard"]
    result = train_for.with_options(gpu=gpu).remote(minutes, args, run_name)
    local = Path(__file__).resolve().parents[1] / "runs" / f"modal-{run_name}"
    local.mkdir(parents=True)
    (local / "train.log").write_text(result.pop("log"))
    (local / "pip-freeze.txt").write_text(result.pop("pip_freeze"))
    (local / "scalars.json").write_text(json.dumps(result.pop("scalars"), indent=1) + "\n")
    (local / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("\n".join(summarize(json.loads((local / "scalars.json").read_text()))))
    print(f"Saved to {local}")


@app.local_entrypoint()
def stage2(data_run: str = "runs/pilot-20261001T002945Z", gpu: str = "L40S", batchsize: int = 8):
    """Released training code on our 1,024 training questions, evaluated on our 512 held-out ones."""
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"stage2-{stamp}"
    args = ["--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL, "--tokenizer_dir", MODEL,
            "--epochs", "2", "--lr", "1e-5", "--batchsize", str(batchsize), "--log_with", "tensorboard"]
    local = root / "runs" / f"modal-{run_name}"
    local.mkdir(parents=True)
    (local / "ids.json").write_text(json.dumps(ids) + "\n")
    result = train_and_evaluate_subset.with_options(gpu=gpu).remote(ids, args, run_name)
    for name, text in result.pop("logs").items():
        (local / f"{name}.log").write_text(text)
    for name, rows in result.pop("results").items():
        (local / f"eval_{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    (local / "scalars.json").write_text(json.dumps(result.pop("scalars"), indent=1) + "\n")
    result.update(data_run=data_run, gpu=gpu, train_args=args)
    (local / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved to {local}")


@app.local_entrypoint()
def stage2_evaluate(run_dir: str, gpu: str = "L40S"):
    """Re-run the released evaluation on the checkpoints an earlier stage2 run left on the Volume."""
    local = Path(run_dir).resolve()
    run_name = local.name.removeprefix("modal-")
    ids = json.loads((local / "ids.json").read_text())
    result = train_and_evaluate_subset.with_options(gpu=gpu).remote(ids, [], run_name)
    for name, text in result.pop("logs").items():
        if name != "train":
            (local / f"{name}.log").write_text(text)
    for name, rows in result.pop("results").items():
        (local / f"eval_{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    previous = json.loads((local / "result.json").read_text())
    previous["metrics"].update(result["metrics"])
    previous["timings"].update(result["timings"])
    (local / "result.json").write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps(previous["metrics"], indent=2))


@app.local_entrypoint()
def exact(data_run: str = "runs/pilot-20261001T002945Z", gpu: str = "L40S", max_steps: int = 0,
          modes: str = "discrete-exact,fractional"):
    """Our exact objectives on the released code's Llama setup; one GPU container per objective."""
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"exact-{'smoke-' if max_steps else ''}{stamp}"
    local = root / "runs" / f"modal-{run_name}"
    local.mkdir(parents=True)
    (local / "ids.json").write_text(json.dumps(ids) + "\n")
    calls = [(ids, mode, run_name, max_steps) for mode in modes.split(",")]
    summary = {}
    for result in train_and_evaluate_exact.with_options(gpu=gpu).starmap(calls, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"FAILED: {result!r}")
            continue
        mode_dir = local / result["mode"]
        mode_dir.mkdir()
        for name, text in result.pop("logs").items():
            (mode_dir / f"{name}.log").write_text(text)
        for name, rows in result.pop("results").items():
            (mode_dir / f"eval_{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        (mode_dir / "metrics.jsonl").write_text(result.pop("train_metrics"))
        (mode_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        summary[result["mode"]] = result
        print(json.dumps(result, indent=2))
    (local / "summary.json").write_text(json.dumps(
        {f"{mode}-{name}": m for mode, r in summary.items() for name, m in r["metrics"].items()}, indent=2) + "\n")
    print(f"Saved to {local}")


@app.local_entrypoint()
def curves(data_run: str = "runs/pilot-20261001T002945Z", gpu: str = "L40S", save_every: int = 32,
           labels: str = "ppo,discrete-exact-1x,fractional-1x,discrete-exact-8x,fractional-8x",
           train_limit: int = 0, eval_limit: int = 0):
    """Learning curves on the released Llama setup: PPO vs exact objectives at 1 and 8 updates/batch.

    Every configuration saves a snapshot every `save_every` steps and timestamps each step; every
    snapshot is then scored with the released evaluation, in parallel containers.
    """
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    if train_limit or eval_limit:  # smoke tests
        ids = {"train": ids["train"][:train_limit or None], "validation": ids["validation"][:eval_limit or None]}
    configs = {
        "ppo": ["subset.py", "train", "IDS_JSON", "--save-every", str(save_every), "--",
                "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL,
                "--tokenizer_dir", MODEL, "--epochs", "2", "--lr", "1e-5", "--batchsize", "8",
                "--log_with", "tensorboard"],
    }
    for mode in ["discrete-exact", "fractional"]:
        configs[f"{mode}-1x"] = ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", mode, "--save-every", str(save_every)]
        configs[f"{mode}-8x"] = [*configs[f"{mode}-1x"], "--passes", "4", "--minibatch", "4"]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"curves-{'smoke-' if train_limit or eval_limit else ''}{stamp}"
    local = root / "runs" / f"modal-{run_name}"
    local.mkdir(parents=True)
    (local / "ids.json").write_text(json.dumps(ids) + "\n")
    (local / "configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    chosen = labels.split(",")
    trained = {}
    for result in train_with_snapshots.with_options(gpu=gpu).starmap(
            [(ids, label, configs[label], run_name) for label in chosen], return_exceptions=True):
        if isinstance(result, Exception):
            print(f"TRAINING FAILED: {result!r}")
            continue
        label_dir = local / result["label"]
        label_dir.mkdir()
        (label_dir / "train.log").write_text(result.pop("log"))
        (label_dir / "timing.jsonl").write_text(result.pop("timing"))
        (label_dir / "train.json").write_text(json.dumps(result, indent=2) + "\n")
        trained[result["label"]] = result
        print(f"trained {result['label']}: exit {result['exit_code']}, {len(result['snapshots'])} snapshots")
    jobs = [(ids, snap) for result in trained.values() for snap in result["snapshots"]]
    curve = {}
    for result in evaluate_checkpoint.with_options(gpu=gpu).starmap(jobs, return_exceptions=True):
        if isinstance(result, Exception) or "error" in result:
            print(f"EVAL FAILED: {result if isinstance(result, Exception) else result['error'][-500:]}")
            continue
        label, snap = result["model_dir"].split("/")[-2:]
        step = int(snap.removeprefix("snapshot-step"))
        (local / label / f"eval_step{step:05d}.json").write_text(json.dumps(result["results"]) + "\n")
        curve.setdefault(label, {})[step] = result["metrics"]
    (local / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    for label, points in curve.items():
        print(label, {step: round(m["ece"], 3) for step, m in sorted(points.items())})
    print(f"Saved to {local}")


@app.function(volumes={VOL: volume}, timeout=HOUR, gpu="L40S")
def verify_shared_prefix_remote(ids_by_split: dict) -> dict:
    import subprocess

    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    proc = subprocess.run(["python", "verify_shared_prefix.py", "/tmp/ids.json"], cwd=CODE,
                          capture_output=True, text=True)
    report = json.loads(Path("/tmp/verify_report.json").read_text()) if Path("/tmp/verify_report.json").exists() else None
    return dict(exit_code=proc.returncode, log=proc.stdout[-20000:] + proc.stderr[-8000:], report=report)


@app.local_entrypoint()
def verify_shared_prefix(data_run: str = "runs/pilot-20261001T002945Z"):
    root = Path(__file__).resolve().parents[1]
    ids = {"train": [json.loads(l)["id"] for l in (root / data_run / "data" / "train.jsonl").read_text().splitlines()]}
    result = verify_shared_prefix_remote.remote(ids)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local = root / "runs" / f"modal-verify-shared-prefix-{stamp}"
    local.mkdir(parents=True)
    (local / "log.txt").write_text(result["log"])
    (local / "report.json").write_text(json.dumps(result["report"], indent=1) + "\n")
    print("exit", result["exit_code"])
    if result["report"]:
        for c in result["report"]["cases"]:
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items() if k != "logp_full"})
        print(result["report"]["seconds_per_question_fwd_bwd"])
    else:
        print(result["log"][-4000:])
    print(f"Saved to {local}")


@app.function(volumes={VOL: volume}, timeout=HOUR, gpu="L40S")
def verify_single_pass_remote(ids_by_split: dict, adapter: str) -> dict:
    import subprocess

    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    proc = subprocess.run(["python", "verify_single_pass.py", "/tmp/ids.json", adapter], cwd=CODE,
                          capture_output=True, text=True)
    path = Path("/tmp/verify_single_report.json")
    return dict(exit_code=proc.returncode, log=proc.stdout[-20000:] + proc.stderr[-8000:],
                report=json.loads(path.read_text()) if path.exists() else None)


@app.local_entrypoint()
def verify_single_pass(data_run: str = "runs/pilot-20261001T002945Z",
                       adapter: str = f"{VOL}/outputs/exact-20261001T171427Z/discrete-exact/model_finetuned_epoch2"):
    root = Path(__file__).resolve().parents[1]
    ids = {"train": [json.loads(l)["id"] for l in (root / data_run / "data" / "train.jsonl").read_text().splitlines()]}
    result = verify_single_pass_remote.remote(ids, adapter)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local = root / "runs" / f"modal-verify-single-pass-{stamp}"
    local.mkdir(parents=True)
    (local / "log.txt").write_text(result["log"])
    (local / "report.json").write_text(json.dumps(result["report"], indent=1) + "\n")
    print("exit", result["exit_code"])
    if not result["report"]:
        print(result["log"][-4000:])
    print(f"Saved to {local}")


@app.local_entrypoint()
def evaluate_dirs(dirs: str, data_run: str = "runs/pilot-20261001T002945Z", gpu: str = "L40S"):
    """Released evaluation (+ unsampled columns) on checkpoint directories already on the Volume."""
    root = Path(__file__).resolve().parents[1]
    ids = {"validation": [json.loads(l)["id"] for l in (root / data_run / "data" / "eval.jsonl").read_text().splitlines()]}
    for result in evaluate_checkpoint.with_options(gpu=gpu).starmap([(ids, d) for d in dirs.split(",")],
                                                                    return_exceptions=True):
        if isinstance(result, Exception) or "error" in result:
            print("FAILED", result if isinstance(result, Exception) else result["error"][-3000:])
        else:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["metrics"].items()}))
