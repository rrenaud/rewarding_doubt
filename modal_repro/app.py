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
    # The minimal-diff exact-confidence option for the released Train.py (--objective exact; default ppo
    # leaves the released behaviour unchanged). See patches/exact_confidence.patch.
    .add_local_file(Path(__file__).parents[1] / "patches" / "exact_confidence.patch", "/opt/exact_confidence.patch", copy=True)
    .run_commands("git -C /opt/RewardingDoubt apply /opt/exact_confidence.patch")
    .add_local_file(Path(__file__).parent / "subset.py", f"{CODE}/subset.py", copy=True)
    .add_local_file(Path(__file__).parent / "exact_llama.py", f"{CODE}/exact_llama.py", copy=True)
    .add_local_file(Path(__file__).parent / "shared_prefix.py", f"{CODE}/shared_prefix.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_shared_prefix.py", f"{CODE}/verify_shared_prefix.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_single_pass.py", f"{CODE}/verify_single_pass.py", copy=True)
    .add_local_file(Path(__file__).parent / "bench_batching.py", f"{CODE}/bench_batching.py", copy=True)
    .add_local_file(Path(__file__).parent / "thinking_llama.py", f"{CODE}/thinking_llama.py", copy=True)
    .add_local_file(Path(__file__).parent / "frozen_eval.py", f"{CODE}/frozen_eval.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_levels.py", f"{CODE}/verify_levels.py", copy=True)
    .add_local_file(Path(__file__).parent / "debug_levels.py", f"{CODE}/debug_levels.py", copy=True)
    .add_local_file(Path(__file__).parent / "verify_fast_levels.py", f"{CODE}/verify_fast_levels.py", copy=True)
    .add_local_file(Path(__file__).parent / "fast_loop.py", f"{CODE}/fast_loop.py", copy=True)
    .add_local_file(Path(__file__).parent / "minimal_trainer.py", f"{CODE}/minimal_trainer.py", copy=True)
    .add_local_file(Path(__file__).parent / "attn_bias_check.py", f"{CODE}/attn_bias_check.py", copy=True)
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

    volume.reload()  # a warm container must see checkpoints committed by other containers (resume)
    out_dir = f"{VOL}/outputs/{run_name}/{label}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ids_path = f"{out_dir}/ids.json"
    Path(ids_path).write_text(json.dumps(ids_by_split))
    command = [arg.replace("OUT_DIR", out_dir).replace("IDS_JSON", ids_path) for arg in command]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:  # a CPU-only call
        gpu = "none"
    # A preempted container is restarted with the same input; the trainer then resumes from its last
    # checkpoint (subset.py, exact_llama.py), so checkpoints must reach the volume as soon as they exist:
    # commit whenever the trainer announces a saved checkpoint, snapshot or status (RD_SAVED lines).
    from rewarding_doubt.checkpoint import SAVED_MARKER
    with open(f"{out_dir}/train.log", "a") as log:
        log.write(f"=== attempt {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} on {gpu} ===\n")
        proc = subprocess.Popen(["python", *command], cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            if line.startswith(SAVED_MARKER):
                log.flush()
                volume.commit()
        code = proc.wait()
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


@app.function(volumes={VOL: volume}, timeout=8 * HOUR)
def train_group(ids_by_split: dict, items: list, run_name: str) -> list:
    """Train several configurations concurrently on one GPU, one process each.

    `items` is [(label, command), ...] as for train_with_snapshots. Use one item per GPU: with
    batched updates a single run already keeps an L40S busy (see exact_variants). Declared without a
    GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    procs = []
    for label, command in items:
        out_dir = f"{VOL}/outputs/{run_name}/{label}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ids_path = f"{out_dir}/ids.json"
        Path(ids_path).write_text(json.dumps(ids_by_split))
        args = [a.replace("OUT_DIR", out_dir).replace("IDS_JSON", ids_path) for a in command]
        log = open(f"{out_dir}/train.log", "w")
        procs.append((label, out_dir, log, subprocess.Popen(["python", *args], cwd=CODE, stdout=log, stderr=subprocess.STDOUT)))
    results = []
    for label, out_dir, log, proc in procs:
        code = proc.wait()
        log.close()
        timing = next((Path(out_dir) / n for n in ["metrics.jsonl", "steps.jsonl"] if (Path(out_dir) / n).exists()), None)
        results.append(dict(label=label, exit_code=code, gpu=gpu, runs_on_gpu=len(items),
                            scalars=tensorboard_scalars(out_dir),
                            snapshots=sorted(str(p) for p in Path(out_dir).iterdir() if p.name.startswith("snapshot-step")),
                            timing=timing.read_text() if timing else "", log=Path(f"{out_dir}/train.log").read_text()))
    volume.commit()
    return results


@app.function(volumes={VOL: volume}, timeout=2 * HOUR)
def evaluate_group(ids_by_split: dict, model_dirs: list) -> list:
    """The released evaluation on several checkpoints concurrently on one GPU.

    Declared without a GPU; callers attach one with `.with_options(gpu=...)`.
    """
    import subprocess

    volume.reload()
    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    procs = [(d, subprocess.Popen(["python", "subset.py", "evaluate", "/tmp/ids.json", d, f"{d}/eval.json"], cwd=CODE,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for d in model_dirs]
    results = []
    for d, proc in procs:
        log = proc.communicate()[0]
        if proc.returncode:
            results.append(dict(model_dir=d, error=log[-3000:]))
        else:
            results.append(dict(model_dir=d, metrics=json.loads(Path(f"{d}/eval_metrics.json").read_text()),
                                results=json.loads(Path(f"{d}/eval.json").read_text())))
    volume.commit()
    return results


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
    _train_and_evaluate(configs, labels.split(","), ids, run_name, gpu)


def _train_and_evaluate(configs: dict, chosen: list, ids: dict, run_name: str, gpu: str, per_gpu: int = 1):
    """Train the chosen configurations (`per_gpu` per container), then score every snapshot."""
    root = Path(__file__).resolve().parents[1]
    local = root / "runs" / f"modal-{run_name}"
    local.mkdir(parents=True)
    (local / "ids.json").write_text(json.dumps(ids) + "\n")
    (local / "configs.json").write_text(json.dumps({k: configs[k] for k in chosen}, indent=2) + "\n")
    trained = {}
    groups = [[(label, configs[label]) for label in chosen[i:i + per_gpu]] for i in range(0, len(chosen), per_gpu)]
    flat = []
    for group in train_group.with_options(gpu=gpu).starmap([(ids, g, run_name) for g in groups], return_exceptions=True):
        flat.extend([group] if isinstance(group, Exception) else group)
    for result in flat:
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
    snaps = [snap for result in trained.values() for snap in result["snapshots"]]
    eval_groups = [snaps[i:i + per_gpu] for i in range(0, len(snaps), per_gpu)]
    evaluated = []
    for group in evaluate_group.with_options(gpu=gpu).starmap([(ids, g) for g in eval_groups], return_exceptions=True):
        evaluated.extend([group] if isinstance(group, Exception) else group)
    curve = {}
    for result in evaluated:
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


@app.local_entrypoint()
def baseline_matched(data_run: str = "runs/pilot-20261001T002945Z", gpu: str = "L40S", seeds: str = "1,2,3",
                     modes: str = "discrete-exact,fractional", save_every: int = 64,
                     train_limit: int = 0, eval_limit: int = 0):
    """Single-pass exact objectives with the released PPO's own reward, invalid penalty and KL,
    8 updates per batch, several seeds."""
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    if train_limit or eval_limit:  # smoke tests
        ids = {"train": ids["train"][:train_limit or None], "validation": ids["validation"][:eval_limit or None]}
    configs = {f"{mode}-baseline-s{seed}": [
        "exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", mode, "--seed", seed, "--scoring", "single",
        "--regularization", "baseline", "--reward", "released", "--passes", "4", "--minibatch", "4",
        "--save-every", str(save_every)] for mode in modes.split(",") for seed in seeds.split(",")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"baseline-matched-{'smoke-' if train_limit or eval_limit else ''}{stamp}"
    _train_and_evaluate(configs, list(configs), ids, run_name, gpu)


EXACT_VARIANTS = {
    # Discrete-exact, single pass, 8 updates/batch, released reward + -30 + KL, plus TRL's value
    # head as an auxiliary regression of the expected reward.
    "de-kl-vh": ["--regularization", "baseline", "--reward", "released", "--value-head"],
    # Discrete-exact, single pass, 8 updates/batch, paper-scale reward with the format hinges, no KL.
    "de-hinge": ["--regularization", "hinge", "--reward", "paper"],
}


@app.local_entrypoint()
def exact_variants(variants: str = "de-kl-vh,de-hinge", seeds: str = "1,2,3", gpu: str = "L40S",
                   data_run: str = "runs/pilot-20261001T002945Z", save_every: int = 64,
                   train_limit: int = 0, eval_limit: int = 0, tag: str = "", per_gpu: int = 1):
    """Seeded runs of discrete-exact variants on the released Llama setup, one run per L40S.

    Packing is not worth it: with batched updates one run keeps the GPU busy (3 runs on one L40S
    took 11.6-12.3 s/step each vs 3.6 s/step alone), and concurrent evaluations run out of memory.
    """
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    if train_limit or eval_limit:
        ids = {"train": ids["train"][:train_limit or None], "validation": ids["validation"][:eval_limit or None]}
    common = ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", "discrete-exact", "--scoring", "single",
              "--passes", "4", "--minibatch", "4", "--save-every", str(save_every)]
    configs = {f"{v}-s{s}": [*common, "--seed", s, *EXACT_VARIANTS[v]] for v in variants.split(",") for s in seeds.split(",")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"exact-variants-{tag + '-' if tag else ''}{'smoke-' if train_limit or eval_limit else ''}{stamp}"
    _train_and_evaluate(configs, list(configs), ids, run_name, gpu, per_gpu)


@app.function(volumes={VOL: volume}, timeout=HOUR)
def bench_batching_remote(ids_by_split: dict) -> dict:
    """Declared without a GPU; callers attach one with `.with_options(gpu=...)`."""
    import subprocess

    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    proc = subprocess.run(["python", "bench_batching.py", "/tmp/ids.json"], cwd=CODE, capture_output=True, text=True)
    path = Path("/tmp/bench_batching.json")
    return dict(exit_code=proc.returncode, log=proc.stdout[-20000:] + proc.stderr[-8000:],
                report=json.loads(path.read_text()) if path.exists() else None)


@app.local_entrypoint()
def bench_batching(gpus: str = "L40S", data_run: str = "runs/pilot-20261001T002945Z"):
    root = Path(__file__).resolve().parents[1]
    ids = {"train": [json.loads(l)["id"] for l in (root / data_run / "data" / "train.jsonl").read_text().splitlines()]}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local = root / "runs" / f"modal-bench-batching-{stamp}"
    local.mkdir(parents=True)
    calls = [(gpu, bench_batching_remote.with_options(gpu=gpu).spawn(ids)) for gpu in gpus.split(",")]
    for gpu, call in calls:
        result = call.get()
        (local / f"{gpu}.log").write_text(result["log"])
        (local / f"{gpu}.json").write_text(json.dumps(result["report"], indent=1) + "\n")
        print(gpu, "exit", result["exit_code"], "saved")
    print(f"Saved to {local}")


PPO_VARIANTS = {
    "ppo-kl": [],                                      # the released configuration, seeded
    "ppo-nokl-hinge": ["--no-kl", "--hinge", "1.0"],   # baseline - KL + discrete-exact's hinges
    "ppo-kl-fast": ["--fast"],                         # same, with numerically equivalent speedups
    "ppo-nokl-hinge-fast": ["--fast", "--no-kl", "--hinge", "1.0"],
}


@app.local_entrypoint()
def ppo_variants(variants: str = "ppo-kl,ppo-nokl-hinge", seeds: str = "1,2,3", gpu: str = "L40S",
                 data_run: str = "runs/pilot-20261001T002945Z", save_every: int = 64,
                 train_limit: int = 0, eval_limit: int = 0):
    """Seeded runs of the released PPO (Train.py unmodified) with and without KL / with the hinge."""
    root = Path(__file__).resolve().parents[1]
    ids = {split: [json.loads(line)["id"] for line in (root / data_run / "data" / f"{name}.jsonl").read_text().splitlines()]
           for split, name in [("train", "train"), ("validation", "eval")]}
    if train_limit or eval_limit:
        ids = {"train": ids["train"][:train_limit or None], "validation": ids["validation"][:eval_limit or None]}
    train_args = ["--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL,
                  "--tokenizer_dir", MODEL, "--epochs", "2", "--lr", "1e-5", "--batchsize", "8", "--log_with", "tensorboard"]
    configs = {f"{v}-s{s}": ["subset.py", "train", "IDS_JSON", "--save-every", str(save_every), "--seed", s,
                             *PPO_VARIANTS[v], "--", *train_args]
               for v in variants.split(",") for s in seeds.split(",")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"ppo-variants-{'smoke-' if train_limit or eval_limit else ''}{stamp}"
    _train_and_evaluate(configs, list(configs), ids, run_name, gpu)


@app.local_entrypoint()
def hparam_round(search_dir: str, round: str, score_test: bool = False, gpu: str = "L40S",
                 train_limit: int = 0, eval_limit: int = 0):
    """Train and score one round of scripts/hparam_search.py on the dev split (and test if asked)."""
    import shutil

    root = Path(__file__).resolve().parents[1]
    search = Path(search_dir).resolve()
    configs = json.loads((search / f"{round}_configs.json").read_text())
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()],
           "validation": json.loads((search / "dev_ids.json").read_text())}
    if train_limit or eval_limit:  # smoke tests
        ids = {"train": ids["train"][:train_limit or None], "validation": ids["validation"][:eval_limit or None]}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"hparam-{round}-{stamp}"
    _train_and_evaluate({k: v["command"] for k, v in configs.items()}, list(configs), ids, run_name, gpu)
    out = search / round
    out.mkdir(exist_ok=True)
    shutil.copy(root / "runs" / f"modal-{run_name}" / "curve.json", out / "curve.json")
    (out / "run.txt").write_text(f"runs/modal-{run_name}\n")
    if score_test:
        test_ids = {"validation": [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()]}
        dirs = [f"{VOL}/outputs/{run_name}/{label}/snapshot-step00256" for label in configs]
        # evaluate_group writes eval.json next to each checkpoint; test results go to a sibling copy.
        test = {}
        for result in evaluate_test.with_options(gpu=gpu).starmap([(test_ids, d) for d in dirs], return_exceptions=True):
            if isinstance(result, Exception) or "error" in result:
                print("TEST EVAL FAILED", result if isinstance(result, Exception) else result["error"][-500:])
                continue
            test[result["model_dir"].split("/")[-2]] = result["metrics"]
        (out / "curve_test.json").write_text(json.dumps(test, indent=2) + "\n")
    print(f"Round {round} done: {out}")


@app.function(volumes={VOL: volume}, timeout=4 * HOUR)
def evaluate_test(ids_by_split: dict, model_dir: str, frozen: bool = False, out_dir: str = "", name: str = "test") -> dict:
    """The released evaluation on the test questions, writing to eval_<name>*.json beside the checkpoint (or in
    out_dir, e.g. for a hub model name); with `frozen`, frozen_eval.py (base-model answers, adapted confidence)
    to eval_<name>_frozen*.json."""
    import subprocess

    volume.reload()
    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    Path(out_dir or model_dir).mkdir(parents=True, exist_ok=True)
    out_json = f"{out_dir or model_dir}/eval_{name}{'_frozen' if frozen else ''}.json"
    if Path(out_json.replace(".json", "_metrics.json")).exists():  # already evaluated (e.g. before a restart)
        return dict(model_dir=model_dir, metrics=json.loads(Path(out_json.replace(".json", "_metrics.json")).read_text()))
    command = ["frozen_eval.py", "/tmp/ids.json", model_dir, out_json] if frozen else ["subset.py", "evaluate", "/tmp/ids.json", model_dir, out_json]
    proc = subprocess.run(["python", *command], cwd=CODE, capture_output=True, text=True)
    volume.commit()
    if proc.returncode:
        return dict(model_dir=model_dir, error=proc.stdout[-3000:] + proc.stderr[-3000:])
    return dict(model_dir=model_dir, metrics=json.loads(Path(out_json.replace(".json", "_metrics.json")).read_text()))


@app.local_entrypoint()
def release_eval(dirs: str, split: str = "dev", gpu: str = "L40S"):
    """The released evaluation (subset.py evaluate: the model writes answer and confidence, F1 > 0.5, torchmetrics
    ECE) of each comma-separated model dir (adapter dirs on the volume, or a hub name for the untrained model), in
    parallel. split: dev (the 512 dev questions), heldout (the validation split without them), full (all of it,
    the paper's test set). Results to eval_<split>_metrics.json beside each adapter (hub models: /vol/release-eval/)."""
    root = Path(__file__).resolve().parents[1]
    dev = json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())
    ids = {"dev": {"validation": dev}, "heldout": {"validation": {"exclude": dev}}, "full": {"validation": "all"}}[split]
    jobs = []
    for d in dirs.split(","):
        out = "" if d.startswith(VOL) else f"{VOL}/release-eval/{d.replace('/', '__')}"
        jobs.append((ids, d, False, out, split))
    for r in evaluate_test.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception) or "error" in r:
            print("FAILED", repr(r)[:500] if isinstance(r, Exception) else r["error"][-1500:]); continue
        m = r["metrics"]
        print("RESULT", json.dumps(dict(model=r["model_dir"], split=split, **{k: round(v, 4) if isinstance(v, float) else v
                                                                            for k, v in m.items() if k != "model"})), flush=True)


@app.local_entrypoint()
def evaluate_base(search_dir: str, gpu: str = "L40S"):
    """The base model on the search's dev split (reference for the accuracy eligibility rule)."""
    search = Path(search_dir).resolve()
    ids = {"validation": json.loads((search / "dev_ids.json").read_text())}
    result = evaluate_base_remote.with_options(gpu=gpu).remote(ids)
    (search / "base_dev_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


@app.function(volumes={VOL: volume}, timeout=HOUR)
def evaluate_base_remote(ids_by_split: dict) -> dict:
    import subprocess

    Path("/tmp/ids.json").write_text(json.dumps(ids_by_split))
    out = "/tmp/base_eval.json"
    proc = subprocess.run(["python", "subset.py", "evaluate", "/tmp/ids.json", MODEL, out], cwd=CODE,
                          capture_output=True, text=True)
    if proc.returncode:
        return dict(error=proc.stdout[-3000:] + proc.stderr[-3000:])
    return json.loads(Path(out.replace(".json", "_metrics.json")).read_text())


@app.function(volumes={VOL: volume}, timeout=600)
def read_files(paths: list) -> dict:
    """Text of each existing file on the volume."""
    volume.reload()
    return {p: Path(p).read_text() for p in paths if Path(p).exists()}


@app.local_entrypoint()
def thinking(configs: str, run_name: str = "", train_limit: int = 512, eval_limit: int = 0, gpu: str = "L40S"):
    """Thinking-before-confidence runs (thinking_llama.py) on the search's train subset and dev split.

    `configs` is a JSON file {label: [thinking_llama.py arguments after IDS_JSON OUT_DIR]}.
    Results land in runs/modal-thinking-{run_name}/{label}/.
    """
    root = Path(__file__).resolve().parents[1]
    search = root / "runs/hparam-search-20261002T044831Z"
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:train_limit or None],
           "validation": json.loads((search / "dev_ids.json").read_text())[:eval_limit or None]}
    configs = json.loads(Path(configs).read_text())
    run_name = run_name or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = f"thinking-{run_name}"
    out = root / "runs" / f"modal-{run}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs.json").write_text(json.dumps(configs, indent=1) + "\n")
    jobs = [(ids, label, ["thinking_llama.py", "IDS_JSON", "OUT_DIR", *args], run) for label, args in configs.items()]
    for result in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print("FAILED", repr(result)[:2000])
            continue
        label = result["label"]
        d = out / label
        d.mkdir(exist_ok=True)
        (d / "train.log").write_text(result["log"])
        remote = f"{VOL}/outputs/{run}/{label}"
        for path, text in read_files.remote([f"{remote}/{n}" for n in ("results.json", "eval_dev.json", "metrics.jsonl",
                                                                      "check_samples.jsonl")]).items():
            (d / Path(path).name).write_text(text)
        status = "ok" if (d / "results.json").exists() else f"FAILED exit {result['exit_code']}"
        print(f"{label}: {status} ({result['gpu']})", flush=True)
        if status != "ok":
            print(result["log"][-3000:])
    print(f"done: {out}")


@app.local_entrypoint()
def resume_test(gpu: str = "L40S"):
    """Interrupt-and-resume check for both trainers on 32 questions (8 steps over 2 epochs).

    Per trainer: A runs uninterrupted; B stops after 3 steps (--stop-after, as if preempted),
    then the same command without it resumes in the same directory. Prints each run's per-step
    batch fingerprints and the resume lines; the logs land in runs/modal-resume-test-<stamp>/.
    """
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:32],
           "validation": json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())[:16]}
    exact = ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", "discrete-exact", "--scoring", "single",
             "--regularization", "hinge", "--reward", "paper", "--passes", "2", "--minibatch", "4", "--lr", "4.01e-05",
             "--format-weight", "1.01", "--seed", "1", "--epochs", "2", "--checkpoint-every", "3"]
    ppo = ["subset.py", "train", "IDS_JSON", "--seed", "1", "--fast", "--no-kl", "--grading", "f1", "--hinge", "0.39",
           "--hinge-threshold", "0.9", "--ppo", "ppo_epochs=2", "--checkpoint-every", "3", "--",
           "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL, "--tokenizer_dir", MODEL,
           "--epochs", "2", "--lr", "2.27e-05", "--batchsize", "8", "--log_with", "tensorboard"]
    stop = lambda cmd: cmd[:cmd.index("--")] + ["--stop-after", "3"] + cmd[cmd.index("--"):] if "--" in cmd else cmd + ["--stop-after", "3"]
    run = "resume-test-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trainer = train_with_snapshots.with_options(gpu=gpu)
    first = [("exact-A", exact), ("exact-B", stop(exact)), ("ppo-A", ppo), ("ppo-B", stop(ppo))]
    results = {r["label"] + ("" if r["label"].endswith("A") else "1"): r for r in
               trainer.starmap([(ids, label, cmd, run) for label, cmd in first])}
    for r in trainer.starmap([(ids, label, cmd, run) for label, cmd in [("exact-B", exact), ("ppo-B", ppo)]]):
        results[r["label"] + "2"] = r
    out = root / "runs" / f"modal-{run}"
    out.mkdir(parents=True, exist_ok=True)
    names = ["metrics.jsonl", "steps.jsonl", "stability.jsonl", "stability_events.jsonl", "generations.jsonl", "run_meta.json"]
    for label in ["exact-A", "exact-B", "ppo-A", "ppo-B"]:
        files = read_files.remote([f"{VOL}/outputs/{run}/{label}/{n}" for n in names])
        (out / label).mkdir(exist_ok=True)
        for path, text in files.items():
            (out / label / Path(path).name).write_text(text)
    for key, r in sorted(results.items()):
        (out / f"{key}.log").write_text(r["log"])
        lines = [l for l in r["log"].splitlines() if l.startswith(("resumed", "checkpoint", "stopping", "STABILITY", "Traceback"))
                 or "Error" in l]
        print(f"== {key}: exit {r['exit_code']}", *lines[-12:], sep="\n  ")
    print(f"logs: {out}")


@app.local_entrypoint()
def frozen_test(gpu: str = "L40S"):
    """Short check of --frozen-answers: both trainers, then frozen_eval.py on their snapshots."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:96],
           "validation": json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())[:64]}
    exact = ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", "discrete-exact", "--scoring", "single", "--regularization",
             "hinge", "--reward", "paper", "--passes", "2", "--minibatch", "4", "--lr", "4.01e-05", "--format-weight", "1.01",
             "--seed", "1", "--epochs", "1", "--max-steps", "12", "--save-every", "12"]
    ppo = ["subset.py", "train", "IDS_JSON", "--seed", "1", "--fast", "--grading", "f1", "--frozen-answers", "--save-every", "12",
           "--max-steps", "12", "--", "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL,
           "--tokenizer_dir", MODEL, "--epochs", "1", "--lr", "1e-05", "--batchsize", "8", "--log_with", "tensorboard"]
    run = "frozen-test-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trainer = train_with_snapshots.with_options(gpu=gpu)
    jobs = [("exact-frozen", exact + ["--frozen-answers"]), ("ppo-frozen", ppo)]
    results = {r["label"]: r for r in trainer.starmap([(ids, label, cmd, run) for label, cmd in jobs])}
    evals = [(ids, f"eval-{label}", ["frozen_eval.py", "IDS_JSON", f"{VOL}/outputs/{run}/{label}/snapshot-step00012",
                                    "OUT_DIR/eval.json"], run) for label in ("exact-frozen", "ppo-frozen")]
    results.update({r["label"]: r for r in trainer.starmap(evals)})
    out = root / "runs" / f"modal-{run}"
    out.mkdir(parents=True, exist_ok=True)
    for label, r in sorted(results.items()):
        (out / f"{label}.log").write_text(r["log"])
        print(f"== {label}: exit {r['exit_code']}")
        print("\n".join(l for l in r["log"].splitlines()[-4:]))
    names = ["metrics.jsonl", "steps.jsonl", "stability.jsonl", "generations.jsonl", "eval_metrics.json"]
    for label in [j[0] for j in jobs] + ["eval-exact-frozen", "eval-ppo-frozen"]:
        for path, text in read_files.remote([f"{VOL}/outputs/{run}/{label}/{n}" for n in names]).items():
            (out / label).mkdir(exist_ok=True)
            (out / label / Path(path).name).write_text(text)
    print(f"logs: {out}")


@app.local_entrypoint()
def frozen_round(search_dir: str, round: str = "frozen_compare", gpu: str = "L40S", eval_run: str = "",
                 test: bool = True):
    """Train {round}_configs.json (commands with --frozen-answers) on the search's train/dev split and
    score every snapshot on dev and step 256 on test with frozen_eval.py (base answer, adapted
    confidence). Writes {round}/curve.json ({label: {step: dev metrics}}) and curve_test.json.
    --eval-run RUN re-scores the snapshots of an earlier run of this entrypoint without training;
    --no-test scores dev only (sweeps select on dev and leave test alone)."""
    root = Path(__file__).resolve().parents[1]
    search = Path(search_dir).resolve()
    configs = json.loads((search / f"{round}_configs.json").read_text())
    data = root / "runs/pilot-20261001T002945Z/data"
    train = [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()]
    dev = {"train": train, "validation": json.loads((search / "dev_ids.json").read_text())}
    test = {"train": train, "validation": [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()]}
    run = eval_run or f"{round}-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trainer = train_with_snapshots.with_options(gpu=gpu)
    trained = list_snapshots.remote(run, list(configs)) if eval_run else {}
    for r in [] if eval_run else trainer.starmap([(dev, label, cfg["command"], run) for label, cfg in configs.items()],
                             return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception) or r["exit_code"]:
            print("TRAIN FAILED", r if isinstance(r, Exception) else (r["label"], r["log"][-1500:]))
            continue
        trained[r["label"]] = r["snapshots"]
        print(f"trained {r['label']}: {[Path(s).name for s in r['snapshots']]}", flush=True)
    jobs = []
    for label, snapshots in trained.items():
        for snapshot in snapshots:
            step = int(Path(snapshot).name.split("step")[1])
            jobs.append((dev, f"eval-{label}-dev-{step}", ["frozen_eval.py", "IDS_JSON", snapshot, "OUT_DIR/eval.json"], run))
            if test and step == max(int(Path(s).name.split("step")[1]) for s in snapshots):
                jobs.append((test, f"eval-{label}-test-{step}", ["frozen_eval.py", "IDS_JSON", snapshot, "OUT_DIR/eval.json"], run))
    curve, curve_test = {}, {}
    for r in trainer.starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception) or r["exit_code"]:
            print("EVAL FAILED", r if isinstance(r, Exception) else (r["label"], r["log"][-1500:]))
            continue
        _, rest = r["label"].split("eval-", 1)
        label, split, step = rest.rsplit("-", 2)
        metrics = json.loads(read_files.remote([f"{VOL}/outputs/{run}/{r['label']}/eval_metrics.json"]).popitem()[1])
        (curve if split == "dev" else curve_test).setdefault(label, {})[step] = metrics
    out = search / round
    out.mkdir(exist_ok=True)
    (out / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")
    (out / "curve_test.json").write_text(json.dumps({k: v[max(v, key=int)] for k, v in curve_test.items()}, indent=1) + "\n")
    (out / "run.txt").write_text(f"{VOL}/outputs/{run}\n")
    print(f"done: {out}")


@app.function(volumes={VOL: volume}, timeout=600)
def list_snapshots(run: str, labels: list) -> dict:
    """{label: [snapshot dirs]} for a run's training outputs on the volume."""
    volume.reload()
    return {label: sorted(str(p) for p in Path(f"{VOL}/outputs/{run}/{label}").glob("snapshot-step*")) for label in labels
            if Path(f"{VOL}/outputs/{run}/{label}").exists()}


@app.local_entrypoint()
def qwen_check(gpu: str = "L40S"):
    """Port check for Qwen-2.5-3B: level scoring vs teacher forcing (both models), the base model's
    dev scores with the frozen evaluator, and 12-step exact + KL and PPO + KL runs with frozen answers."""
    root = Path(__file__).resolve().parents[1]
    # The pre-quantized unsloth/Qwen2.5-3B-Instruct-bnb-4bit needs a newer Unsloth than the pinned one;
    # the 16-bit checkpoint loads and is quantized to 4-bit on load.
    qwen = "unsloth/Qwen2.5-3B-Instruct"
    data = root / "runs/pilot-20261001T002945Z/data"
    train = [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()]
    ids = {"train": train[:96], "validation": json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())}
    run = "qwen-check-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = [
        ("verify", ["verify_levels.py", qwen, MODEL]),
        ("base-dev", ["frozen_eval.py", "IDS_JSON", qwen, "OUT_DIR/eval.json"]),
        ("exact-kl", ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--model", qwen, "--mode", "discrete-exact", "--scoring", "single",
                      "--regularization", "baseline", "--reward", "released", "--passes", "4", "--minibatch", "4", "--lr", "1e-05",
                      "--seed", "1", "--epochs", "1", "--max-steps", "12", "--save-every", "12", "--frozen-answers"]),
        ("ppo-kl", ["subset.py", "train", "IDS_JSON", "--seed", "1", "--fast", "--grading", "f1", "--frozen-answers",
                    "--save-every", "12", "--max-steps", "12", "--", "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth",
                    "--model_dir", qwen, "--tokenizer_dir", qwen, "--epochs", "1", "--lr", "1e-05", "--batchsize", "8",
                    "--log_with", "tensorboard"]),
    ]
    out = root / "runs" / f"modal-{run}"
    out.mkdir(parents=True, exist_ok=True)
    for r in train_with_snapshots.with_options(gpu=gpu).starmap([(ids, label, cmd, run) for label, cmd in jobs],
                                                                return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:1500]); continue
        (out / f"{r['label']}.log").write_text(r["log"])
        print(f"== {r['label']}: exit {r['exit_code']}")
        print("\n".join([l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l][-3:])[:1500])
    for path, text in read_files.remote([f"{VOL}/outputs/{run}/base-dev/eval_metrics.json",
                                         f"{VOL}/outputs/{run}/exact-kl/metrics.jsonl", f"{VOL}/outputs/{run}/ppo-kl/stability.jsonl"]).items():
        (out / path.split("/")[-2]).mkdir(exist_ok=True)
        (out / path.split("/")[-2] / Path(path).name).write_text(text)
    print(f"logs: {out}")


@app.local_entrypoint()
def try_load(names: str, gpu: str = "L40S"):
    """Which model names the pinned Unsloth loads in 4-bit (comma-separated)."""
    root = Path(__file__).resolve().parents[1]
    ids = {"train": [], "validation": []}
    run = "try-load-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    code = ("import sys; from unsloth import FastLanguageModel; "
            "m, t = FastLanguageModel.from_pretrained(model_name=sys.argv[1], max_seq_length=1048, dtype=None, load_in_4bit=True); "
            "print('LOADED', sys.argv[1], type(m).__name__, t.eos_token)")
    jobs = [(ids, f"load-{i}", ["-c", code, name], run) for i, name in enumerate(names.split(","))]
    for r in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:500]); continue
        print(r["label"], "exit", r["exit_code"], [l[:300] for l in r["log"].splitlines() if "LOADED" in l or "Error" in l][-2:])


@app.local_entrypoint()
def verify_levels(gpu: str = "L40S", debug: bool = False):
    """LevelScheme vs teacher forcing on Qwen-2.5-3B and Llama-3-8B (verify_levels.py)."""
    run = "verify-levels-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({"train": [], "validation": []}, "verify",
        ["debug_levels.py", "unsloth/Qwen2.5-3B-Instruct"] if debug else ["verify_levels.py", "unsloth/Qwen2.5-3B-Instruct", MODEL], run)
    print("exit", r["exit_code"]); print("\n".join(l for l in r["log"].splitlines() if l.startswith(("{", "first", "P(", "batching")) or l[:2].strip().isdigit() or "Error" in l))


@app.local_entrypoint()
def minimal_diff(train_limit: int = 1024, eval_limit: int = 0, seeds: str = "1,2,3", epochs: int = 2, gpu: str = "L40S",
                 model: str = "unsloth/Qwen2.5-3B-Instruct", tag: str = "", objectives: str = "exact,ppo"):
    """The released Train.py with patches/exact_confidence.patch: --objective exact vs ppo, otherwise as
    released (exact-match reward grading, adaptive KL, ppo_epochs 4, lr 1e-5, batch 8), then the released
    inference + metrics on the 512 test questions (subset.py evaluate) for each final epoch adapter."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    train = [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:train_limit]
    test = [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()][:eval_limit or None]
    ids = {"train": train, "validation": test}
    run = f"minimal-diff{tag}-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = []
    for objective in objectives.split(","):
        for s in seeds.split(","):
            jobs.append((ids, f"{objective}-s{s}", ["subset.py", "train", "IDS_JSON", "--seed", s, "--",
                         "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", model, "--tokenizer_dir", model,
                         "--epochs", str(epochs), "--lr", "1e-05", "--batchsize", "8", "--log_with", "tensorboard", "--objective", objective], run))
    out = root / "runs" / f"modal-{run}"
    out.mkdir(parents=True, exist_ok=True)
    trained = []
    for r in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("TRAIN FAILED", repr(r)[:800]); continue
        (out / f"{r['label']}.log").write_text(r["log"])
        print(f"trained {r['label']}: exit {r['exit_code']}", flush=True)
        if r["exit_code"] == 0:
            trained.append(r["label"])
    results = {}
    dirs = [f"{VOL}/outputs/{run}/{label}/model_finetuned_epoch{epochs}" for label in trained]
    for r in evaluate_test.with_options(gpu=gpu).starmap([({"validation": test}, d) for d in dirs], return_exceptions=True):
        if isinstance(r, Exception) or "error" in r:
            print("EVAL FAILED", r if isinstance(r, Exception) else r["error"][-800:]); continue
        results[r["model_dir"].split("/")[-2]] = r["metrics"]
    (out / "test_metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    for label, m in sorted(results.items()):
        print(label, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
    print(f"done: {out}")


@app.local_entrypoint()
def verify_fast(gpu: str = "L40S"):
    """exact_fast vs exact level scoring (verify_fast_levels.py) on Qwen-2.5-3B and Llama-3-8B."""
    run = "verify-fast-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({"train": [], "validation": []}, "verify",
        ["verify_fast_levels.py", "unsloth/Qwen2.5-3B-Instruct", MODEL], run)
    print("exit", r["exit_code"]); print("\n".join(l for l in r["log"].splitlines() if " levels fast" in l or l.startswith(("RESULT", "Traceback")) or "Error" in l))


@app.local_entrypoint()
def exact_hooks_test(gpu: str = "L40S"):
    """--objective exact_fast through subset.py: snapshots, checkpoints, logs, then stop at step 3 and resume."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:32],
           "validation": json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())[:16]}
    q = "unsloth/Qwen2.5-3B-Instruct"
    cmd = ["subset.py", "train", "IDS_JSON", "--seed", "1", "--save-every", "2", "--checkpoint-every", "2", "--", "--out_dir", "OUT_DIR",
           "--dataset", "triviaqa", "--is_unsloth", "--model_dir", q, "--tokenizer_dir", q, "--epochs", "2", "--lr", "1e-05",
           "--batchsize", "8", "--log_with", "tensorboard", "--objective", "exact_fast"]
    stopped = cmd[:cmd.index("--")] + ["--stop-after", "3"] + cmd[cmd.index("--"):]
    run = "exact-hooks-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trainer = train_with_snapshots.with_options(gpu=gpu)
    for label_cmd in (stopped, cmd):
        r = trainer.remote(ids, "exact", label_cmd, run)
        print("== exit", r["exit_code"], "snapshots", [Path(s).name for s in r["snapshots"]])
        print("\n".join(l for l in r["log"].splitlines() if l.startswith(("resumed", "checkpoint", "stopping", "exact step", "STABILITY", "Traceback")) or "Error" in l)[-2500:])
    files = read_files.remote([f"{VOL}/outputs/{run}/exact/{n}" for n in ("steps.jsonl", "stability.jsonl", "status.json")])
    for path, text in files.items():
        print("--", Path(path).name); print(text[-1200:])


@app.local_entrypoint()
def eval_one(model_dir: str, split: str = "dev", gpu: str = "L40S"):
    """The released evaluation of one adapter directory on the volume, on the 512 dev or 512 test questions."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = (json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text()) if split == "dev"
           else [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()])
    r = evaluate_test.with_options(gpu=gpu).remote({"validation": ids}, model_dir)
    print(json.dumps(r.get("metrics") or r.get("error", "")[-2000:], indent=1))


FAST_CACHE = f"{VOL}/fast/qwen25-3b-cache.pt"


@app.local_entrypoint()
def fast_cache(n_train: int = 8000, gpu: str = "L40S", model: str = "unsloth/Qwen2.5-3B-Instruct", out: str = FAST_CACHE):
    """Build the fast loop's rollout cache (fast_loop.py cache): base-model answers for n_train random
    training questions and the 512 dev questions, graded, with reference level log-probs. Llama-3-8B:
    --model unsloth/llama-3-8b-Instruct-bnb-4bit --out /vol/fast/llama3-8b-cache.pt (LLAMA_CACHE); the 16-bit
    repo is killed for host memory while Unsloth quantizes it on load. minimal_ref --model then switches to bf16."""
    root = Path(__file__).resolve().parents[1]
    ids = {"train": "all", "validation": json.loads((root / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())}
    r = train_with_snapshots.with_options(gpu=gpu).remote(ids, "cache", ["fast_loop.py", "cache", "IDS_JSON", out,
                                                                          "--n-train", str(n_train), "--model", model],
                                                          "fast-cache-" + Path(out).stem)
    print("exit", r["exit_code"]); print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def fast_train(configs: str, gpu: str = "L40S"):
    """Run fast_loop.py train for each {label: [args]} in a JSON file, in parallel; curves to runs/fast/<label>/."""
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(Path(configs).read_text())
    run = "fast-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = [({}, label, ["fast_loop.py", "train", FAST_CACHE, "OUT_DIR", *args], run) for label, args in cfg.items()]
    for r in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:800]); continue
        out = root / "runs" / "fast" / r["label"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.log").write_text(r["log"])
        for path, text in read_files.remote([f"{VOL}/outputs/{run}/{r['label']}/{n}" for n in ("curve.json", "metrics.jsonl")]).items():
            (out / Path(path).name).write_text(text)
        print(f"== {r['label']} exit {r['exit_code']}")
        print("\n".join(l for l in r["log"].splitlines() if l.startswith('{"step"'))[-1500:])


@app.local_entrypoint()
def fast_profile(gpu: str = "L40S"):
    """fast_loop.py profile: where a training update's time goes."""
    run = "fast-profile-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "profile", ["fast_loop.py", "profile", FAST_CACHE], run)
    print("exit", r["exit_code"], r["gpu"]); print(r["log"][-12000:])


@app.local_entrypoint()
def fast_profile_lora(gpu: str = "L40S"):
    """fast_loop.py profile-lora: time and memory with adapters trained in only some layers."""
    run = "fast-profile-lora-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "profile-lora", ["fast_loop.py", "profile-lora", FAST_CACHE], run)
    print("exit", r["exit_code"], r["gpu"])
    print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def attn_bias_check(gpu: str = "L40S"):
    """attn_bias_check.py: attention-output biases through Unsloth training, decoding and disable_adapter."""
    run = "attn-bias-check-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "check", ["attn_bias_check.py"], run)
    print("exit", r["exit_code"]); print(r["log"][-4000:])


@app.local_entrypoint()
def attn_bias_smoke(gpu: str = "L40S", steps: int = 20, lr: str = "2e-4", frozen: bool = False):
    """subset.py train --attn-bias 18-35 for a few steps (patched released code, --objective exact), then
    the released evaluation of the last snapshot: biases train alone, are saved and are loaded."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    train = [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:256]
    test = [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()][:64]
    run = "attn-bias-smoke-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = "unsloth/Qwen2.5-3B-Instruct"
    r = train_with_snapshots.with_options(gpu=gpu).remote({"train": train, "validation": test}, "smoke", [
        "subset.py", "train", "IDS_JSON", "--seed", "1", "--save-every", str(steps // 2), "--max-steps", str(steps),
        "--attn-bias", "18-35", *(["--frozen-answers"] if frozen else []), "--", "--out_dir", "OUT_DIR", "--dataset", "triviaqa",
        "--is_unsloth", "--model_dir", model,
        "--tokenizer_dir", model, "--epochs", "1", "--lr", lr, "--batchsize", "8", "--log_with", "tensorboard", "--objective", "exact"], run)
    print("exit", r["exit_code"], "snapshots", r["snapshots"])
    print("\n".join(l for l in r["log"].splitlines() if "attention-output" in l or "Error" in l or "Traceback" in l))
    print("steps:", "\n".join(r["timing"].splitlines()[:3] + r["timing"].splitlines()[-2:]))
    if r["snapshots"]:
        snap = r["snapshots"][-1]
        print("snapshot:", snap)
        e = evaluate_test.with_options(gpu=gpu).remote({"validation": test}, snap, frozen)
        print("eval:", e.get("error") or {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e["metrics"].items()})


@app.function(volumes={VOL: volume}, timeout=12 * HOUR, nonpreemptible=True)
def train_then_eval(ids_by_split: dict, label: str, command: list[str], run_name: str, gpu: str = "L40S",
                    frozen: bool = False) -> dict:
    """Train (train_with_snapshots), then the released evaluation of every snapshot on the validation
    split (evaluate_test, in parallel); writes {step: metrics} to the run's curve.json on the volume.
    Runs remotely, so a long run does not depend on the local client staying connected.

    Non-preemptible: a preempted orchestrator is restarted and calls training again, which cancels the
    running training call (it then resumes from its last committed checkpoint). If it is restarted
    anyway, a run whose status.json says completed is not trained again."""
    volume.reload()
    out_dir = Path(f"{VOL}/outputs/{run_name}/{label}")
    status = json.loads((out_dir / "status.json").read_text()) if (out_dir / "status.json").exists() else {}
    if status.get("status") == "completed":
        r = dict(exit_code=0, snapshots=sorted(str(p) for p in out_dir.iterdir() if p.name.startswith("snapshot-step")))
    else:
        r = train_with_snapshots.with_options(gpu=gpu).remote(ids_by_split, label, command, run_name)
    curve = {}
    jobs = [({"validation": ids_by_split["validation"]}, snap, frozen) for snap in r["snapshots"]]
    for e in evaluate_test.with_options(gpu=gpu).starmap(jobs, return_exceptions=True):
        if isinstance(e, Exception) or "error" in e:
            print("EVAL FAILED", repr(e)[:500] if isinstance(e, Exception) else e["error"][-500:])
            continue
        curve[int(e["model_dir"].split("snapshot-step")[1])] = e["metrics"]
    volume.reload()
    Path(f"{VOL}/outputs/{run_name}/{label}/curve.json").write_text(json.dumps(dict(sorted(curve.items())), indent=1))
    volume.commit()
    return dict(label=label, exit_code=r["exit_code"], snapshots=r["snapshots"], curve=curve)


@app.local_entrypoint()
def attn_bias_full(lrs: str = "1e-4,2e-4", layers: str = "18-35", steps: int = 3000, gpu: str = "L40S",
                   frozen_answers: bool = True):
    """Released code + patch, --objective exact, answers generated by the policy, training only
    attention-output biases (subset.py --attn-bias) on the full training set, as the KL-grid runs
    (seed 1, batch 8, 2 epochs capped at `steps`, snapshots every 500, adaptive KL as released);
    then every snapshot on the same 512 dev questions. Spawned remotely: run with `modal run --detach`
    and read outputs/<run>/<label>/curve.json from the volume.

    frozen_answers (default): answers come from the base model, with LoRA and biases off, in training
    (--frozen-answers) and in evaluation (frozen_eval.py). Without it, a policy-generated run lost answer
    accuracy (0.315 at step 500 against about 0.40 for LoRA runs), since the biases act on every token
    but only the confidence is trained."""
    root = Path(__file__).resolve().parents[1]
    ids = json.loads((root / "runs/runpod/klgrid-exact-adapt20/ids.json").read_text())  # {"train": "all", "validation": 512 dev ids}
    run = "attn-bias-full-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = "unsloth/Qwen2.5-3B-Instruct"
    calls = []
    for lr in lrs.split(","):
        label = f"attnbias{layers}-lr{lr}{'-frozen' if frozen_answers else ''}-s1"
        command = ["subset.py", "train", "IDS_JSON", "--seed", "1", "--save-every", "500", "--checkpoint-every", "100",
                   "--max-steps", str(steps), "--attn-bias", layers, *(["--frozen-answers"] if frozen_answers else []),
                   "--", "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", model, "--tokenizer_dir", model, "--epochs", "2", "--lr", lr,
                   "--batchsize", "8", "--log_with", "tensorboard", "--objective", "exact"]
        calls.append(train_then_eval.spawn(ids, label, command, run, gpu, frozen_answers))
        print(f"spawned {run}/{label}")
    for c in calls:
        print("call id", c.object_id)


@app.local_entrypoint()
def restart_check(gpu: str = "L40S"):
    """Resume after a simulated preemption: train --stop-after 12 (checkpoints every 5 steps, committed
    on the trainer's RD_SAVED lines), then the same command again, which must resume and finish at
    step 20. Also shows the exact objective's per-step answer accuracy in stability.jsonl."""
    root = Path(__file__).resolve().parents[1]
    data = root / "runs/pilot-20261001T002945Z/data"
    ids = {"train": [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()][:256],
           "validation": [json.loads(l)["id"] for l in (data / "eval.jsonl").read_text().splitlines()][:64]}
    run = "restart-check-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = "unsloth/Qwen2.5-3B-Instruct"
    base = ["subset.py", "train", "IDS_JSON", "--seed", "1", "--save-every", "10", "--checkpoint-every", "5", "--max-steps", "20",
            "--attn-bias", "18-35"]
    tail = ["--", "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", model, "--tokenizer_dir", model,
            "--epochs", "1", "--lr", "2e-4", "--batchsize", "8", "--log_with", "tensorboard", "--objective", "exact"]
    first = train_with_snapshots.with_options(gpu=gpu).remote(ids, "run", base + ["--stop-after", "12"] + tail, run)
    second = train_with_snapshots.with_options(gpu=gpu).remote(ids, "run", base + tail, run)
    for name, r in (("first", first), ("second", second)):
        print(name, "exit", r["exit_code"], "snapshots", [p.split("/")[-1] for p in r["snapshots"]])
    print("\n".join(l for l in second["log"].splitlines() if l.startswith("===") or "resumed" in l or l.startswith("RD_SAVED")))
    files = read_files.remote([f"{VOL}/outputs/{run}/run/stability.jsonl", f"{VOL}/outputs/{run}/run/status.json"])
    rows = [json.loads(l) for l in files.get(f"{VOL}/outputs/{run}/run/stability.jsonl", "").splitlines()]
    print("stability steps", [r["step"] for r in rows])
    print("batch accuracy", [r.get("batch_accuracy") for r in rows])
    print("status", files.get(f"{VOL}/outputs/{run}/run/status.json", "missing"))


@app.local_entrypoint()
def fast_eval(dirs: str, gpu: str = "L40S"):
    """fast_loop.py eval for comma-separated adapter dirs on the volume; prints one JSON line per dir."""
    run = "fast-eval-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs = [({}, f"e{i}", ["fast_loop.py", "eval", FAST_CACHE, d, "OUT_DIR/fast_eval.json"], run) for i, d in enumerate(dirs.split(","))]
    names = dirs.split(",")
    for r in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:500]); continue
        line = next((l for l in r["log"].splitlines() if l.startswith("{")), r["log"][-800:])
        print("RESULT", names[int(r["label"][1:])], line)


MINIMAL_CACHE = f"{VOL}/fast/qwen25-3b-cache-bf16ref.pt"
TOPK_CACHE = f"{VOL}/fast/qwen25-3b-cache-bf16ref-top64.pt"  # MINIMAL_CACHE + answer reference top-64 (answer-ref)
LLAMA_CACHE = f"{VOL}/fast/llama3-8b-cache.pt"  # fast_cache --model unsloth/llama-3-8b-Instruct
LLAMA_TOPK_CACHE = f"{VOL}/fast/llama3-8b-cache-bf16ref-top64.pt"  # minimal_ref, then minimal_answer_ref, of LLAMA_CACHE
LLAMA_GT = f"{VOL}/fast/llama3-8b-gt.json"
GT = f"{VOL}/fast/gt.json"  # {question_id: gt_candidates} of every cached row (fast_loop.py gt)


@app.local_entrypoint()
def fast_gt(cache: str = FAST_CACHE, out: str = GT):
    """fast_loop.py gt: the cached rows' gold answers (train and dev) for minimal_trainer.py --gt."""
    # a GPU only because fast_loop.py imports Unsloth, which needs one at import
    r = train_with_snapshots.with_options(gpu="T4").remote({}, "gt", ["fast_loop.py", "gt", cache, out], "fast-gt-" + Path(out).stem)
    print("exit", r["exit_code"]); print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.function(timeout=HOUR)
def run_tests(files: dict, target: str) -> str:
    """pytest in the image (transformers / peft pins) on test files sent from the local checkout."""
    import os
    import subprocess
    root = Path("/tmp/rd_tests")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    subprocess.run(["python", "-m", "pip", "install", "-q", "pytest"], capture_output=True)
    proc = subprocess.run(["python", "-m", "pytest", "-q", str(root / target)], cwd=CODE, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": f"{CODE}:/opt/rd"})
    return proc.stdout[-6000:] + proc.stderr[-3000:]


@app.local_entrypoint()
def minimal_tests():
    """tests/test_minimal_trainer.py in the Modal image (CPU), against the image's minimal_trainer.py."""
    root = Path(__file__).resolve().parents[1]
    print(run_tests.remote({"tests/test_minimal_trainer.py": (root / "tests/test_minimal_trainer.py").read_text()},
                           "tests/test_minimal_trainer.py"))


@app.local_entrypoint()
def minimal_ref(gpu: str = "L40S", source: str = FAST_CACHE, out: str = MINIMAL_CACHE, model: str = ""):
    """minimal_trainer.py ref: the fast cache with reference levels recomputed by the bf16 model, the
    untrained bf16 dev metrics, and the shared-prefix check against a full forward; log to runs/minimal/ref/."""
    run = "minimal-ref-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "ref", ["minimal_trainer.py", "ref", source, out,
                                                                   *(["--model", model] if model else [])], run)
    print("exit", r["exit_code"], r["gpu"])
    log_dir = "ref" if out == MINIMAL_CACHE else "ref-" + Path(out).stem
    out = Path(__file__).resolve().parents[1] / "runs" / "minimal" / log_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.log").write_text(r["log"])
    print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def minimal_answer_ref(gpu: str = "L40S", k: int = 64, source: str = MINIMAL_CACHE, out: str = TOPK_CACHE):
    """minimal_trainer.py answer-ref: TOPK_CACHE, the bf16 cache plus the reference's top-k log-probs at every
    cached answer token (for --answer-kl without a reference pass)."""
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "answer-ref", ["minimal_trainer.py", "answer-ref", source,
                                                                             out, "--k", str(k)], "minimal-answer-ref-" + Path(out).stem)
    print("exit", r["exit_code"], r["gpu"]); print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def minimal_restart_check(gpu: str = "L40S"):
    """minimal_trainer.py resume: online training stopped by --stop-after 12 (checkpoints every 5 steps), then
    the same command again, which must resume at step 12 and complete at step 20 with each step logged once."""
    run = "minimal-restart-check-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = ["minimal_trainer.py", "train", TOPK_CACHE, "OUT_DIR", "--steps", "20", "--eval-every", "10", "--checkpoint-every", "5",
               "--online", "--gt", GT, "--lora-modules", "none", "--attn-bias", "18-35", "--lr", "1e-2", "--answer-kl", "0.1"]
    first = train_with_snapshots.with_options(gpu=gpu).remote({}, "run", command + ["--stop-after", "12"], run)
    second = train_with_snapshots.with_options(gpu=gpu).remote({}, "run", command, run)
    print("first exit", first["exit_code"], "second exit", second["exit_code"])
    print("\n".join(l for l in second["log"].splitlines() if l.startswith(("===", "resumed", "stopping", "RD_SAVED")) or "Error" in l))
    files = read_files.remote([f"{VOL}/outputs/{run}/run/{n}" for n in ("metrics.jsonl", "status.json", "curve.json")])
    steps = [json.loads(l)["step"] for l in files.get(f"{VOL}/outputs/{run}/run/metrics.jsonl", "").splitlines()]
    print("logged steps", steps, "unique and complete:", steps == list(range(1, 21)))
    print("curve steps", sorted(json.loads(files.get(f"{VOL}/outputs/{run}/run/curve.json", "{}"))))
    print("status", files.get(f"{VOL}/outputs/{run}/run/status.json", "missing").replace("\n", " ")[:120])


@app.local_entrypoint()
def minimal_diagnose(rows: int = 64, gpu: str = "L40S"):
    """minimal_trainer.py diagnose on the fast cache (4-bit reference): shared prefix vs full forward vs
    one row at a time, bf16 and fp32."""
    run = "minimal-diagnose-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "diagnose", ["minimal_trainer.py", "diagnose", FAST_CACHE, "--rows", str(rows)], run)
    print("exit", r["exit_code"], r["gpu"])
    print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def minimal_memory(gpu: str = "L40S"):
    """minimal_trainer.py memory: peak memory of one all-layer LoRA update, with and without the shared prefix."""
    run = "minimal-memory-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = train_with_snapshots.with_options(gpu=gpu).remote({}, "memory", ["minimal_trainer.py", "memory", MINIMAL_CACHE], run)
    print("exit", r["exit_code"], r["gpu"])
    print("\n".join(l for l in r["log"].splitlines() if l.startswith("{") or "Error" in l))


@app.local_entrypoint()
def minimal_train(configs: str, gpu: str = "L40S", fetch: str = "", cache: str = MINIMAL_CACHE):
    """minimal_trainer.py train for each {label: [args]} in a JSON file, in parallel, on the bf16-reference
    cache; train.log, metrics.jsonl and curve.json to runs/minimal/<label>/. With --fetch RUN (same configs),
    only copies a finished run's outputs from the volume, for when the local client was cut off."""
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(Path(configs).read_text())
    run = fetch or "minimal-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("run", run, flush=True)

    def save(label):
        out = root / "runs" / "minimal" / label
        out.mkdir(parents=True, exist_ok=True)
        files = read_files.remote([f"{VOL}/outputs/{run}/{label}/{n}" for n in ("train.log", "curve.json", "metrics.jsonl")])
        for path, text in files.items():
            (out / Path(path).name).write_text(text)
        log = files.get(f"{VOL}/outputs/{run}/{label}/train.log", "")
        print(f"== {label}")
        print("\n".join(l for l in log.splitlines() if l.startswith('{"step"') or "Error" in l)[-1500:])
    if fetch:
        for label in cfg:
            save(label)
        return
    jobs = [({}, label, ["minimal_trainer.py", "train", cache, "OUT_DIR", *args], run) for label, args in cfg.items()]
    for r in train_with_snapshots.with_options(gpu=gpu).starmap(jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:800]); continue
        print(f"exit {r['exit_code']} on {r['gpu']}")
        save(r["label"])

