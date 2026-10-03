"""Launch one training job on a RunPod pod (runpod/bootstrap.sh -> runpod/entry.sh).

    python scripts/runpod_launch.py NAME --train-cmd "python exact_llama.py IDS_JSON OUT_DIR ..." \\
        [--ids-train-limit 1024 | --ids-train-all] [--post-cmd ...] [--interruptible] [--dry-run]
    python scripts/runpod_launch.py setup --setup-only     # install the environment on the volume first

The pod runs ghcr.io/rrenaud/rewarding-doubt (runpod/Dockerfile, built by GitHub Actions), writes to
/workspace/runs/NAME on the network volume and terminates itself at a final outcome; results stay
on the volume (fetch them with scripts/runpod_sync.py). --stock-image instead runs python:3.11 with
our code packed into an env var and the environment installed once on the volume
(runpod/bootstrap.sh), for when the image cannot be used. Question IDs: the
training subset (first N of runs/pilot-.../train.jsonl, or the whole split) and the search's
512 dev questions as the validation split.

Secrets passed into the pod's environment: the RunPod API key, so the pod can terminate itself, and
the W&B key from ~/.wandb_rewarding_doubt_key.txt if that file exists (live charts; --no-wandb to skip).
"""
import argparse
import base64
import io
import json
import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
VOLUME_ID = "591nr7jbh1"  # "rewarding-doubt", 75 GB, EU-RO-1
CODE = ["modal_repro/subset.py", "modal_repro/exact_llama.py", "modal_repro/shared_prefix.py",
        "modal_repro/thinking_llama.py", "src/rewarding_doubt", "runpod"]
BOOT = 'mkdir -p /opt/code && echo "$CODE_B64" | base64 -d | tar xz -C /opt/code && exec bash /opt/code/runpod/bootstrap.sh'


def api_key():
    return os.environ.get("RUNPOD_API_KEY") or (Path.home() / ".runpod_rewarding_doubt_key.txt").read_text().strip()


def call(method, path, body=None):
    request = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                                              "User-Agent": "rewarding-doubt-launcher/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path}: HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}")


def code_b64():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in CODE:
            tar.add(ROOT / path, arcname=path, filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buffer.getvalue()).decode()


def ids(args):
    data = ROOT / "runs/pilot-20261001T002945Z/data"
    train = [json.loads(l)["id"] for l in (data / "train.jsonl").read_text().splitlines()]
    return {"train": "all" if args.ids_train_all else train[:args.ids_train_limit],
            "validation": json.loads((ROOT / "runs/hparam-search-20261002T044831Z/dev_ids.json").read_text())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--train-cmd", help="trainer command run from SingleAnswerSetting/; OUT_DIR, IDS_JSON substituted")
    parser.add_argument("--post-cmd", default="")
    parser.add_argument("--ids-train-limit", type=int, default=1024)
    parser.add_argument("--ids-train-all", action="store_true")
    parser.add_argument("--gpu", action="append", default=None, help="RunPod GPU type id (repeatable, in order)")
    parser.add_argument("--interruptible", action="store_true", help="spot pod (RunPod stopped offering these in 2026; kept for if they return)")
    parser.add_argument("--community", action="store_true", help="Community Cloud instead of Secure Cloud")
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--stall-minutes", type=int, default=45, help="first start downloads ~40 GB of model and data")
    parser.add_argument("--on-exit", default="terminate", choices=["terminate", "stop", "none"])
    parser.add_argument("--image", default="ghcr.io/rrenaud/rewarding-doubt:latest")
    parser.add_argument("--stock-image", action="store_true", help="python:3.11 + bootstrap.sh instead of --image")
    parser.add_argument("--expt", help="experiment id stamped on every generations.jsonl line (default: NAME up to its first '-')")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--force-resume", action="store_true",
                        help="rerun NAME even if it already ended (crashed, hung, ...): resumes from its checkpoint")
    parser.add_argument("--setup-only", action="store_true", help="with --stock-image: install the environment only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.setup_only and not args.train_cmd:
        parser.error("--train-cmd is required unless --setup-only")
    if args.setup_only and not args.stock_image:
        parser.error("--setup-only only applies to --stock-image")
    env = dict(RUNPOD_API_KEY=api_key(), RD_EXPT=args.expt or args.name.split("-")[0], ON_EXIT=args.on_exit, RUN_DIR=f"/workspace/runs/{args.name}",
               MAX_HOURS=str(int(args.max_hours)), STALL_MINUTES=str(args.stall_minutes))
    if args.setup_only:
        env["SETUP_ONLY"] = "1"
    else:
        env.update(TRAIN_CMD=args.train_cmd, POST_CMD=args.post_cmd,
                   IDS_JSON_B64=base64.b64encode(json.dumps(ids(args)).encode()).decode())
    if args.stock_image:
        env["CODE_B64"] = code_b64()
    if args.force_resume:
        env["FORCE_RESUME"] = "1"
    wandb_key = Path.home() / ".wandb_rewarding_doubt_key.txt"
    if wandb_key.exists() and not args.no_wandb:
        env.update(WANDB_API_KEY=wandb_key.read_text().strip(), WANDB_MODE="online", WANDB_PROJECT="rewarding-doubt",
                   WANDB_ENTITY="multi-tokenizer")
    volume = next(v for v in call("GET", "/networkvolumes") if v["id"] == VOLUME_ID)
    body = dict(name=f"rd-{args.name}", imageName="python:3.11-bookworm" if args.stock_image else args.image,
                dockerStartCmd=["bash", "-c", BOOT] if args.stock_image else [],
                gpuTypeIds=args.gpu or ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX A6000"],
                gpuTypePriority="custom", gpuCount=1, cloudType="COMMUNITY" if args.community else "SECURE",
                interruptible=args.interruptible, networkVolumeId=VOLUME_ID, volumeMountPath="/workspace",
                dataCenterIds=[volume["dataCenterId"]], containerDiskInGb=30, minRAMPerGPU=24, ports=[], env=env)
    if args.dry_run:
        shown = {k: (v if k != "env" else {e: (x if e in ("ON_EXIT", "RUN_DIR", "MAX_HOURS", "STALL_MINUTES", "TRAIN_CMD",
                                                          "POST_CMD", "SETUP_ONLY", "UPLOAD_CMD", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "RD_EXPT", "FORCE_RESUME") else f"<{len(x)} chars>")
                                           for e, x in v.items()}) for k, v in body.items()}
        print(json.dumps(shown, indent=1))
        return
    pod = call("POST", "/pods", body)
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "desiredStatus", "interruptible", "gpu")}, indent=1))


if __name__ == "__main__":
    main()
