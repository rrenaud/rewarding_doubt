"""Promote a Modal run of minimal_trainer.py to RunPod: the same command, on an image built from committed code, with
the data it reads copied to the network volume.

    python scripts/promote_to_runpod.py LABEL [--steps N] [--name NAME] [--gpu TYPE ...] [--max-hours H] [--dry-run]

LABEL is a minimal_train label; its launch record (runs/minimal/LABEL/launch.json, written by modal_repro/app.py)
holds the command, the git commit and whether its code was committed. Steps:

1. Code: refuses a run launched from uncommitted code (--allow-dirty-source overrides), or a local checkout with
   uncommitted code or commits not on origin/main; reports code changes between the run's commit and HEAD.
2. Image: the newest successful GitHub Actions image build at an ancestor of HEAD with no changes since to the
   files the image contains (the workflow's paths); waits for one in progress at HEAD.
3. Data: every /vol/... argument (caches, gold answers, --init checkpoints) is copied from the Modal volume to the
   RunPod volume unless already there, and rewritten to /workspace/....
4. Command: --steps replaced if given; --checkpoint-every 100 added if absent (pods can be preempted).
5. Launch: runpod_launch.py with the image pinned to that commit and eval_snapshots.py as the post command when the
   run saves adapters, trying GPU tiers in order until one has stock, each with the --accumulate that fits it
   (Llama: 48 GB cards, A100 80GB, 5090, 4090; see LLAMA_TIERS). --gpu (with --accumulate) replaces the tiers. The promotion is recorded in
   runs/runpod/NAME/promoted_from.json.

--continue also copies the Modal run's latest checkpoint into the RunPod run, so with a larger --steps it carries on
from where Modal stopped (optimizer, data position and controller included) instead of starting over.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from runpod_launch import VOLUME_ID  # noqa: E402

CODE_PATHS = ["modal_repro", "src", "runpod", "patches"]
MODAL_VOLUME = "rewarding-doubt-repro"
GPUS_48GB = ["NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA A40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"]
GPUS_80GB = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
# Llama-3-8B tiers, tried in order until RunPod has stock: (GPU types, --accumulate). Accumulation splits the same
# 32-question update into K backward passes: identical updates, less memory, slower. Measured peaks: LoRA on layers
# 16-31 34.6 GB at 1, 18.0 at 4 (2 interpolates to ~25); all-layer LoRA 53 GB at 1, 33 at 2, 25 at 4 (8 untested).
LLAMA_TIERS = {
    "late": [(GPUS_48GB, 1), (GPUS_80GB, 1), (["NVIDIA GeForce RTX 5090"], 2), (["NVIDIA GeForce RTX 4090"], 4)],
    "all": [(GPUS_48GB, 2), (GPUS_80GB, 1), (["NVIDIA GeForce RTX 5090"], 4), (["NVIDIA GeForce RTX 4090"], 8)],
}
QWEN_TIERS = [(["NVIDIA GeForce RTX 4090", *GPUS_48GB], None)]  # Qwen-2.5-3B fits a 4090 as is


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_code(launch, allow_dirty_source):
    if launch.get("git_dirty") and not allow_dirty_source:
        raise SystemExit(f"{launch['label']} ran from uncommitted code (at {launch['git_sha'][:8]}); commit it, check it "
                         "matches what ran, and pass --allow-dirty-source")
    if git("status", "--porcelain", "--", *CODE_PATHS).stdout.strip():
        raise SystemExit("uncommitted changes under " + ", ".join(CODE_PATHS) + ": commit and push first")
    git("fetch", "-q", "origin", "main")
    if git("merge-base", "--is-ancestor", "HEAD", "origin/main").returncode:
        raise SystemExit("HEAD is not on origin/main: push first (the image is built from main)")
    head = git("rev-parse", "HEAD").stdout.strip()
    changed = git("diff", "--stat", launch["git_sha"], head, "--", *CODE_PATHS).stdout.strip()
    if changed:
        print(f"note: code changed since the run's commit {launch['git_sha'][:8]}:\n{changed}\n")
    return head


def image_paths():
    """The paths whose changes rebuild the image (the workflow's push filter)."""
    text = (ROOT / ".github/workflows/docker.yml").read_text()
    start = text.index("paths:")
    block = text[text.index("[", start) + 1:text.index("]", start)]
    return [p.strip().strip('"') for p in block.replace("\n", " ").split(",") if p.strip()]


def find_image(head):
    paths = image_paths()
    runs = json.loads(subprocess.run(["gh", "run", "list", "--workflow", "docker.yml", "--limit", "40", "--json",
                                      "databaseId,headSha,status,conclusion"], cwd=ROOT, capture_output=True, text=True).stdout)
    for r in runs:  # newest first
        sha = r["headSha"]
        if git("merge-base", "--is-ancestor", sha, head).returncode:
            continue
        if git("diff", "--quiet", sha, head, "--", *paths).returncode:
            continue  # image-relevant files changed after this build
        if r["status"] != "completed":
            print(f"waiting for image build {r['databaseId']} at {sha[:8]}...", flush=True)
            subprocess.run(["gh", "run", "watch", str(r["databaseId"]), "--exit-status"], cwd=ROOT, capture_output=True)
            r["conclusion"] = json.loads(subprocess.run(["gh", "run", "view", str(r["databaseId"]), "--json", "conclusion"],
                                                        cwd=ROOT, capture_output=True, text=True).stdout)["conclusion"]
        if r["conclusion"] == "success":
            return sha
        raise SystemExit(f"the image build at {sha[:8]} (run {r['databaseId']}) did not succeed")
    raise SystemExit("no image built from the current image-relevant code; push a commit that triggers the docker workflow")


def copy_to_runpod(modal_path, dry_run, dest_rel=None):
    """Copy /vol/<rel> (file or directory) from the Modal volume to /workspace/<dest_rel> (default: the same <rel>)
    unless already there."""
    from runpod_sync import client, objects
    rel = modal_path[len("/vol/"):].rstrip("/")
    dest_rel = dest_rel or rel
    s3 = client()
    if any(True for _ in objects(s3, dest_rel)):
        print(f"  {modal_path}: already on the RunPod volume at /workspace/{dest_rel}")
        return
    print(f"  {modal_path}: copying to /workspace/{dest_rel}{' (dry run)' if dry_run else ''}", flush=True)
    if dry_run:
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["modal", "volume", "get", "--force", MODAL_VOLUME, rel, tmp], check=True, capture_output=True)
        local = Path(tmp) / Path(rel).name
        files = [local] if local.is_file() else [p for p in local.rglob("*") if p.is_file()]
        for f in files:
            key = dest_rel if f == local else f"{dest_rel}/{f.relative_to(local)}"
            s3.upload_file(str(f), VOLUME_ID, key)


def all_layer_lora(args):
    trains_lora = "--lora-modules" not in args or args[args.index("--lora-modules") + 1] != "none"
    return trains_lora and ("--lora-layers" not in args or args[args.index("--lora-layers") + 1] == "all")


def tiers(command, gpus, accumulate):
    """[(GPU types, --accumulate or None to keep the command's)] to try in order."""
    if gpus:
        return [(gpus, accumulate or None)]
    if "llama" not in command[2].lower():
        return [(g, accumulate or a) for g, a in QWEN_TIERS]
    return LLAMA_TIERS["all" if all_layer_lora(command) else "late"]


def translate(command, steps, accumulate=None):
    """The RunPod train command: /vol paths rewritten, --steps / --checkpoint-every / --accumulate set."""
    args = list(command)
    if steps and "--steps" in args:
        args[args.index("--steps") + 1] = str(steps)
    elif steps:
        args += ["--steps", str(steps)]
    if accumulate:
        if "--accumulate" in args:
            args[args.index("--accumulate") + 1] = str(accumulate)
        else:
            args += ["--accumulate", str(accumulate)]
    if "--checkpoint-every" not in args:
        args += ["--checkpoint-every", "100"]
    vol_paths = [a for a in args if a.startswith("/vol/")]
    args = [a.replace("/vol/", "/workspace/", 1) if a.startswith("/vol/") else a for a in args]
    return args, vol_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--steps", type=int, default=0, help="run this many steps instead of the original's")
    parser.add_argument("--name", default="", help="RunPod run name (default: LABEL-rp)")
    parser.add_argument("--gpu", action="append", default=None)
    parser.add_argument("--max-hours", type=float, default=6)
    parser.add_argument("--accumulate", type=int, default=0, help="with --gpu: set --accumulate for those GPUs")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="copy the Modal run's latest checkpoint into the RunPod run, which then resumes from it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    launch = json.loads((ROOT / "runs/minimal" / args.label / "launch.json").read_text())
    head = check_code(launch, args.allow_dirty_source)
    image_sha = find_image(head)
    _, vol_paths = translate(launch["command"], args.steps)
    print(f"image: ghcr.io/rrenaud/rewarding-doubt:{image_sha[:8]}")
    print("data:")
    for p in vol_paths:
        copy_to_runpod(p, args.dry_run)
    name = args.name or f"{args.label}-rp"
    if args.resume:  # the trainer resumes from OUT_DIR/checkpoint when it exists (same command, larger --steps)
        copy_to_runpod(f"/vol/outputs/{launch['modal_run']}/{args.label}/checkpoint", args.dry_run, dest_rel=f"runs/{name}/checkpoint")
    for gpus, accumulate in tiers(launch["command"], args.gpu, args.accumulate):
        command, _ = translate(launch["command"], args.steps, accumulate)
        post = "python /opt/runpod/eval_snapshots.py OUT_DIR IDS_JSON" if "--save-adapter-every" in command else ""
        launcher = ["python3", str(ROOT / "scripts/runpod_launch.py"), name, "--expt", name.split("-")[0], "--max-hours",
                    str(args.max_hours), "--image", f"ghcr.io/rrenaud/rewarding-doubt:{image_sha}", "--train-cmd", "python " + " ".join(command)]
        for g in gpus:
            launcher += ["--gpu", g]
        if post:
            launcher += ["--post-cmd", post]
        if args.dry_run:
            launcher.append("--dry-run")
        print(f"trying {', '.join(g.replace('NVIDIA ', '') for g in gpus)} with --accumulate {accumulate or 'as in the run'}", flush=True)
        result = subprocess.run(launcher, capture_output=True, text=True)
        out = result.stdout.strip()
        if "no instances currently available" in out + result.stderr:
            print("  none available", flush=True)
            continue
        try:
            pod = json.loads(out)
        except json.JSONDecodeError:
            print(out[-1500:], result.stderr[-1500:])
            raise SystemExit("launch failed")
        print("train command:", "python " + " ".join(command))
        print(json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "desiredStatus")} if not args.dry_run else
                         {k: pod.get(k) for k in ("name", "imageName", "gpuTypeIds")}, indent=1))
        if not args.dry_run:
            record = ROOT / "runs/runpod" / name
            record.mkdir(parents=True, exist_ok=True)
            (record / "promoted_from.json").write_text(json.dumps(dict(launch, image_sha=image_sha, runpod_command=command,
                                                                       gpus=gpus, post_cmd=post), indent=1) + "\n")
        return
    raise SystemExit("no GPU tier has stock in the volume's datacenter; retry later")


if __name__ == "__main__":
    main()
