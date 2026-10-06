# Training on RunPod

Long runs go to RunPod pods instead of Modal: roughly 2-4x cheaper per GPU-hour for multi-hour
jobs. Each pod runs one training job to an outcome, records it, and terminates itself.

| Piece | What it does |
|---|---|
| `runpod/Dockerfile` | the Modal image as `ghcr.io/rrenaud/rewarding-doubt` (public), built by `.github/workflows/docker.yml` on code changes |
| `runpod/entry.sh` | runs the trainer; outcome from its exit code or a watchdog; only `preempted` reruns on restart; then terminates the pod |
| `scripts/runpod_launch.py` | creates a pod for one run on the network volume |
| `scripts/runpod_sync.py` | copies runs from the volume to `runs/runpod/` over its S3 API; `--list` shows outcomes |
| `scripts/runpod_reaper.py` | lists `rd-*` pods; `--apply` terminates any running past `MAX_HOURS` + 1 h |
| `runpod/bootstrap.sh` | fallback without the image (`--stock-image`): environment installed once on the volume |

Storage: network volume `rewarding-doubt` (`591nr7jbh1`, 75 GB, EU-RO-1) at `/workspace`, so pods
must run in EU-RO-1. It holds the Hugging Face cache (model and TriviaQA, downloaded once) and
`runs/NAME/` for every run. Credentials: `~/.runpod_rewarding_doubt_key.txt` (API key; also passed
to each pod so it can terminate itself) and `~/.runpod_rewarding_doubt_key_s3.txt` (S3 secret).

## Outcomes (`runs/NAME/entry_status.json`)

| Outcome | Cause | On pod restart |
|---|---|---|
| completed | trainer exit 0 (then `--post-cmd`, if any) | not rerun |
| diverged | exit 3: a `--stop-on` stability flag rose; `checkpoint-healthy/` holds the last good state | not rerun |
| preempted | exit 143: SIGTERM (spot reclaim, manual stop) after a checkpoint | **resumes from `checkpoint/`** |
| hung / timeout | watchdog: no log output for `--stall-minutes`, or past `--max-hours` | not rerun |
| crashed | any other exit code | not rerun |

## Example

New runs use `--frozen-answers` (answers from the base model, so confidence training cannot change
them; see `modal_repro/exact_llama.py`) and score snapshots with the matching evaluator:

```bash
python scripts/runpod_launch.py long3-exact-s1 --ids-train-all --max-hours 8 \
  --post-cmd "python /opt/runpod/eval_snapshots.py OUT_DIR IDS_JSON --frozen-answers" \
  --train-cmd "python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact --scoring single \
    --regularization hinge --reward paper --passes 2 --minibatch 4 --lr 4.01e-05 --format-weight 1.01 \
    --seed 1 --epochs 1 --max-steps 4000 --save-every 500 --frozen-answers \
    --stop-on collapsed,answers_degraded,format_broken,nonfinite"
python scripts/runpod_sync.py --list
python scripts/runpod_sync.py long3-exact-s1
python scripts/runpod_reaper.py          # backstop; --apply to terminate overdue pods
```

For PPO, `subset.py train IDS_JSON --frozen-answers ...` does the same for Train.py's answer step.

Tested 2026-10-03 (`runs/runpod/resume-test-{1,image}`): 32 questions, stop at step 3 as if
preempted; RunPod restarted the container within 7 s, the run resumed and finished 8 steps,
and the pod terminated itself.

## The minimal trainer

`modal_repro/minimal_trainer.py` (plain transformers + PEFT; `docs/minimal_trainer.md`) is in the image and
reads its data from the network volume: `/workspace/fast/qwen25-3b-cache-bf16ref-top64.pt` (the cached
rollouts with bf16 reference levels and the answer reference top-64) and `/workspace/fast/gt.json` (gold
answers of every cached question). They come from the Modal volume; to refresh them:

```bash
modal volume get rewarding-doubt-repro fast/qwen25-3b-cache-bf16ref-top64.pt /tmp/top64.pt
python scripts/runpod_sync.py --push /tmp/top64.pt fast/qwen25-3b-cache-bf16ref-top64.pt
```

With `--checkpoint-every N` it checkpoints every N steps and on SIGTERM (exit 143), so a preempted pod
resumes. It reports its own dev metrics, so no `--post-cmd` is needed (the launcher's question IDs are
unused). Example, an online attention-bias run:

```bash
python scripts/runpod_launch.py online-bias-s1 --max-hours 4 \
  --train-cmd "python minimal_trainer.py train /workspace/fast/qwen25-3b-cache-bf16ref-top64.pt OUT_DIR \
    --online --gt /workspace/fast/gt.json --lora-modules none --attn-bias 18-35 --lr 1e-2 --answer-kl 0.1 \
    --steps 3000 --eval-every 100 --regen-every 500 --checkpoint-every 100"
```

Memory: all-layer LoRA peaks at about 33 GB over a run, so it needs a 48 GB GPU
(`--gpu "NVIDIA RTX A6000" --gpu "NVIDIA L40S"`); attention biases and LoRA on fewer layers fit a 24 GB 4090.
Tested on Modal (`minimal_restart_check`): stopped at step 12 of 20, the same command resumed at step 12
and finished with every step logged once.
