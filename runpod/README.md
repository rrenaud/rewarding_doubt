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

```bash
python scripts/runpod_launch.py exact-long-s1 --ids-train-all --max-hours 8 \
  --train-cmd "python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact --scoring single \
    --regularization hinge --reward paper --passes 2 --minibatch 4 --lr 1e-05 --format-weight 1.01 \
    --seed 1 --epochs 1 --max-steps 4000 --save-every 500 \
    --stop-on collapsed,answers_degraded,format_broken,nonfinite"
python scripts/runpod_sync.py --list
python scripts/runpod_sync.py exact-long-s1
python scripts/runpod_reaper.py          # backstop; --apply to terminate overdue pods
```

Tested 2026-10-03 (`runs/runpod/resume-test-{1,image}`): 32 questions, stop at step 3 as if
preempted; RunPod restarted the container within 7 s, the run resumed and finished 8 steps,
and the pod terminated itself.
