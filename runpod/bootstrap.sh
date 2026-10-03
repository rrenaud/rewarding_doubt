#!/usr/bin/env bash
# Pod start command (no custom image needed): set up the pinned environment once on the network
# volume, assemble this pod's code directory, then hand over to entry.sh.
#
# The launcher (scripts/runpod_launch.py) unpacks our code to /opt/code from the CODE_B64 env var
# and runs this script on a stock python:3.11 image (git, gcc). The released RewardingDoubt code
# at the pinned commit and a virtualenv with its pinned requirements live on the volume under
# /workspace/envs and are reused by every later pod. SETUP_ONLY=1 stops after the setup.
set -euo pipefail
COMMIT=416c9e0de22a387c35a51260e7739fa28a37865e
ROOT=/workspace/envs/rd-${COMMIT:0:7}
mkdir -p /workspace/envs
log() { echo "$(date -u +%FT%TZ) bootstrap: $*"; }
if [ ! -f "$ROOT/.ready" ]; then
  # One pod sets up at a time (flock on the volume; launch the first pod alone to be safe).
  exec 9> /workspace/envs/.lock
  flock 9
  if [ ! -f "$ROOT/.ready" ]; then
    log "installing the pinned environment into $ROOT (first pod on this volume; ~10-20 min)"
    rm -rf "$ROOT"
    git clone -q https://github.com/pasta99/RewardingDoubt "$ROOT/RewardingDoubt"
    git -C "$ROOT/RewardingDoubt" checkout -q "$COMMIT"
    python3.11 -m venv "$ROOT/venv"
    "$ROOT/venv/bin/pip" install -q --no-cache-dir -r "$ROOT/RewardingDoubt/requirements.txt"
    touch "$ROOT/.ready"
  fi
  flock -u 9
fi
# This pod's code: the released SingleAnswerSetting plus our trainers next to Train.py.
export CODE_DIR=/opt/rd/SingleAnswerSetting
rm -rf /opt/rd && mkdir -p /opt/rd
cp -r "$ROOT/RewardingDoubt/SingleAnswerSetting" "$CODE_DIR"
cp /opt/code/modal_repro/{subset,exact_llama,shared_prefix,thinking_llama}.py "$CODE_DIR/"
export PATH="$ROOT/venv/bin:$PATH" PYTHONPATH=/opt/code/src HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 \
  TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled
if [ "${SETUP_ONLY:-0}" = 1 ]; then
  log "setup only: done"
  python /opt/code/runpod/selfstop.py "${ON_EXIT:-terminate}"
  while true; do sleep 3600; done
fi
exec bash /opt/code/runpod/entry.sh
