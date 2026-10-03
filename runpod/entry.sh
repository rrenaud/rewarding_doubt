#!/usr/bin/env bash
# Pod entry point: run one training job to an outcome, record it, then shut the pod down.
#
# Required environment:
#   RUN_DIR        output directory on the network volume, e.g. /workspace/runs/exact-long-s1
#   TRAIN_CMD      trainer command run from CODE_DIR; OUT_DIR and IDS_JSON are substituted, e.g.
#                  "python exact_llama.py IDS_JSON OUT_DIR --mode discrete-exact ... --stop-on collapsed,..."
#   IDS_JSON_B64   base64 of the question-ID JSON ({"train": [...] or "all", "validation": [...]}),
#                  or IDS_PATH, a file on the volume
# Optional:
#   POST_CMD       run after a completed training (e.g. evaluation); same substitutions
#   MAX_HOURS      wall-clock cap across all attempts of this run (default 12)
#   STALL_MINUTES  kill the trainer if its log has not changed for this long (default 30)
#   ON_EXIT        terminate | stop | none: what to do with the pod at the end (default terminate)
#   RUNPOD_API_KEY used by selfstop.py (RUNPOD_POD_ID is set by RunPod in every pod)
#   FORCE_RESUME   1: rerun even after a final outcome (e.g. resume a crash after fixing its cause);
#                  the previous status is kept as entry_status.<time>.json
#   UPLOAD_CMD     run after every final outcome, before the pod shuts down (e.g. copy RUN_DIR off
#                  the volume); OUT_DIR is substituted. Its failure is logged, not fatal.
#
# Outcomes (RUN_DIR/entry_status.json), by trainer exit code or watchdog:
#   0 completed | 3 diverged (a --stop-on stability flag) | 143 preempted | watchdog: hung, timeout
#   | anything else: crashed.
# Only "preempted" is resumable: the pod is being stopped by RunPod, so the script just exits and the
# same command resumes from RUN_DIR/checkpoint when the pod starts again. Every other outcome is
# final: a restarted pod sees entry_status.json and shuts down without rerunning.
set -u
: "${RUN_DIR:?RUN_DIR is required}" "${TRAIN_CMD:?TRAIN_CMD is required}"
CODE_DIR=${CODE_DIR:-/opt/RewardingDoubt/SingleAnswerSetting}
MAX_HOURS=${MAX_HOURS:-12}
STALL_SECONDS=${STALL_SECONDS:-$(( ${STALL_MINUTES:-30} * 60 ))}
CHECK_SECONDS=${CHECK_SECONDS:-30}
KILL_GRACE_SECONDS=${KILL_GRACE_SECONDS:-180}
ON_EXIT=${ON_EXIT:-terminate}
SELFSTOP=${SELFSTOP:-"python $(dirname "$0")/selfstop.py"}
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/train.log
ENTRY_LOG=$RUN_DIR/entry.log
STATUS=$RUN_DIR/entry_status.json
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ENTRY_LOG"; }

shutdown_pod() {
  [ "$ON_EXIT" = none ] && { note "ON_EXIT=none: leaving the pod running"; finish_idle; }
  for attempt in 1 2 3 4 5 6; do
    $SELFSTOP "$ON_EXIT" >> "$ENTRY_LOG" 2>&1 && { note "pod $ON_EXIT requested"; finish_idle; }
    note "selfstop attempt $attempt failed; retrying"
    sleep $(( attempt * ${SELFSTOP_RETRY_SECONDS:-30} ))
  done
  note "could not $ON_EXIT the pod: it keeps billing until stopped by hand or by scripts/runpod_reaper.py"
  finish_idle
}
# A pod restarts a container whose main process exits, so after a final outcome stay idle instead
# (ENTRY_TEST=1 exits, for tests).
finish_idle() { [ "${ENTRY_TEST:-0}" = 1 ] && exit 0; while true; do sleep 3600; done; }

write_status() {  # outcome exit_code [detail]
  python - "$1" "$2" "${3:-}" "$RUN_DIR" "$STATUS" <<'PY'
import json, os, sys, time
outcome, code, detail, run_dir, path = sys.argv[1:]
trainer = os.path.join(run_dir, "status.json")
info = dict(outcome=outcome, exit_code=int(code), detail=detail or None, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trainer_status=json.load(open(trainer)) if os.path.exists(trainer) else None)
json.dump(info, open(path + ".tmp", "w"), indent=1)
os.replace(path + ".tmp", path)
PY
}

if [ -f "$STATUS" ] && [ "${FORCE_RESUME:-0}" = 1 ]; then
  note "FORCE_RESUME: rerunning after outcome $(python -c "import json; print(json.load(open('$STATUS'))['outcome'])")"
  mv "$STATUS" "$RUN_DIR/entry_status.$(date -u +%Y%m%dT%H%M%SZ).json"
  rm -f "$RUN_DIR/status.json" "$RUN_DIR/entry_started"  # a forced rerun gets its own MAX_HOURS
fi
if [ -f "$STATUS" ]; then
  previous=$(python -c "import json; print(json.load(open('$STATUS'))['outcome'])")
  if [ "$previous" != preempted ]; then
    note "run already finished ($previous); not rerunning"
    shutdown_pod
  fi
fi
[ -f "$RUN_DIR/entry_started" ] || date +%s > "$RUN_DIR/entry_started"
deadline=$(( $(cat "$RUN_DIR/entry_started") + ${MAX_SECONDS:-$(( ${MAX_HOURS%.*} * 3600 ))} ))

IDS=$RUN_DIR/ids.json
if [ -n "${IDS_JSON_B64:-}" ]; then echo "$IDS_JSON_B64" | base64 -d > "$IDS"; else cp "${IDS_PATH:?IDS_JSON_B64 or IDS_PATH is required}" "$IDS"; fi
expand() { local c=${1//OUT_DIR/$RUN_DIR}; echo "${c//IDS_JSON/$IDS}"; }

child=
on_term() {  # RunPod is stopping the pod (spot preemption or a manual stop)
  note "SIGTERM: forwarding to the trainer so it checkpoints"
  [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  for _ in $(seq "$KILL_GRACE_SECONDS"); do kill -0 "$child" 2>/dev/null || break; sleep 1; done
  write_status preempted 143 "SIGTERM to the pod"
  exit 143
}
trap on_term TERM INT

note "attempt start (image ${IMAGE_GIT_SHA:-stock}): $(expand "$TRAIN_CMD")"
cd "$CODE_DIR"
bash -c "exec $(expand "$TRAIN_CMD")" >> "$LOG" 2>&1 &
child=$!
killed_for=
while kill -0 "$child" 2>/dev/null; do
  sleep "$CHECK_SECONDS" & wait $!
  now=$(date +%s)
  last=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  if [ $(( now - last )) -ge "$STALL_SECONDS" ]; then killed_for=hung
  elif [ "$now" -ge "$deadline" ]; then killed_for=timeout
  fi
  if [ -n "$killed_for" ]; then
    note "watchdog: $killed_for (log idle $(( now - last )) s); SIGTERM, then SIGKILL after ${KILL_GRACE_SECONDS} s"
    kill -TERM "$child" 2>/dev/null
    for _ in $(seq "$KILL_GRACE_SECONDS"); do kill -0 "$child" 2>/dev/null || break; sleep 1; done
    kill -KILL "$child" 2>/dev/null
    break
  fi
done
wait "$child"; code=$?
child=
note "trainer exited with code $code"

if [ -n "$killed_for" ]; then write_status "$killed_for" "$code"
else
  case $code in
    0)
      outcome=completed detail=
      if [ -n "${POST_CMD:-}" ]; then
        note "post: $(expand "$POST_CMD")"
        bash -c "exec $(expand "$POST_CMD")" >> "$RUN_DIR/post.log" 2>&1
        post=$?; [ $post -ne 0 ] && detail="post command exited with $post"
      fi
      write_status "$outcome" 0 "$detail" ;;
    3) write_status diverged 3 ;;
    143) write_status preempted 143 "trainer received SIGTERM"; note "preempted; exiting so the pod can restart and resume"; exit 143 ;;
    *) write_status crashed "$code" ;;
  esac
fi
note "outcome: $(python -c "import json; print(json.load(open('$STATUS'))['outcome'])")"
if [ -n "${UPLOAD_CMD:-}" ]; then
  note "upload: $(expand "$UPLOAD_CMD")"
  timeout 1800 bash -c "$(expand "$UPLOAD_CMD")" >> "$ENTRY_LOG" 2>&1 || note "upload failed (exit $?); results stay on the volume"
fi
shutdown_pod
