"""runpod/entry.sh with a fake trainer: every outcome, the restart rules, and the self-stop call."""
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE = '''
import os, signal, sys, time
mode = sys.argv[1]
signal.signal(signal.SIGTERM, lambda *a: (print("checkpoint on SIGTERM", flush=True), sys.exit(143)))
print("step 1", flush=True)
if mode == "ok": sys.exit(0)
if mode == "diverge": sys.exit(3)
if mode == "crash": sys.exit(1)
if mode == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # truly stuck: only SIGKILL ends it
    time.sleep(600)
if mode == "slow":  # keeps logging, never finishes
    while True:
        print("step", flush=True); time.sleep(0.3)
'''


def entry(tmp_path, mode, background=False, **env):
    (tmp_path / "fake.py").write_text(FAKE)
    run_dir = tmp_path / "run"
    full = dict(os.environ, RUN_DIR=str(run_dir), TRAIN_CMD=f"{sys.executable} {tmp_path}/fake.py {mode}",
                IDS_JSON_B64=base64.b64encode(b'{"train": "all"}').decode(), CODE_DIR=str(tmp_path),
                CHECK_SECONDS="1", STALL_SECONDS="3", KILL_GRACE_SECONDS="2", ENTRY_TEST="1",
                **{"SELFSTOP": "echo selfstop", **env})
    proc = subprocess.Popen(["bash", str(ROOT / "runpod/entry.sh")], env=full, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if background:
        return proc, run_dir
    out, _ = proc.communicate(timeout=60)
    return proc.returncode, out, run_dir


def outcome(run_dir):
    return json.loads((run_dir / "entry_status.json").read_text())["outcome"]


def test_completed_runs_post_and_terminates(tmp_path):
    code, out, run_dir = entry(tmp_path, "ok", POST_CMD="echo evaluating OUT_DIR")
    assert outcome(run_dir) == "completed" and "pod terminate requested" in out
    assert f"evaluating {run_dir}" in (run_dir / "post.log").read_text()
    assert json.loads((run_dir / "ids.json").read_text()) == {"train": "all"}
    # A restarted pod must not rerun a finished job.
    code, out, _ = entry(tmp_path, "ok")
    assert "already finished (completed)" in out and (run_dir / "train.log").read_text().count("step 1") == 1


def test_diverged_and_crashed_are_final(tmp_path):
    for mode, expected in (("diverge", "diverged"), ("crash", "crashed")):
        (tmp_path / mode).mkdir()
        _, out, run_dir = entry(tmp_path / mode, mode)
        assert outcome(run_dir) == expected and "pod terminate requested" in out


def test_watchdog_kills_a_hung_trainer(tmp_path):
    _, out, run_dir = entry(tmp_path, "hang")
    assert outcome(run_dir) == "hung" and "watchdog: hung" in out


def test_watchdog_enforces_the_time_cap(tmp_path):
    _, out, run_dir = entry(tmp_path, "slow", MAX_SECONDS="3")
    assert outcome(run_dir) == "timeout" and "checkpoint on SIGTERM" in (run_dir / "train.log").read_text()


def test_preemption_forwards_sigterm_and_allows_resume(tmp_path):
    proc, run_dir = entry(tmp_path, "slow", background=True)
    time.sleep(2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143 and outcome(run_dir) == "preempted"
    assert "checkpoint on SIGTERM" in (run_dir / "train.log").read_text() and "selfstop" not in out
    # The restarted pod reruns the command (the real trainer resumes from its checkpoint).
    _, out, _ = entry(tmp_path, "ok")
    assert outcome(run_dir) == "completed"


def test_failed_selfstop_is_retried_then_reported(tmp_path):
    _, out, run_dir = entry(tmp_path, "ok", SELFSTOP="false", SELFSTOP_RETRY_SECONDS="0")
    log = (run_dir / "entry.log").read_text()
    assert "selfstop attempt 6 failed" in log and "keeps billing" in log


def test_force_resume_reruns_a_finished_run(tmp_path):
    _, _, run_dir = entry(tmp_path, "crash")
    assert outcome(run_dir) == "crashed"
    _, out, _ = entry(tmp_path, "ok", FORCE_RESUME="1")
    assert outcome(run_dir) == "completed" and "FORCE_RESUME: rerunning after outcome crashed" in out
    assert len(list(run_dir.glob("entry_status.*.json"))) == 1
