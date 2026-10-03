"""Backstop for runaway pods: list our pods (named rd-*) and terminate any running past its cap.

    python scripts/runpod_reaper.py            # list, terminate nothing
    python scripts/runpod_reaper.py --apply    # terminate rd-* pods older than MAX_HOURS + grace

A pod normally terminates itself (runpod/entry.sh). This catches the cases where it cannot: the
self-stop call failed, the entry script died, or the pod never booted. Age is measured from the
pod's lastStartedAt; the cap is the pod's own MAX_HOURS env var (default 12) plus --grace-hours.
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_launch import call  # noqa: E402


def parse_time(text):
    """RunPod timestamps look like '2026-10-03 01:11:59.272 +0000 UTC'."""
    text = text.replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--grace-hours", type=float, default=1.0)
    args = parser.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    for pod in call("GET", "/pods"):
        if not pod["name"].startswith("rd-"):
            continue
        started = pod.get("lastStartedAt") or pod.get("createdAt")
        age = (now - parse_time(started)).total_seconds() / 3600 if started else 0.
        cap = float((pod.get("env") or {}).get("MAX_HOURS", 12)) + args.grace_hours
        over = pod["desiredStatus"] == "RUNNING" and age > cap
        print(f"{pod['id']} {pod['name']:<32} {pod['desiredStatus']:<9} {age:6.2f} h (cap {cap:g} h) "
              f"${pod.get('costPerHr', 0)}/h{'  OVER CAP' if over else ''}")
        if over and args.apply:
            call("DELETE", f"/pods/{pod['id']}")
            print(f"  terminated {pod['id']}")


if __name__ == "__main__":
    main()
