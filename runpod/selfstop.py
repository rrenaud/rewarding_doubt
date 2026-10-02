"""Stop or terminate the pod this runs in (RUNPOD_POD_ID), with RUNPOD_API_KEY.

    python selfstop.py terminate|stop

RunPod REST API v1: DELETE /pods/{id} terminates (the pod and its container disk are deleted;
the network volume stays), POST /pods/{id}/stop stops (GPU billing ends, the pod's own volume
disk is kept and billed). Exit code 0 on success.
"""
import os
import sys
import urllib.request


def main(action):
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        print(f"selfstop: RUNPOD_POD_ID or RUNPOD_API_KEY missing; not running {action}", flush=True)
        return 2
    url, method = {"terminate": (f"https://rest.runpod.io/v1/pods/{pod}", "DELETE"),
                   "stop": (f"https://rest.runpod.io/v1/pods/{pod}/stop", "POST")}[action]
    request = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {key}", "User-Agent": "rewarding-doubt-selfstop/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"selfstop: {action} pod {pod}: HTTP {response.status}", flush=True)
            return 0
    except Exception as e:  # noqa: BLE001 - report; entry.sh retries
        print(f"selfstop: {action} pod {pod} failed: {e!r}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
