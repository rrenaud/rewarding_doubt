"""Copy run directories from the RunPod network volume to runs/runpod/ over its S3 API.

    python scripts/runpod_sync.py                 # every run under /workspace/runs
    python scripts/runpod_sync.py NAME [NAME...]  # selected runs
    python scripts/runpod_sync.py --list          # runs on the volume with their outcome

Skips files whose local copy has the same size, and the resumable checkpoints (checkpoint/,
checkpoint-healthy/, ~130 MB each) unless --checkpoints. Credentials: the S3 secret in
~/.runpod_rewarding_doubt_key_s3.txt; the access key is the RunPod user ID, read from the API.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_launch import VOLUME_ID, api_key, call  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = {"EU-RO-1": "https://s3api-eu-ro-1.runpod.io"}


def user_id():
    request = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": "{ myself { id } }"}).encode(),
                                     headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                                              "User-Agent": "rewarding-doubt-sync/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["data"]["myself"]["id"]


def client():
    region = next(v for v in call("GET", "/networkvolumes") if v["id"] == VOLUME_ID)["dataCenterId"]
    secret = (Path.home() / ".runpod_rewarding_doubt_key_s3.txt").read_text().strip()
    return boto3.client("s3", endpoint_url=ENDPOINTS[region], region_name=region.lower(), aws_access_key_id=user_id(),
                        aws_secret_access_key=secret, config=Config(signature_version="s3v4"))


def objects(s3, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=VOLUME_ID, Prefix=prefix):
        yield from page.get("Contents", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    s3 = client()
    names = args.names or [p["Prefix"].split("/")[1] for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=VOLUME_ID, Prefix="runs/", Delimiter="/") for p in page.get("CommonPrefixes", [])]
    for name in names:
        if args.list:
            try:
                status = json.loads(s3.get_object(Bucket=VOLUME_ID, Key=f"runs/{name}/entry_status.json")["Body"].read())
                print(f"{name:<32} {status['outcome']:<10} {status['time']}  {status.get('detail') or ''}")
            except s3.exceptions.NoSuchKey:
                print(f"{name:<32} running or not started")
            continue
        copied = skipped = 0
        for obj in objects(s3, f"runs/{name}/"):
            relative = obj["Key"][len("runs/"):]
            if not args.checkpoints and relative.split("/")[1].startswith("checkpoint"):
                continue
            local = ROOT / "runs/runpod" / relative
            if local.exists() and local.stat().st_size == obj["Size"]:
                skipped += 1
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(VOLUME_ID, obj["Key"], str(local))
            copied += 1
        print(f"{name}: {copied} files copied, {skipped} unchanged -> runs/runpod/{name}")


if __name__ == "__main__":
    main()
