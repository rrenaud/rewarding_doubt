"""Resumable training checkpoints: trainable weights, optimizer, RNG streams and loop position.

A checkpoint is one directory (`OUT_DIR/checkpoint`) holding `state.pt`. It is written to a
temporary sibling and swapped in with renames, so an interruption mid-save leaves the previous
checkpoint intact.

`OUT_DIR/checkpoint-healthy` is the newest checkpoint that was followed by a full checkpoint
interval with no stability flag. A flag rises up to one monitor window after trouble starts, so
the latest checkpoint of a run that diverges may already be damaged; the healthy one predates it
(as long as the checkpoint interval is at least the monitor window).

Only parameters with requires_grad are stored (the LoRA weights, plus a value head if any); the
frozen 4-bit base model is reloaded from its hub snapshot as usual.
"""
import json
import os
import random
import shutil

import numpy as np
import torch


def trainable_state(*modules):
    """{module index: {name: tensor}} for every parameter that requires grad."""
    return {i: {n: p.detach().cpu().clone() for n, p in m.named_parameters() if p.requires_grad}
            for i, m in enumerate(modules)}


def load_trainable_state(state, *modules):
    for i, m in enumerate(modules):
        params = dict(m.named_parameters())
        missing = [n for n, p in params.items() if p.requires_grad and n not in state[i]]
        if missing:
            raise ValueError(f"checkpoint lacks {len(missing)} trainable parameters, e.g. {missing[0]}")
        with torch.no_grad():
            for n, value in state[i].items():
                params[n].copy_(value.to(params[n].device, params[n].dtype))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(directory, state):
    """Atomically replace `directory` with a checkpoint holding `state` (a picklable dict)."""
    tmp, old = directory + ".tmp", directory + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save(state, os.path.join(tmp, "state.pt"))
    with open(os.path.join(tmp, "info.json"), "w") as f:
        json.dump({k: v for k, v in state.items() if isinstance(v, (int, float, str))}, f)
    if os.path.exists(directory):
        shutil.rmtree(old, ignore_errors=True)
        os.rename(directory, old)
    os.rename(tmp, directory)
    shutil.rmtree(old, ignore_errors=True)


def save_rotating_checkpoint(directory, state, previous_was_healthy):
    """Save `state` as the latest checkpoint; first promote the current latest to `<dir>-healthy`
    if nothing went wrong since it was written."""
    healthy = directory + "-healthy"
    if previous_was_healthy and os.path.exists(directory):
        shutil.rmtree(healthy + ".old", ignore_errors=True)
        if os.path.exists(healthy):
            os.rename(healthy, healthy + ".old")
        shutil.copytree(directory, healthy + ".tmp")
        os.rename(healthy + ".tmp", healthy)
        shutil.rmtree(healthy + ".old", ignore_errors=True)
    save_checkpoint(directory, state)


def write_status(out_dir, status, **details):
    """OUT_DIR/status.json: completed | diverged | preempted, for launchers and reports."""
    healthy = os.path.join(out_dir, "checkpoint-healthy", "info.json")
    info = dict(status=status, **details,
                healthy_checkpoint=json.load(open(healthy)) if os.path.exists(healthy) else None)
    with open(os.path.join(out_dir, "status.json.tmp"), "w") as f:
        json.dump(info, f, indent=1)
    os.replace(os.path.join(out_dir, "status.json.tmp"), os.path.join(out_dir, "status.json"))


def load_checkpoint(directory):
    """The saved state, or None. Recovers from a save interrupted between its two renames."""
    for d in (directory, directory + ".old", directory + "-healthy"):
        path = os.path.join(d, "state.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
    return None


def truncate_jsonl(path, last_step, key="step"):
    """Drop log lines written after the checkpoint (they will be written again on resume)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [l for l in f if l.strip() and json.loads(l).get(key, 0) <= last_step]
    with open(path, "w") as f:
        f.writelines(lines)
