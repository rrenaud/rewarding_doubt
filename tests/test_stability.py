import json
import math
import os
import random

import torch

from rewarding_doubt.checkpoint import (load_checkpoint, load_trainable_state, rng_state, save_checkpoint,
                                        set_rng_state, trainable_state, truncate_jsonl)
from rewarding_doubt.stability import StabilityMonitor


def run(monitor, steps, confidences, correct):
    events = []
    for step in range(1, steps + 1):
        record, new = monitor.update(step, confidences(step), correct(step), loss=0.1)
        events += new
    return record, events


def test_healthy_run_raises_nothing():
    rng = random.Random(0)
    record, events = run(StabilityMonitor(), 200, lambda s: [rng.choice([5, 6, 7, 8, 9]) for _ in range(8)],
                         lambda s: [rng.random() < 0.65 for _ in range(8)])
    assert events == [] and record["flags"] == []
    assert 0.5 < record["window_accuracy"] < 0.8


def test_collapse_after_warmup_is_flagged_once():
    # Like exact seed 8: varied confidences, then every row says 2 from step 150 on.
    rng = random.Random(1)
    conf = lambda s: [2] * 8 if s >= 150 else [rng.choice([5, 6, 7, 8]) for _ in range(8)]
    record, events = run(StabilityMonitor(), 200, conf, lambda s: [rng.random() < 0.65 for _ in range(8)])
    assert [e["flag"] for e in events] == ["collapsed"]
    assert 150 < events[0]["step"] <= 166
    assert "collapsed" in record["flags"] and record["window_levels"] == 1


def test_constant_base_model_during_warmup_is_not_collapse():
    rng = random.Random(2)
    conf = lambda s: [10] * 8 if s <= 20 else [rng.choice([4, 6, 8, 10]) for _ in range(8)]
    _, events = run(StabilityMonitor(), 60, conf, lambda s: [rng.random() < 0.65 for _ in range(8)])
    assert events == []


def test_format_and_accuracy_flags():
    rng = random.Random(3)
    conf = lambda s: [None] * 4 + [7] * 2 + [8] * 2 if s > 100 else [rng.choice([6, 7, 8]) for _ in range(8)]
    acc = lambda s: [rng.random() < (0.3 if s > 100 else 0.65) for _ in range(8)]
    _, events = run(StabilityMonitor(), 140, conf, acc)
    assert {e["flag"] for e in events} == {"format_broken", "answers_degraded"}


def test_nonfinite_scalar():
    record, events = StabilityMonitor().update(1, [7] * 8, [1] * 8, loss=float("nan"))
    assert record["flags"] == ["nonfinite"] and events[0]["nonfinite"] == ["loss"]


def test_monitor_state_round_trip():
    a, b = StabilityMonitor(), StabilityMonitor()
    rng = random.Random(4)
    for s in range(1, 40):
        a.update(s, [rng.choice([6, 7, 8]) for _ in range(8)], [1, 0] * 4)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    ra, _ = a.update(40, [7] * 8, [1] * 8)
    rb, _ = b.update(40, [7] * 8, [1] * 8)
    assert ra == rb


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))
    model[0].weight.requires_grad_(False)  # stands in for the frozen base model
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    random.seed(5)
    directory = str(tmp_path / "checkpoint")
    save_checkpoint(directory, dict(step=7, weights=trainable_state(model), optimizer=opt.state_dict(), rng=rng_state()))
    expected = (random.random(), torch.rand(1).item())

    fresh = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))
    fresh[0].weight.requires_grad_(False)
    fresh_opt = torch.optim.Adam([p for p in fresh.parameters() if p.requires_grad], lr=0.1)
    state = load_checkpoint(directory)
    load_trainable_state(state["weights"], fresh)
    fresh_opt.load_state_dict(state["optimizer"])
    set_rng_state(state["rng"])
    assert state["step"] == 7
    assert torch.equal(fresh[1].weight, model[1].weight) and torch.equal(fresh[0].bias, model[0].bias)
    assert not torch.equal(fresh[0].weight, model[0].weight)  # frozen weights are not stored
    assert torch.equal(fresh_opt.state_dict()["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"])
    assert (random.random(), torch.rand(1).item()) == expected
    assert json.load(open(os.path.join(directory, "info.json"))) == {"step": 7}


def test_interrupted_save_recovers_previous(tmp_path):
    directory = str(tmp_path / "checkpoint")
    save_checkpoint(directory, dict(step=1))
    os.rename(directory, directory + ".old")  # crash between the two renames
    assert load_checkpoint(directory)["step"] == 1
    assert load_checkpoint(str(tmp_path / "none")) is None


def test_truncate_jsonl(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text("".join(json.dumps(dict(step=s)) + "\n" for s in range(1, 8)))
    truncate_jsonl(str(path), 4)
    assert [json.loads(l)["step"] for l in path.read_text().splitlines()] == [1, 2, 3, 4]


def test_rotating_checkpoint_keeps_last_healthy(tmp_path):
    from rewarding_doubt.checkpoint import save_rotating_checkpoint, write_status
    directory = str(tmp_path / "checkpoint")
    monitor = StabilityMonitor()
    saved = 0
    for step in range(1, 129):
        conf = [2] * 8 if step >= 70 else [random.Random(step).choice([5, 6, 7, 8]) for _ in range(8)]
        monitor.update(step, conf, [1, 0] * 4)
        if step % 32 == 0:
            save_rotating_checkpoint(directory, dict(step=step), monitor.clean_since(saved))
            saved = step
    # Collapse starts at 70 and is flagged by ~86: the checkpoint at 64 is followed by flags, so
    # the last confirmed-healthy one is 32.
    assert load_checkpoint(directory)["step"] == 128
    assert load_checkpoint(directory + "-healthy")["step"] == 32
    write_status(str(tmp_path), "diverged", step=128, flags=["collapsed"])
    status = json.load(open(tmp_path / "status.json"))
    assert status["status"] == "diverged" and status["healthy_checkpoint"] == {"step": 32}
