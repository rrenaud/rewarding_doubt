"""Successive-halving hyperparameter search: exact + hinge vs PPO (fast, no KL) + hinge.

    python scripts/hparam_search.py init                 # dev split, protocol, round-1 configs
    modal run modal_repro/app.py::hparam_round --search-dir DIR --round round1
    python scripts/hparam_search.py select DIR round1    # rank on dev Brier, write round-2 configs
    modal run modal_repro/app.py::hparam_round --search-dir DIR --round round2
    python scripts/hparam_search.py select DIR round2    # pick finalists, write final configs
    modal run modal_repro/app.py::hparam_round --search-dir DIR --round final --score-test

Amended (see PROTOCOL.md): after round 1, greedy one-dimensional sweeps.
    python scripts/hparam_search.py start DIR            # best round-1 config per arm -> current.json
    python scripts/hparam_search.py sweep DIR stage1     # write stage1_configs.json around current
    modal run modal_repro/app.py::hparam_round --search-dir DIR --round stage1
    python scripts/hparam_search.py advance DIR stage1   # keep the best point per arm
    ... stage2, stage3, stage4, then:
    python scripts/hparam_search.py final DIR

The protocol (PROTOCOL.md, written by `init`) is fixed before any result is seen.
"""
import datetime
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RUN = ROOT / "runs/pilot-20261001T002945Z"
MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ROUND1_PER_ARM, ROUND2_KEEP, ROUND2_SEEDS, FINAL_SEEDS = 16, 4, (1, 2), (4, 5, 6)
DEFAULTS = {"exact": dict(lr=1e-5, updates=4, hinge_weight=1.0, hinge_threshold=0.95),
            "ppo": dict(lr=1e-5, updates=4, hinge_weight=1.0, hinge_threshold=0.95, cliprange=0.2, vf_coef=0.1)}

PROTOCOL = """# Hyperparameter search protocol (fixed before any result)

Arms, batch 8, one L40S per run, the same 1,024 training questions:
- exact: discrete-exact, single pass, format hinge (number mass + stop after the sampled number), paper-scale reward, no KL.
- ppo: released Train.py with --fast (numerically identical speedups), KL off, the same two hinges.

Selection: dev Brier score of the sampled confidence (released evaluation), at the round's last snapshot,
averaged over seeds. Dev = 512 TriviaQA `unfiltered` validation questions disjoint from the 512 test
questions. Test is scored once, in the final round.

Search spaces (random, seeded):
- both: lr log-uniform [3e-6, 3e-4]; updates per batch via passes / ppo_epochs in {1, 2, 4, 8}
  (x 2 minibatches of 4); hinge weight log-uniform [0.1, 10]; hinge threshold in {0.9, 0.95, 0.99}.
- ppo only: cliprange in {0.1, 0.2, 0.3}; vf_coef log-uniform [0.03, 0.3].
Each arm's round 1 includes its current default (lr 1e-5, 4 passes/epochs, weight 1, threshold 0.95).

Successive halving, equal budget per arm:
1. round1: 16 configs per arm, 1 epoch (128 steps), seed 1.
2. round2: top 4 per arm by dev Brier, 2 epochs (256 steps), seeds 1 and 2.
3. final: best per arm by mean dev Brier, 2 epochs, fresh seeds 4, 5, 6; scored on dev and test.
"""


def command(arm, params, seed, epochs):
    save_every = 128  # end of each epoch
    if arm == "exact":
        return ["exact_llama.py", "IDS_JSON", "OUT_DIR", "--mode", "discrete-exact", "--scoring", "single",
                "--regularization", "hinge", "--reward", "paper", "--passes", str(params["updates"]), "--minibatch", "4",
                "--lr", repr(params["lr"]), "--format-weight", repr(params["hinge_weight"]),
                "--format-threshold", repr(params["hinge_threshold"]), "--seed", str(seed), "--epochs", str(epochs),
                "--save-every", str(save_every)]
    return ["subset.py", "train", "IDS_JSON", "--save-every", str(save_every), "--seed", str(seed), "--fast", "--no-kl",
            "--hinge", repr(params["hinge_weight"]), "--hinge-threshold", repr(params["hinge_threshold"]),
            "--ppo", f"ppo_epochs={params['updates']}", "--ppo", f"cliprange={params['cliprange']}",
            "--ppo", f"vf_coef={params['vf_coef']}", "--",
            "--out_dir", "OUT_DIR", "--dataset", "triviaqa", "--is_unsloth", "--model_dir", MODEL,
            "--tokenizer_dir", MODEL, "--epochs", str(epochs), "--lr", repr(params["lr"]), "--batchsize", "8",
            "--log_with", "tensorboard"]


def sample(arm, rng):
    log_uniform = lambda lo, hi: float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.3g}")
    params = dict(lr=log_uniform(3e-6, 3e-4), updates=rng.choice([1, 2, 4, 8]),
                  hinge_weight=log_uniform(0.1, 10), hinge_threshold=rng.choice([0.9, 0.95, 0.99]))
    if arm == "ppo":
        params.update(cliprange=rng.choice([0.1, 0.2, 0.3]), vf_coef=log_uniform(0.03, 0.3))
    return params


def write_round(search, name, entries, epochs, ids_note):
    configs = {e["label"]: dict(arm=e["arm"], params=e["params"], seed=e["seed"], epochs=epochs,
                                command=command(e["arm"], e["params"], e["seed"], epochs)) for e in entries}
    (search / f"{name}_configs.json").write_text(json.dumps(configs, indent=1) + "\n")
    print(f"wrote {len(configs)} configs to {search / f'{name}_configs.json'} ({ids_note})")


def init():
    from datasets import load_dataset
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    search = ROOT / "runs" / f"hparam-search-{stamp}"
    search.mkdir(parents=True)
    test_ids = {json.loads(l)["id"] for l in (DATA_RUN / "data/eval.jsonl").read_text().splitlines()}
    pool = [r["question_id"] for r in load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext",
                                                   split="validation", streaming=True)
            if r["question_id"] not in test_ids]
    dev = random.Random(0).sample(sorted(pool), 512)
    assert not set(dev) & test_ids
    (search / "dev_ids.json").write_text(json.dumps(dev) + "\n")
    (search / "PROTOCOL.md").write_text(PROTOCOL)
    rng = random.Random(1234)
    entries = []
    for arm in ("exact", "ppo"):
        candidates = [DEFAULTS[arm]] + [sample(arm, rng) for _ in range(ROUND1_PER_ARM - 1)]
        entries += [dict(label=f"{arm}-c{i:02d}-s1", arm=arm, params=p, seed=1) for i, p in enumerate(candidates)]
    write_round(search, "round1", entries, epochs=1, ids_note=f"dev {len(dev)} disjoint from test {len(test_ids)}")
    print(search)


def eligible(search, m):
    """Amendment 2: format intact and answers not damaged (vs the base model on dev)."""
    base = json.loads((search / "base_dev_metrics.json").read_text())["accuracy"]
    return m["wrong_format_rate"] <= 0.02 and m["accuracy"] >= base - 0.02


def results(search, name):
    """{label: metrics at the round's last snapshot}, plus the configs."""
    configs = json.loads((search / f"{name}_configs.json").read_text())
    curve = json.loads((search / name / "curve.json").read_text())
    last = {label: points[max(points, key=int)] for label, points in curve.items()}
    return configs, last


def select(search, name):
    configs, last = results(search, name)
    groups = {}
    for label, cfg in configs.items():
        key = label.rsplit("-s", 1)[0]
        if label in last:
            groups.setdefault(key, dict(arm=cfg["arm"], params=cfg["params"], runs=[]))["runs"].append(last[label])
    table = []
    for key, g in groups.items():
        mean = lambda k: statistics.fmean(r[k] for r in g["runs"])
        table.append(dict(config=key, arm=g["arm"], params=g["params"], seeds=len(g["runs"]),
                          eligible=all(eligible(search, r) for r in g["runs"]), accuracy=mean("accuracy"),
                          wrong_format=mean("wrong_format_rate"), brier=mean("brier"),
                          ece=mean("ece"), auroc=mean("auroc")))
    table.sort(key=lambda r: (r["arm"], not r["eligible"], r["brier"]))
    (search / f"{name}_ranking.json").write_text(json.dumps(table, indent=1) + "\n")
    for r in table:
        print(f"{r['arm']:5s} {r['config']:12s} {'ok ' if r['eligible'] else 'OUT'} acc {r['accuracy']:.3f} "
              f"fmt {r['wrong_format']:.3f} brier {r['brier']:.4f} ece {r['ece']:.3f} auroc {r['auroc']:.3f} {r['params']}")
    missing = [l for l in configs if l not in last]
    if missing:
        print("missing results:", missing)
    nxt = None  # amended protocol: round 1 only sets the start point (see `start`)
    if not nxt:
        return
    entries = []
    for arm in ("exact", "ppo"):
        ranked = [r for r in table if r["arm"] == arm]
        keep = ranked[:ROUND2_KEEP] if nxt == "round2" else ranked[:1]
        seeds = ROUND2_SEEDS if nxt == "round2" else FINAL_SEEDS
        entries += [dict(label=f"{r['config']}-s{s}", arm=arm, params=r["params"], seed=s) for r in keep for s in seeds]
    write_round(search, nxt, entries, epochs=2, ids_note="dev")


def sweep_points(arm, current, stage):
    """The stage's grid for one arm: a list of parameter dicts, always including `current`."""
    vary = lambda **kw: {**current, **kw}
    if stage in ("stage1", "stage3"):
        return [vary(lr=float(f"{current['lr'] * m:.3g}")) for m in (1 / 3, 1 / 2, 1, 2, 3)]
    if stage == "stage2":
        return [vary(updates=u) for u in (1, 2, 4, 8)]
    if stage == "stage4" and arm == "exact":
        w, t = current["hinge_weight"], current["hinge_threshold"]
        points = [vary(hinge_weight=float(f"{w * m:.3g}")) for m in (1 / 3, 1, 3)]
        return points + [vary(hinge_threshold=x) for x in (0.9, 0.95, 0.99) if x != t]
    if stage == "stage4" and arm == "ppo":
        c, v = current["cliprange"], current["vf_coef"]
        points = [vary(cliprange=x) for x in (0.1, 0.2, 0.3)]
        if c not in (0.1, 0.2, 0.3):
            points.append(current)
        return points + [vary(vf_coef=float(f"{v * m:.3g}")) for m in (1 / 3, 3)]
    raise ValueError(stage)


def start(search):
    table = json.loads((search / "round1_ranking.json").read_text())
    current = {arm: next(r["params"] for r in table if r["arm"] == arm and r["eligible"]) for arm in ("exact", "ppo")}
    (search / "current.json").write_text(json.dumps(current, indent=1) + "\n")
    (search / "history.jsonl").write_text(json.dumps(dict(stage="round1", current=current)) + "\n")
    print(json.dumps(current, indent=1))


def sweep(search, stage):
    current = json.loads((search / "current.json").read_text())
    entries = []
    for arm in ("exact", "ppo"):
        unique = []
        for p in sweep_points(arm, current[arm], stage):
            if p not in unique:
                unique.append(p)
        entries += [dict(label=f"{arm}-{stage}-p{i}-s1", arm=arm, params=p, seed=1) for i, p in enumerate(unique)]
    write_round(search, stage, entries, epochs=1, ids_note="dev")


def advance(search, stage):
    configs, last = results(search, stage)
    current = json.loads((search / "current.json").read_text())
    rows = []
    for arm in ("exact", "ppo"):
        scored = sorted((not eligible(search, last[l]), last[l]["brier"], l) for l, c in configs.items()
                        if c["arm"] == arm and l in last)
        for out, brier, label in scored:
            m = last[label]
            rows.append(dict(stage=stage, arm=arm, label=label, params=configs[label]["params"], eligible=not out,
                             accuracy=m["accuracy"], wrong_format=m["wrong_format_rate"], brier=brier, ece=m["ece"],
                             auroc=m["auroc"], ece_unsampled=m.get("ece_unsampled"), auroc_unsampled=m.get("auroc_unsampled")))
            print(f"{arm:5s} {label:22s} {'OUT' if out else 'ok '} acc {m['accuracy']:.3f} brier {brier:.4f} "
                  f"ece {m['ece']:.3f} auroc {m['auroc']:.3f} {configs[label]['params']}")
        if scored and not scored[0][0]:  # keep the current point if nothing is eligible
            current[arm] = configs[scored[0][2]]["params"]
    missing = [l for l in configs if l not in last]
    if missing:
        print("missing results:", missing)
    (search / f"{stage}_ranking.json").write_text(json.dumps(rows, indent=1) + "\n")
    (search / "current.json").write_text(json.dumps(current, indent=1) + "\n")
    with open(search / "history.jsonl", "a") as f:
        f.write(json.dumps(dict(stage=stage, current=current)) + "\n")
    print("current:", json.dumps(current))


def final(search):
    current = json.loads((search / "current.json").read_text())
    entries = [dict(label=f"{arm}-final-s{s}", arm=arm, params=current[arm], seed=s) for arm in ("exact", "ppo") for s in FINAL_SEEDS]
    write_round(search, "final", entries, epochs=2, ids_note="dev, then test once")


def brier_configs(search, name, mixes, seeds, epochs=2):
    """Exact + hinge at the tuned config with reward = (1 - m) log + m Brier."""
    current = json.loads((search / "current.json").read_text())["exact"]
    configs = {f"exact-brier{m:.2f}-s{s}": dict(arm="exact", params={**current, "brier_mix": m}, seed=s, epochs=epochs,
                                              command=command("exact", current, s, epochs) + ["--reward-mix", str(m)])
               for m in mixes for s in seeds}
    (search / f"{name}_configs.json").write_text(json.dumps(configs, indent=1) + "\n")
    print(f"wrote {len(configs)} configs to {name}_configs.json")


def choose_brier(search, rounds):
    """Lowest mean dev Brier per mix over all seeds; disqualified if any seed is ineligible."""
    by_mix = {}
    for name in rounds:
        configs = json.loads((search / f"{name}_configs.json").read_text())
        curve = json.loads((search / name / "curve.json").read_text())
        for label, cfg in configs.items():
            m = cfg["params"]["brier_mix"]
            r = curve[label][max(curve[label], key=int)] if label in curve else None
            by_mix.setdefault(m, []).append(r)
    table = []
    for m, rows in sorted(by_mix.items()):
        ok = all(r is not None and eligible(search, r) for r in rows)
        scored = [r for r in rows if r is not None]
        mean = lambda k: statistics.fmean(r[k] for r in scored)
        table.append(dict(mix=m, seeds=len(rows), all_eligible=ok, brier=mean("brier"), ece=mean("ece"), auroc=mean("auroc")))
        print(f"m={m:.2f} seeds {len(rows)} {'ok ' if ok else 'OUT'} brier {mean('brier'):.4f} ece {mean('ece'):.3f} auroc {mean('auroc'):.3f}")
    best = min((r for r in table if r["all_eligible"]), key=lambda r: r["brier"])
    (search / "brier_choice.json").write_text(json.dumps(dict(table=table, best=best["mix"]), indent=1) + "\n")
    print("best mix:", best["mix"])
    return best["mix"]


if __name__ == "__main__":
    if sys.argv[1] == "init":
        init()
    elif sys.argv[1] == "select":
        select(Path(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "start":
        start(Path(sys.argv[2]))
    elif sys.argv[1] == "sweep":
        sweep(Path(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "advance":
        advance(Path(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "final":
        final(Path(sys.argv[2]))
    elif sys.argv[1] == "brier-configs":
        brier_configs(Path(sys.argv[2]), sys.argv[3], [float(x) for x in sys.argv[4].split(",")],
                      [int(x) for x in sys.argv[5].split(",")])
    elif sys.argv[1] == "choose-brier":
        choose_brier(Path(sys.argv[2]), sys.argv[3].split(","))
