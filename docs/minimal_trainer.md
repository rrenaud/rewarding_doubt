# A minimal trainer for the exact objective

October 6, 2026 (PST). `modal_repro/minimal_trainer.py` trains Qwen-2.5-3B to state a calibrated confidence with
the exact objective, on the fast loop's cached answers (`docs/fast_loop_log.md`), using only Hugging Face
`transformers`, PEFT and PyTorch. Each run is on one L40S on Modal; outputs and configs are in `runs/minimal/`.

**Summary.** It matches the fast loop's dev metrics within seed noise and takes 489 ms per step instead of 611
(all-layer LoRA), or 302 instead of 442 (layers 18–35). Most of the gain is from running the 132-token prompt
prefix, shared by every row, once per batch. None of 9 seeds collapsed to a constant confidence.

## How it works

- **Model**: Qwen-2.5-3B-Instruct in bf16, PyTorch SDPA attention, no gradient checkpointing.
- **Adapters**: PEFT LoRA (rank 8, alpha 8) only on the modules and layers that train (`--lora-modules`,
  `--lora-layers`), or zero-initialized vectors added to attention or MLP outputs by forward hooks
  (`--attn-bias`, `--residual-bias`). Inputs never require grad, so backward stops at the first trained layer.
- **Scoring**: one forward pass gives the 11 confidence levels, with logits only at the positions they read.
  The prompt prefix runs once per batch with grad; its key/value cache is expanded over the batch and only
  the answers run.
- **Same as the fast loop**: the objective (`baseline_matched_objective`, fixed KL weight 0.05), Adam at 3e-4,
  batches of 32 length-bucketed questions, 300 steps, dev metrics every 50 steps.

## Checks

**Reference levels.** The cache's reference log-probabilities came from the fast loop's 4-bit model and differ
from bf16 by up to 10 nats, so `minimal_trainer.py ref` recomputed them (26 s). Untrained bf16 model on the 507
dev answers: Brier 0.516, ECE 0.539, AUROC 0.816 (the 4-bit model: 0.566, 0.573, 0.787).

**Shared prefix.** In fp32 it matches a full forward to 0.003 nats, the same as a batched full forward against
one row at a time. In bf16 both comparisons differ by up to 2 nats on some rows: bf16 scoring has that much
noise however it is batched, so a bf16 bar of 0.01 cannot be met. A CPU test (`tests/test_minimal_trainer.py`,
tiny random Qwen2 with LoRA and both biases) checks levels and gradients in fp32.

## Results against the fast loop

Dev metrics of the expected confidence at step 300, mean ± sd over seeds:

| run | seeds | Brier | ECE | AUROC |
|---|---:|---|---|---|
| all-layer LoRA, minimal | 9 | 0.116 ± 0.017 | 0.048 ± 0.017 | 0.917 ± 0.010 |
| all-layer LoRA, fast loop | 3 | 0.120 ± 0.004 | 0.040 ± 0.014 | 0.915 ± 0.005 |
| attention bias 18–35 (lr 1e-2), minimal | 2 | 0.122 ± 0.008 | 0.065 ± 0.034 | 0.914 ± 0.006 |
| attention bias 18–35 (lr 1e-2), fast loop | 2 | 0.114 ± 0.002 | 0.050 ± 0.022 | 0.908 ± 0.001 |

A run takes 2.9 minutes for all-layer LoRA and 1.1 for the attention bias (from the first update, including
6 dev evaluations), against the fast loop's 3.5 and 2.7.

**Stability.** No seed collapsed to one confidence for every answer (Brier 0.32, AUROC near 0.5). The worst
dev point after step 50 is seed 3 at step 150 (Brier 0.176, AUROC 0.865), recovered by step 200. The fast
loop's seed 3 dips at step 200 too; batch order depends only on the seed, so seed 3 meets a hard stretch of
batches. An earlier 4-bit version of this trainer (commit bfbe499, since removed) collapsed on that stretch
(seed 3, steps 100–200) and on seed 2, 2 of 6 seeds; 0 of 9 here is fewer, but not conclusively (Fisher
exact p ≈ 0.14).

## Speed and memory

Milliseconds per step of 32 questions (mean over steps 11–60 of 60-step runs, `--time-steps`):

| what trains | forward | loss | backward | total | fast loop |
|---|---:|---:|---:|---:|---:|
| all-layer LoRA | 189 | 29 | 264 | **489** | 611 |
| LoRA, layers 18–35 | 136 | 24 | 138 | **302** | 442 |
| LoRA, layers 18–35, no shared prefix | 334 | 14 | 326 | 676 | |
| attention bias, layers 18–35 | 85 | 20 | 65 | **172** | |

Peak memory for one batch of 32 rows, all-layer LoRA: 14.1 GB with the shared prefix, 39.3 GB without
(weights 5.9 GB). Over a run the peak is about 33 GB, set by the longest batches. Activations take about
200 KB per token per layer, nearly half of it PEFT's fp32 copies of the LoRA inputs; the fast loop's Unsloth
kernels used less. All-layer LoRA without the shared prefix does not fit in 48 GB.

## Differences from the fast loop, and what is not done

- bf16 weights instead of 4-bit; fp32 LoRA weights where the fast loop's were bf16 (TRL casts them).
- Not carried over: gate biases, building the cache (`fast_loop.py cache` still does it), and online mode
  (answers regenerated during training).
- The loss is 32 separate calls of the objective (20–30 ms per step); not batched yet.
- All numbers are the fast loop's dev proxy, not the released evaluation.

## Reproducing

```bash
modal run modal_repro/app.py::minimal_ref         # bf16 reference cache and untrained metrics
modal run modal_repro/app.py::minimal_diagnose    # shared prefix vs full forward
modal run modal_repro/app.py::minimal_memory      # peak memory
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/validation_configs.json
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/stability_bf16_configs.json
modal run --detach modal_repro/app.py::minimal_train --configs runs/minimal/timing_configs.json
pytest tests/test_minimal_trainer.py              # needs transformers 4.48 and peft (the Modal image's pins)
```
