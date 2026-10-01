# Baseline-matched exact objectives, 3 seeds (Llama-3-8B, Modal)

Single-pass exact objectives on the released code's setup, regularized exactly like the
released PPO. Every setting below is shared with `Train.py`:

- 4-bit Llama-3-8B-Instruct with LoRA r=8, on-policy answers, batch 8, lr 1e-5, 2 epochs over our 1,024 questions.
- **8 Adam updates per batch** (4 passes × 2 minibatches).
- **The released reward scale:** ×10, a +2.5 bonus for a correct answer, and **−30 for invalid output**. It is applied
  exactly, without renormalizing over the 11 levels. Not stopping after the number is charged via the
  sampled-number estimate.
- **KL to the base model**, β = 0.05, with TRL's adaptive controller (target 6, horizon 10,000).

The only remaining difference from PPO is how the confidence is scored. Here all 11 levels come from
one forward pass (`core.baseline_matched_objective`, `exact_llama.py --scoring single
--regularization baseline`). PPO scores one sampled string and uses a value head.

Each checkpoint was scored with the released evaluation on the same 512 held-out questions. The
evaluation also reports the new **unsampled** columns: confidence read as the mean of the 11-level
softmax at the step that emitted the number, on the same rows and labels.

## Result at the end of training (step 256), mean ± sd over seeds 1–3

| | ECE ↓ (sampled) | AUROC ↑ (sampled) | ECE ↓ (unsampled) | AUROC ↑ (unsampled) | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Released PPO (2 identical runs) | 0.074, 0.081 | 0.771, 0.732 | 0.085 (run 1) | 0.785 (run 1) | 0.188 (run 1) |
| **Discrete-exact, baseline-matched** | 0.085 ± 0.013 | **0.808 ± 0.017** | 0.082 ± 0.011 | 0.833 ± 0.014 | **0.174 ± 0.008** |
| Fractional, baseline-matched | 0.170 ± 0.023 | 0.783 ± 0.004 | 0.090 ± 0.005 | **0.840 ± 0.009** | 0.204 ± 0.010 |

For comparison, the hinge versions (1 seed, 11-string scoring, 8 updates) ended at 0.055 / 0.809
(discrete-exact) and 0.072 / 0.783 (fractional). The base model is at 0.333 / 0.560.

- **Discrete-exact matches PPO on calibration and beats it on ranking.** Its ECE is within the PPO runs'
  spread (0.085 ± 0.013 vs 0.074 and 0.081). Its AUROC is higher in every seed (0.793–0.826 vs
  0.732–0.771), and its Brier score is lower. All of this comes with the same reward, the same −30
  penalty, the same KL and the same update count.
- **Fractional calibrates its mean, not its sample.** Read from the mean of π, its ECE is 0.090 ± 0.005,
  about the same as discrete-exact, and its AUROC is the best of all (0.840). The integer it writes
  under the paper's T = 0.6 sampling is badly calibrated (0.170 ± 0.023). That is what its objective
  predicts: it targets the mean confidence and lets π stay spread out, and sampling at T = 0.6 sharpens
  that spread. In the earlier hinge run its sampled ECE was 0.072, so the regularization regime also
  matters.
- **The KL binds, but only lightly.** Mean KL to the base model sat at 5–6.4 nats per batch, near TRL's
  target of 6. β barely moved (0.0492–0.0500), matching PPO's behavior. Discrete-exact's sampled ECE
  (0.085 ± 0.013) is higher than the single hinge run's 0.055. The KL pull toward the overconfident base
  model may cost some calibration, but one hinge seed cannot establish that.
- **It is no slower than PPO.** The runs took 7.4–10.9 s per step (38–54 min for 256 steps), including
  the reference pass and confidence sampling, against PPO's 12 s per step (52 min).

## Format

- **Evaluation:** wrong-format rates were 0.0–0.6% at every checkpoint, with number mass 1.000.
- **Training was looser.** About 4–6% of the T = 1 confidence samples were not ": <number>". The canonical
  ": <number>" mass averaged about 0.94 per batch, and in 95 of 256 batches of seed 1 it fell below 0.95.
  The −30 penalty and KL do not hold the canonical format as tightly as the hinge did. The paper's
  evaluation samples at T = 0.6 and does not see it. **Follow-up:** log the non-number continuations to
  see what they are (alternative spacing, words, or " Confidence" appearing inside the answer).

## Caveats

- Three seeds for the exact methods, but only two PPO runs, and the PPO unsampled columns come from one
  run. A matched set of 3 PPO seeds would make the comparison symmetric (about \$5).
- The 512 held-out questions give paired evaluation intervals of about ±0.04 on each metric for a single
  run. The seed spread above covers training noise as well.
- Snapshots every 64 steps; no curve-based selection is used, only the end of training.

Files: `curve.json` (all metrics per snapshot), `*/timing.jsonl` (per-step training metrics),
`*/eval_step*.json` (per-question outputs). Adapters are on the Modal volume under
`outputs/baseline-matched-20261001T221749Z/`.
