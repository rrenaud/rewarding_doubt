# Credit assignment: {PPO, discrete-exact} × {KL, hinge}, 3 seeds each (Llama-3-8B, Modal)

The question: discrete-exact with the hinge clearly beats the released PPO. Is that because it
scores the **full numeric distribution**, or because it uses the **hinge instead of KL**?

## Arms

All arms share the released `Train.py` setup:

- 4-bit Llama-3-8B-Instruct with LoRA r=8.
- On-policy answers, batch 8, 2 epochs over our 1,024 questions, lr 1e-5.
- 8 Adam updates per batch.
- Seeds 1–3, scored with the released evaluation on the same 512 held-out questions.

The four arms:

- **PPO + KL**: `Train.py` unmodified, with the seed actually varied. The earlier two PPO runs
  both used `Train.py`'s hard-coded seed 2.
- **PPO − KL + hinge**: the same, with `init_kl_coef=0` and the adaptive controller off. The
  policy loss gains the same two hinges discrete-exact uses (`: <number>` mass ≥ 0.95, stop after
  the sampled number ≥ 0.95), computed from the logits PPO already produces for its sampled
  response. Format is otherwise handled by PPO's own −30 reward. See `subset.py add_ppo_hinge`.
- **Exact + KL**: the [baseline-matched run](../modal-baseline-matched-20261001T221749Z/report.md)
  (released reward, −30 penalty and KL, computed exactly).
- **Exact + hinge**: the `de-hinge` arm of
  [exact-variants](../modal-exact-variants-20261002T000216Z) (paper-scale reward, hinges, no KL).

## End of training (step 256), mean ± sd over 3 seeds

| ECE ↓ / AUROC ↑ (Brier ↓) | KL + −30 | Hinge, no KL | Effect of hinge |
|---|---|---|---|
| **PPO** | 0.127 ± 0.056 / 0.706 ± 0.019 (0.217) | 0.070 ± 0.037 / 0.744 ± 0.015 (0.190) | ECE −0.057, AUROC +0.038 |
| **Discrete-exact** | 0.085 ± 0.013 / 0.808 ± 0.017 (0.174) | 0.057 ± 0.008 / 0.789 ± 0.026 (0.174) | ECE −0.028, AUROC −0.019 |
| **Effect of full distribution** | ECE −0.042, AUROC +0.102 | ECE −0.013, AUROC +0.045 | |

Per-seed ECE: PPO+KL 0.073 / 0.183 / 0.124; PPO−KL+hinge 0.059 / 0.111 / 0.040;
exact+KL 0.073 / 0.098 / 0.083; exact+hinge 0.065 / 0.050 / 0.055. Answer accuracy is 63.4–64.1%
in every arm (base model 63.3%).

## Findings

- **The regularization change earns most of the calibration gain.** Swapping KL for the hinge
  lowers ECE for both estimators. With the hinge, PPO (0.070) and exact (0.057) are within PPO's
  seed noise of each other.
- **The full distribution earns most of the ranking gain**: +0.10 AUROC with KL and +0.045 with
  the hinge. It also lowers Brier in both columns.
- **The full distribution makes outcomes reliable.** Seed-to-seed ECE spread is ±0.008–0.013 for
  exact against ±0.037–0.056 for PPO, a 3–5× reduction.
- **The earlier PPO comparison was optimistic.** The two previous PPO runs (ECE 0.074, 0.081) sit
  at the good end of PPO's range. With varied seeds, PPO+KL averages 0.127.

## Caveats

- PPO's ECE standard errors are about ±0.02–0.03, so the exact-vs-PPO ECE gap under the hinge
  is not established. The AUROC gaps and the variance reduction are consistent across seeds.
- Hinge strength is not exactly matched. It is added to PPO's whitened-advantage policy loss and
  to exact's reward-scale loss with the same weight 1.0. It is inactive above 95%.
- PPO keeps its value head in both cells; the exact cells shown have none. The exact + KL +
  value-head arm reached AUROC 0.826 ± 0.002, so a value head would likely widen the AUROC gap.

Files: `curve.json`, `*/timing.jsonl` (PPO per-step timestamps), `*/eval_step*.json`. Adapters
and `hinge.jsonl` (per-minibatch hinge stats) are on the Modal volume under
`outputs/ppo-variants-20261002T021236Z/`.
