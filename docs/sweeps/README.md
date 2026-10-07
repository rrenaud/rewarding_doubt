# Hyperparameter sweeps: index and lessons

Every hyperparameter sweep run in this project from October 1 to 7, 2026, one report per sweep. Each report gives the
question, the inputs (what was swept and what was held fixed), the results, what we learned, and two ratings.
Numbers are copied from the run logs and the documents they cite; "dev" is the 512-question selection split unless
a report says otherwise.

**Rating scales.**
- **Motivation:** ★★★ a specific hypothesis or decision with the test or rule fixed in advance; ★★ a reasonable
  question run as exploration; ★ convenience or curiosity, or a design too loose to answer its question.
- **Knowledge:** *high* — changed a default, a direction, or the project's understanding; *medium* — a useful,
  narrower or partly superseded result; *low* — little that changed later work.

## Index

| # | sweep | model, trainer | swept | motivation | knowledge | one-line result |
|---|---|---|---|---|---|---|
| [01](01-llama-hparam-search.md) | Hyperparameter search, exact vs PPO | Llama-3-8B, released pipeline | lr, updates per batch, hinge, clip, vf | ★★★ | high | Exact beats PPO on test (AUROC 0.839 vs 0.757); found the Brier loophole that rewards broken answers |
| [02](02-brier-mix-reward.md) | Brier mixed into the reward | Llama, released | mix m | ★★ | low | The small-scale pick (m = 0.5) lost at full scale; log score stays |
| [03](03-frozen-answers-lr.md) | Frozen answers and their lr | Llama, released | answer mode, lr | ★★★ | medium | Frozen answers never break but cost AUROC (0.797 vs 0.839); exact's lr plateau confirmed |
| [04](04-qwen-kl-grid.md) | KL settings grid | Qwen-2.5-3B, released + patch | KL mode × objective | ★★★ | high | PPO is not KL-limited; exact ahead at every setting; adaptive KL hurts very long runs |
| [05](05-llama-long-runs.md) | 4,000-step runs | Llama, own trainers | lr, regularizer | ★★ | medium | The 128-step-tuned lr (4e-5) diverged; 1e-5 stable; exact keeps its lead |
| [06](06-thinking-check.md) | Thinking before the confidence | Llama | arm, β | ★★ | low | No gain from a trained check; β irrelevant at this scale |
| [07](07-fast-schedule.md) | Fast-loop schedule | Qwen, fast loop | batch, lr, KL mode, bucketing | ★★★ | high | One update per 32 questions at a ~10× higher lr matches the released schedule in 5 min instead of 48 |
| [08](08-fast-optimizers.md) | Optimizers | Qwen, fast loop | Adam, Muon, Scaled AdamW, PoLoRA × lr | ★★ | medium | Nothing beats Adam at a matched lr; earlier "wins" were lr confounds; Adam's best lr 3e-4 |
| [09](09-fast-adapter-ablation.md) | Which LoRA adapters | Qwen, fast loop | projections, layer ranges | ★★ | high | One residual-writing projection or half the layers suffices; placement beats size |
| [10](10-fast-minimal-adapters.md) | Bias-vector adapters | Qwen, fast loop | adapter type, lr, layers | ★★ | medium | A vector per layer nearly matches LoRA on the proxy, but at rates that wreck answers |
| [11](11-answer-kl-weight.md) | Answer-KL penalty weight | Qwen, fast loop + minimal | W | ★★★ | high | Unpenalized adapters destroy the answers; a small penalty is nearly free |
| [12](12-offline-search-topk.md) | Offline search, top-k reference | Qwen, minimal | adapter × lr × W/target | ★★★ | medium | LoRA + W = 1 best offline; adaptive target wrong for LoRA |
| [13](13-online-trials.md) | Online trials | Qwen, minimal | adapter, penalty | ★★★ | high | Drift compounds online without the penalty; offline ranking flipped online |
| [14](14-stock-params.md) | Stock parameters (`v_proj` bias, norm gains) | Qwen, minimal | adapter, lr, mode | ★★ | medium | Hook-free `v_proj` bias ≈ hooked bias online; ~1 nat is not universally safe |
| [15](15-runpod-3k-targets.md) | 3,000-step online targets | Qwen, minimal | answer-KL target | ★★ | medium | Unbounded controller blew up at targets 0.5–1; bounded it |
| [16](16-recovery.md) | Recovering drifted answers | Qwen, minimal | lr × W | ★★ | medium | W ≥ 1 restores answers in 50 steps; nearly free at lr 3e-3 |
| [17](17-weight-floor.md) | Floor on the adaptive weight | Qwen, minimal | target × floor | ★★★ | medium–high | Floor 0.1 is the knee; the controller reduces to a fixed weight |
| [18](18-llama-small-adapters.md) | Small adapters on Llama | Llama, minimal | adapter, lr, penalty | ★★ | high | Llama loses accuracy from 0.1 nats with small adapters; Qwen's rates wreck it |
| [19](19-llama-full-lora-lr.md) | Full LoRA on Llama | Llama, minimal | lr/schedule × W | ★★★ | high | Learner not buggy; full LoRA + W = 1 holds accuracy to ~0.8 nats |
| [20](20-llama-oproj-lora.md) | `o_proj`-only LoRA | Llama, minimal | layers × lr × W | ★ | low–medium | Safe only in the late half at ≤ 1e-3, and worse than full LoRA |
| [21](21-llama-late-layers.md) | Late-layer LoRA depth | Llama, minimal | first trained layer | ★★ | high | Layers 16–31 ≥ full LoRA at a fifth of the drift, fits 48 GB |
| [22](22-llama-paper-comparison.md) | Paper comparison | Llama, minimal online, released eval | layers, steps, rank | ★★★ | high | Full validation: AUROC 0.877 (paper 0.859), ECE 0.031 (paper 0.023); longer got worse |
| [23](23-llama-lora-rank.md) | LoRA rank | Llama, minimal | rank 1–8 | ★★ | medium–high | Rank 4 = rank 8 calibration with ~8× less drift |
| [24](24-llama-r4-residual-bias.md) | Residual biases on rank-4 LoRA | Llama, minimal | bias type × bias lr | ★★ | medium | No extra capacity (worse training fit); capacity not the bottleneck |
| [25](25-llama-lr-schedule-2k.md) | lr schedule and averaging, 2,000 steps | Llama, minimal | constant/cosine, lr, averaging | ★★★ | high | Not an lr noise floor: overfitting to 8,000 repeated questions |
| [26](26-llama-cosine-gate.md) | Cosine gate, 1–2 epochs | Llama, minimal | cosine peak lr × length | ★★★ | medium | Gate failed: cosine underfits short runs; full-split runs use a constant lr |

Not covered here (comparisons of methods rather than hyperparameters): the Tinker-phase smoke, pilot, hinge and
paper-PPO runs (`docs/objectives.md`), the objective comparisons in `docs/computation_guide.html`, and the
throughput/cost measurements (`scripts/run_costs.py`, `runpod/README.md`).

## Patterns across sweeps

1. **The learning rate dominates; most other knobs are second order.** The optimizer ([08](08-fast-optimizers.md)),
   the KL mode on short horizons ([07](07-fast-schedule.md)), the reward mix ([02](02-brier-mix-reward.md)) and the
   schedule shape ([26](26-llama-cosine-gate.md)) moved results less than a factor of 2–3 in lr did. Several
   apparent wins were lr confounds: Scaled AdamW and PoLoRA "beat" Adam until Adam was run at a matched rate.
2. **The right lr scales with the update cadence and is roughly constant per question.** Fewer, larger updates need
   a higher rate ([07](07-fast-schedule.md)); our 3e-4 per 32 questions moves the weights about as far per question
   as the paper's 1e-5 with 8 updates per 8 questions.
3. **Optima do not transfer across horizon or model.** The search's 128-step lr diverged at 4,000 steps
   ([05](05-llama-long-runs.md)); Qwen's adapter rates wrecked Llama ([18](18-llama-small-adapters.md)); a 500-step
   lr went past its peak at 2,000 steps ([25](25-llama-lr-schedule-2k.md)). Retune when the horizon or model changes.
4. **The failure mode above the optimum is collapse, about 3× past it.** Runs collapse to one confidence for every
   answer (AUROC ≈ 0.5) at lr 1e-3 for LoRA, ≥ 1.6e-4 in the released pipeline, Muon at 3e-4, and with 4–8 updates
   per batch ([01](01-llama-hparam-search.md), [03](03-frozen-answers-lr.md), [08](08-fast-optimizers.md)). A
   collapsed run can have excellent ECE, so **never select on ECE alone**; Brier and AUROC catch it.
5. **Answer drift is the binding constraint, and it depends on adapter and model.** Adapters trained only on the
   confidence still move the answers ([11](11-answer-kl-weight.md)); online training compounds it
   ([13](13-online-trials.md)). A nat of answer KL is harmless on Qwen up to ~2 nats, but costs accuracy on Llama
   from 0.1 nats with small adapters ([18](18-llama-small-adapters.md)) and ~0.8 nats with full LoRA
   ([19](19-llama-full-lora-lr.md)). KL does not predict accuracy ([14](14-stock-params.md)): measure regenerated
   accuracy. Lower rank and later layers drift less for the same calibration ([21](21-llama-late-layers.md),
   [23](23-llama-lora-rank.md)).
6. **Proxies and selection rules have loopholes.** Brier rewarded broken answers ([01](01-llama-hparam-search.md));
   frozen cached answers hid drift for a whole phase ([09](09-fast-adapter-ablation.md),
   [10](10-fast-minimal-adapters.md) → [11](11-answer-kl-weight.md)); the fast proxy correlates with the released
   evaluation at r ≈ 0.75; offline rankings flipped online ([13](13-online-trials.md)). Eligibility rules
   (format, accuracy) belong in every selection, and the final comparison needs the real protocol
   ([22](22-llama-paper-comparison.md)).
7. **Capacity is rarely the bottleneck; placement sometimes is.** One projection, half the layers, rank 4, or a
   vector per layer match larger adapters ([09](09-fast-adapter-ablation.md), [10](10-fast-minimal-adapters.md),
   [23](23-llama-lora-rank.md)); extra residual biases did not even improve training fit
   ([24](24-llama-r4-residual-bias.md)). But the last quarter of Qwen's layers cannot do it alone.
8. **Data repetition limits long runs.** With 8,000 cached questions, runs overfit after about 4 passes (train Brier
   0.03 vs dev 0.17 at 8 passes) ([25](25-llama-lr-schedule-2k.md)); what looked like lr noise was memorization.
   Distinct questions (the full 87,622-question split) are the next lever, not a schedule.
9. **Many decisions rested on single seeds and small margins.** Seed-to-seed spread is about ±0.01 Brier and ±0.03
   ECE on 512 questions; greedy stages decided on 0.004–0.02 Brier ([01](01-llama-hparam-search.md)), and a 2-seed
   pick did not replicate ([02](02-brier-mix-reward.md)). GPU kernels are nondeterministic, so even same-seed reruns
   drift. The most reliable sweeps used 2–3 seeds and a rule fixed in advance.

## What to do differently

- **Fix the selection rule and the gate before the runs**, including eligibility (format, accuracy) and the margin
  that counts as a win. The best-motivated sweeps (★★★) did this, and they are also the ones whose conclusions held.
- **Two seeds minimum for any decision; three for a headline.** Treat differences below the seed spread as ties.
- **Tune at the horizon and data scale you will run**, or retune before scaling up; short-horizon tuning favours
  high learning rates.
- **Always report regenerated accuracy and answer KL next to calibration**, and Brier/AUROC next to ECE.
- **Check training fit as well as dev** (per-example dumps, `--dump-examples`): it separates underfitting,
  overfitting and capacity questions that dev curves alone conflate.
- **Reuse runs before launching:** a constant-lr run's first N steps answer any shorter constant-lr question.
