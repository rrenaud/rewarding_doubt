# Hyperparameter search protocol (fixed before any result)

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

## Amendment (2026-10-02T05:33Z, before any round-1 result was seen)

Round 1 runs as planned, but rounds 2 and final are replaced by greedy one-dimensional sweeps, which give
per-parameter curves and fit the 10-GPU limit (5 points per arm = one wave).

1. Start: each arm's best round-1 config by dev Brier (round 1 only sets the starting point).
2. Four sweep stages, both arms in parallel, 128 steps (1 epoch), seed 1, scored on dev. Each stage
   re-runs the current point and keeps the value with the lowest dev Brier:
   - stage1: lr x {1/3, 1/2, 1, 2, 3}
   - stage2: updates per batch (exact passes / PPO ppo_epochs) in {1, 2, 4, 8}
   - stage3: lr x {1/3, 1/2, 1, 2, 3} again (lr and updates interact)
   - stage4: exact: hinge weight x {1/3, 1, 3} and threshold {0.9, 0.95, 0.99} (two lines through the
     current point); PPO: cliprange {0.1, 0.2, 0.3} and vf_coef x {1/3, 3} (two lines through it).
3. Final: each arm's final config, 2 epochs (256 steps), fresh seeds 4, 5, 6, scored on dev and once on test.

## Amendment 2 (2026-10-02T05:53Z, after seeing round-1 results)

Round 1 exposed a loophole in "lowest dev Brier": the released evaluation drops malformed replies, and
some high-learning-rate runs damaged the answers themselves (e.g. dev accuracy 9.5% with Brier 0.092, or
94-98% malformed replies). Brier rewarded them. Selection now only considers **eligible** configs:
- dev wrong-format rate <= 2%, and
- dev answer accuracy >= base-model dev accuracy - 0.02 (base: 0.660, so >= 0.640).
Among eligible configs, lowest dev Brier as before. This applies to round 1's start points and to every
sweep stage. Three round-1 runs (exact-c04, ppo-c14, ppo-c15) have no metrics: their outputs were so
malformed that the unsampled readout crashed on zero rows (fixed); they are treated as ineligible.

## Follow-ups (2026-10-02T08:08Z)

- final_ppo_f1: the tuned PPO config retrained with F1 > 0.5 grading in its reward (matching the
  discrete-exact arm and the evaluation), seeds 4-8, 256 steps, dev and test. Removes the
  training-label confound documented in docs/grading.md.
- brier_sweep (exploratory, dev only): exact + hinge at the tuned config, reward = (1 - m) log score +
  m Brier score for m in {0, 0.25, 0.5, 0.75, 1}, seeds 1-2, on the first 512 training questions
  (2 epochs = 128 steps).

## Brier follow-up (2026-10-02T08:47Z, before brier_sweep2 results)

- brier_sweep2: exact + hinge at the tuned config, 512 training questions, 128 steps, dev only;
  m in {0, 0.3, 0.4, 0.5, 0.6}, seeds 3 and 4 (pooled with brier_sweep seeds 1-2 where m matches).
- Choose m: lowest mean dev Brier over all of its seeds; a value is disqualified if any seed is
  ineligible (format or answer damage, as in amendment 2).
- brier_full: the chosen m and m = 0, full training set, 256 steps, seeds 6-8, dev and test.
