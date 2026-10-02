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
