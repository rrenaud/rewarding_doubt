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
