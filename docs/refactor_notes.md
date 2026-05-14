# Refactor notes

The original repository was a compact research-code archive. This refactor introduces a package structure and separates reusable modules from executable scripts.

## Current status

- `src/pandsvn_drl/environment.py` is a lightweight gym-style environment that preserves the high-level decision structure: no sensing, processed-result request, or raw-data request for each neighboring vehicle.
- `src/pandsvn_drl/dqn.py` contains a TensorFlow 2 DQN agent with target network and replay-buffer training.
- `scripts/smoke_test.py` verifies package imports, environment stepping, and replay-buffer sampling.

## Remaining work for full reproduction

- Port all details of the original V2V/cellular communication model.
- Recreate CB and MDP solvers from the paper.
- Recreate all plotting scripts for Figures 5 and 6.
- Add deterministic experiment configs matching the IEEE TIV paper.
