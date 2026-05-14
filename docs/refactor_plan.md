# Refactor plan

## Phase 1: Repository cleanup

- Replace minimal README with paper-aware documentation.
- Add citation metadata.
- Add requirements and `.gitignore`.
- Move original root-level scripts into `legacy/`.
- Add a compact result/configuration summary.

## Phase 2: Modularize the simulator

Refactor `VehicularMobilityTools.py` into modules such as:

```text
src/pandsvn_drl/
├── mobility.py
├── communication.py
├── queues.py
├── environment.py
└── policies.py
```

Important cleanup tasks:

- Remove interactive `input(...)` calls from constructors.
- Use dataclass-based configuration.
- Remove module-level demo code.
- Add deterministic random seeds.
- Add unit tests for mobility, queue update, and link-state calculations.

## Phase 3: Refactor DQN training

- Replace TensorFlow session-style utilities with TensorFlow 2 / Keras code.
- Add a single `train_dqn.py` entry point.
- Add config files for paper settings.
- Save training logs to `outputs/`.

## Phase 4: Reproducibility

- Add CLI commands for DQN_7, CB_3, and MDP_3 experiments.
- Add result plotting scripts.
- Document how to reproduce Figures 5 and 6 from the paper.
