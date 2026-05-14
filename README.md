# Deep Reinforcement Learning for Joint Sensing and Processing in Vehicular Networks

This repository contains research code and documentation for the paper:

> Rajarshi Chattopadhyay and Chen-Khong Tham, **"Joint Sensing and Processing Resource Allocation in Vehicular Ad-Hoc Networks,"** *IEEE Transactions on Intelligent Vehicles*, Vol. 8, No. 1, January 2023.

The project studies how a smart vehicle can improve its field of view and application quality by using sensing and processing resources from neighbouring smart vehicles over unreliable vehicular links. The work formulates joint sensing/processing as a sequential decision-making problem and evaluates **Contextual Bandit (CB)**, **Markov Decision Process (MDP)**, and **Deep Q-Network (DQN)** policies.

## Project idea

A user smart vehicle may request information from neighbouring smart vehicles. Each neighbour can either:

- not sense,
- process sensor data locally and send the result, or
- send raw sensor data to the user vehicle for local processing.

The decision must account for:

- field-of-view / region-of-interest utility,
- queueing and processing constraints,
- V2V link uncertainty,
- cellular fallback,
- task deadlines,
- sensing cost.

## Main components

The original code currently includes modules for:

- vehicular mobility simulation,
- V2V/cellular communication modelling,
- queue and deadline handling,
- replay memory,
- DQN model utilities,
- training-loop utilities.

The original research scripts are preserved as legacy code. A cleaner TensorFlow 2 style scaffold is provided under `src/pandsvn_drl/` to support future refactoring.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── docs/
│   ├── original_repository_state.md
│   └── refactor_plan.md
├── results/
│   ├── README.md
│   └── experiment_summary.csv
├── scripts/
│   └── smoke_test.py
├── src/
│   └── pandsvn_drl/
│       ├── __init__.py
│       ├── config.py
│       ├── dqn_model.py
│       └── replay_buffer.py
└── legacy/
    └── README.md
```

Recommended cleanup: move the original root-level code files into `legacy/`:

```bash
mkdir -p legacy
git mv VehicularMobilityTools.py environment_utils.py memory_utils.py model_utils.py legacy/
```

Then commit the update.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

The repository includes a lightweight smoke test that verifies the cleaned DQN model and replay buffer scaffold:

```bash
python scripts/smoke_test.py
```

This does **not** reproduce the paper experiments. It only checks that the refactored scaffolding imports and runs.

## Experiment setting from the paper

The paper evaluated joint sensing and processing policies in a three-lane highway simulation with:

- 100 km highway stretch,
- user smart vehicle in the middle lane,
- neighbouring smart vehicles entering each lane over time,
- freeway mobility model with speeds between 40 km/h and 100 km/h,
- up to 8 regions of interest (RoIs),
- DQN policy scaled to 7 neighbouring vehicles and 8 RoIs,
- training for 1,950 episodes with 600 time steps per episode,
- comparison against local-processing and offload-only baselines.

## Results summary

The paper reports that:

- CB and MDP policies work for small settings but do not scale well with the number of neighbouring vehicles and RoIs.
- The DQN policy scales better and can optimize sensing/processing decisions over a larger FoV and more neighbouring vehicles.
- The DQN-based scheme achieves higher total utility than local-processing-only and offload-only baselines under varied channel and deadline conditions.
- A guided exploration strategy improves DQN validation performance compared with purely random epsilon-greedy exploration.

See `results/experiment_summary.csv` for a compact summary of the reported experimental configurations.

## Citation

```bibtex
@article{chattopadhyay2023joint,
  title={Joint Sensing and Processing Resource Allocation in Vehicular Ad-Hoc Networks},
  author={Chattopadhyay, Rajarshi and Tham, Chen-Khong},
  journal={IEEE Transactions on Intelligent Vehicles},
  volume={8},
  number={1},
  pages={616--627},
  year={2023},
  publisher={IEEE},
  doi={10.1109/TIV.2021.3124208}
}
```

## Status

This repository is being cleaned and refactored from the original research prototype. The current release is intended as a code companion and documentation archive. A fully reproducible training pipeline will require further refactoring of the original scripts into a unified experiment runner.
