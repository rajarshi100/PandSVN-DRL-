# PandSVN-DRL: Joint Sensing and Processing in Vehicular Networks

This repository contains a cleaned code companion for the IEEE TIV work:

> **Joint Sensing and Processing Resource Allocation in Vehicular Ad-Hoc Networks**  
> Rajarshi Chattopadhyay and Chen-Khong Tham, IEEE Transactions on Intelligent Vehicles, 2023.

The project studies how a smart vehicle can use neighbouring vehicles for sensing and processing under unreliable V2V links, queueing delay, and task deadlines. The paper formulates the problem using contextual bandits (CB), Markov decision processes (MDP), and deep Q-networks (DQN), and compares them with local-processing and offload-only baselines.

## What this version contains

- A small, import-safe Python package under `src/pandsvn_drl/`.
- A TensorFlow 2 DQN agent implementation.
- A lightweight gym-style vehicular sensing/processing simulator for smoke tests and future extension.
- A replay buffer, configuration object, metrics utilities, and training script.
- Original code should be preserved under `legacy/` if available.

## Paper setup summarized

The paper considers:

- A smart vehicle seeking sensing information from regions of interest (RoIs) outside its own field of view.
- Two action types for each neighbouring vehicle: request processed result or request raw sensor data.
- V2V/cellular communication uncertainty, task queues, processing deadlines, and location-dependent utility.
- CB, MDP, and DQN decision policies.
- A DQN setup that scales to **7 neighbouring vehicles** and **8 RoIs** in a **100-km three-lane highway** simulation.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── CITATION.cff
├── src/pandsvn_drl/
│   ├── __init__.py
│   ├── config.py
│   ├── dqn.py
│   ├── environment.py
│   ├── memory.py
│   ├── metrics.py
│   └── train.py
├── scripts/
│   └── smoke_test.py
├── results/
│   └── experiment_summary.csv
├── docs/
│   └── refactor_notes.md
└── legacy/
    └── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

```

For a quick import/simulation check without TensorFlow:

```bash
python scripts/smoke_test.py
```

For a short DQN run:

```bash
python -m pandsvn_drl.train --episodes 5 --steps-per-episode 100
```

## Notes on original scripts

The original repository files were preserved as a research-code archive. Some of them were formatted as very long single-line Python files and included module-level test code. This refactor separates reusable package code from executable scripts.

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
  publisher={IEEE}
}
```
