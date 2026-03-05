# LLM-Assisted Semantically Diverse Teammates Generation for Efficient Multi-agent Coordination

This repository contains the official implementation of SemDiv, a framework that leverages Large Language Models (LLMs) to generate semantically diverse teammates for efficient multi-agent coordination. The approach is evaluated on multiple multi-agent environments, including Level-Based Foraging (LBF), Predator-Prey (PP), StarCraft Multi-Agent Challenge V2 (SMACv2), and Google Research Football (GRF).

## Environment Installation

To set up the required environments, follow the steps below.

1. Install the Level-Based Foraging (LBF) Environment

```
pip install -e pymarl/src/envs/lb-foraging
```

2. Install the Predator-Prey (PP) Environment

```
pip install -e pymarl/src/envs/mpe/multi_agent_particle
```

3. Install the StarCraft Multi-Agent Challenge V2 (SMACv2) Environment

```
pip install -e pymarl/src/envs/smacv2
```

4. Install the Google Research Football (GRF) Environment

Due to size constraints, we have removed the files in:
 - football/gfootball_engine
 - football/third_party/gfootball_engine

To use the GRF environment, manually download these files from the official GRF repository and place them in the respective directories. Then, run:

```
pip install -e football
```

5. Install the HARL Repository

```
pip install -e HARL
```

## Running an Experiment

To run an experiment with SemDiv, follow these steps:

1. Run the training process of SemDiv

```
cd language
python semdiv.py
```

The target environment (LBF, PP, SMACv2, or GRF) can be set in this script.

2. Evaluate the Trained Policies

```
cd pymarl
python src/scripts/test.py
```

for LBF, PP, SMACv2, or

```
cd HARL/example
python test.py
```

for GRF.

3. Run the Head Selection Process

To perform the head selection process, run:

```
cd language
python selection.py
```

## Publication

If you find this repository useful, please cite our paper:

```
@inproceedings{semdiv,
  title     = {LLM-Assisted Semantically Diverse Teammate Generation for Efficient Multi-agent Coordination},
  author    = {Lihe Li and Lei Yuan and Pengsen Liu and Tao Jiang and Yang Yu},
  booktitle = {Proceedings of the Forty-second International Conference on Machine Learning},
  year      = {2025}
}
```
