# Reinforcement Learning from Cross-domain Videos with Video Prediction Models

Accepted at ECML-PKDD, 2026

This is the official implementation of the Cross-domain Video Prediction Rewards (XIPER) agent, an RL agent trained with reward signals provided by a cross-domain video prediction model.

---

## Installation

XIPER is based on [VIPER](https://github.com/Alescontrela/viper_rl/) and [NOT](https://github.com/iamalexkorotin/NeuralOptimalTransport/).  
Please install the dependencies from both repositories before proceeding.

### Overview

XIPER first trains a **domain translation model** and a **video prediction model**, then trains an **RL agent**.  

| Component            | Model      | Data                                               |
|----------------------|------------|----------------------------------------------------|
| Domain Translation   | NOT        | Random in Expert (E) domain, Random in Agent (A) domain |
| Video Prediction     | VideoGPT   | Expert demonstrations in E domain                  |
| RL Agent             | DreamerV3  | Online interaction                                 |

> *E domain = expert domain where demonstrations are collected.  
> A domain = agent domain where the RL agent learns.*

After installing XIPER, the repository should look like:

```
XViper/
  |- OT/                    # scripts for training the OT model (domain translation model)
  |- viper_rl/
    |- dreamerv3/           # DreamerV3 agent
    |- videogpt/            # VideoGPT model (video prediction model)
  |- collect_data_*.py      # scripts for collecting random data for OT
  |- *.sh                   # SLURM job launch scripts
```

The domain translation model logic is implemented in:  
`viper_rl/videogpt/reward_models/videogpt_reward_model.py`.

### Submission Supplementary

The provided zip file includes:
- Pre-trained NOT model
- Pre-trained VideoGPT model
- Training code for both models
- Code for running XIPER with pre-trained models

To get started quickly, you can use the pre-trained models instead of training from scratch.

---

## Train Domain Translation Model (OT)

### 1. Data Collection

Two datasets are required, each containing visual observations only: one from the expert domain and one from the agent domain. You can run `collect_data_atari.py` and `collect_data_dmc.py` to collect random data for Atari and DMC tasks. Make sure you specify:

- task name
- number of episodes
- `output_root`

Data is saved as a `.h5` file (images only).

**For new tasks:**
- Save only visual observations (images) to `.h5` with key `"frames"`.
- Image size must be 64x64.
- At least 50,000 images per dataset.

### 2. Train OT Model

```bash
cd OT/
python train_ot.py
```

- Set dataset paths in `train_ot.py`.  
- Logs (translated examples, FID scores) are sent to your wandb (configure your own).  
- After training, choose the best checkpoint based on FID scores.  

Approximate training time: 24h for 10k steps with ~50k images.  

For a quick start, use the pre-trained NOT model included.

---

## Train Video Prediction Model (VideoGPT)

Training VideoGPT is resource-intensive (e.g., 4xA100 GPUs for several days).  
If your task is the same as in [VIPER](https://github.com/escontra/viper_rl) or [Diffusion Reward](https://github.com/TEA-Lab/diffusion_reward), you can directly use their released checkpoints.

### Quick Start (Reuse VIPER Checkpoint)

```bash
python -m viper_rl_data.download checkpoint dmc
```

This downloads DeepMind Control Suite checkpoints to:  
`<VIPER_INSTALL_PATH>/viper_rl_data/checkpoints/`

Checkpoints can be accessed via:  
```python
from viper_rl_data import VIPER_CHECKPOINT_PATH
```

### Re-train Video Prediction Model

Follow the instructions [here](https://github.com/escontra/viper_rl?tab=readme-ov-file#video-model-training).

---

## Policy Training

### 1. Environment Modifications

Before training the RL agent, modify the environment to simulate domain shifts:

- **DMC Color Suite:**  
  Change agent body color in `materials.xml`:  
  ```xml
  [0.7, 0.5, 0.3, 1] -> [0.3, 0.3, 0.3, 1]
  ```

- **DMC Quadruped:**  
  Fix starting orientation in `quadruped.py`:  
  ```python
  orientation = np.array([0.0, 0.0, 0.0, 1.0])
  ```

- **DMC Body Suite:**  
  - Cheetah body size: `size=0.046 -> 0.066` (in `cheetah.xml`)  
  - Cartpole pole length: `size=0.045 -> 0.095` (in `cartpole.xml`)

- **Atari:**  
  Convert tasks to grayscale (`gray=True`).  
  Pixel state: `(64, 64, 1)`  expand to `(64, 64, 3)` for OT.  
  Patch `imageio/plugins/pillow.py` (line 433):  
  ```python
  ndimage = np.repeat(ndimage, 3, axis=-1)
  ```


### 2. Train DreamerV3 Agent with XIPER

```bash
python scripts/train_dreamer.py   --configs=dmc_vision videogpt_prior_rb   --task=dmc_cartpole_balance   --reward_model=dmc_clen16_fskip4   --reward_model_use_ot=True   --reward_model_ot_path=./pretrained_NOT_cartpole.pth   --logdir=./logdir
```

**Key arguments:**
- `--reward_model_use_ot=True` # use domain translation model  
- `--reward_model_ot_path=<PATH_TO_OT_MODEL>` # path to OT model  
- `--reward_model=<PATH_TO_VIDEOGPT_MODEL>` # path to VideoGPT model  

Check available reward models in:  
`viper_rl/videogpt/reward_models/__init__.py`

---

## Citation

*(To be added after acceptance)*
