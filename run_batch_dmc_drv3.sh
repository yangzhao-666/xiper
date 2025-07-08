#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0-7
#SBATCH --output=/home/zyang2/logdir/slurm/output/%A_%a.out

# Conda setup
conda deactivate
conda deactivate
conda deactivate
conda activate viper

# Module loading
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load FFmpeg/6.0-GCCcore-12.3.0

# MuJoCo setup
export MUJOCO_GL="egl"

# Agent
#AGENT="Viper"
AGENT="Drv3"

# Define tasks and seeds
TASKS=(
    "dmc_cartpole_balance"
    "dmc_cartpole_swingup"
    "dmc_cheetah_run"
    "dmc_cup_catch"
#"dmc_finger_spin"
#"dmc_finger_turn_hard"
#   "dmc_hopper_stand"
    "dmc_pendulum_swingup"
    "dmc_pointmass_easy"
#    "dmc_pointmass_hard"
    "dmc_quadruped_run"
    "dmc_quadruped_walk"
#    "dmc_reacher_easy"
#    "dmc_reacher_hard"
#    "dmc_walker_walk"
)

SEEDS=(0)

# Determine current task/seed from SLURM_ARRAY_TASK_ID
INDEX=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( INDEX / 1 ))
SEED_IDX=$(( INDEX % 1 ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Run training
echo "Running TASK=$TASK | AGENT=$AGENT | SEED=$SEED"

python scripts/train_dreamer.py \
  --configs dmc_vision \
  --task=$TASK \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1
