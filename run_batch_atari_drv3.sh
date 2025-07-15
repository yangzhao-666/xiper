#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0-1
#SBATCH --output=/home/zyang2/logdir/slurm/output/%A_%a.out

# Conda setu
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
AGENT="Drv3_gray"

# Define tasks and seeds
TASKS=(
    "atari_pong"
#    "atari_freeway"
    "atari_kangaroo"
#    "atari_boxing"
#    "atari_atlantis"
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
  --configs atari \
  --task=$TASK \
  --env.atari.gray=True \
  --env.atari.stripe=False \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1
