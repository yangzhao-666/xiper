#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0
#SBATCH --output=/home/zyang/logdir/slurm/output/%A_%a.out

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
AGENT="Viper"
#AGENT="XViper"

# Define tasks and seeds
TASKS=(
        "atari_pong"
#"atari_freeway"
)

SEEDS=(0)

# Determine current task/seed from SLURM_ARRAY_TASK_ID
INDEX=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( INDEX / 3 ))
SEED_IDX=$(( INDEX % 3 ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Run training
echo "Running TASK=$TASK | AGENT=$AGENT | SEED=$SEED"

python scripts/train_dreamer.py \
  --configs=atari videogpt_prior_rb \
  --task=$TASK \
  --reward_model=atari_clen16_fskip4_mask \
  --reward_model_use_ot=False \
  --reward_model_ot_path="/home/zyang/NeuralOptimalTransport/checkpoints/mse/atari/SN_TN_64/0_999.pt" \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1
