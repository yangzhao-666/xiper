#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0-2
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
#AGENT="Viper"
AGENT="Viper_ground"
#AGENT="XViper_separateOT_expert_data_best_ot_ground"
#AGENT="OracleViper"

# Define tasks and seeds
TASKS=(
#    "dmc_cartpole_balance"
    "dmc_quadruped_walk"
#"dmc_cup_catch"
#    "dmc_cheetah_run"
#"dmc_finger_spin"
#    "dmc_pointmass_easy"
#"dmc_hopper_stand"
#"dmc_walker_walk"
#    "dmc_cartpole_swingup"
#"dmc_pendulum_swingup"
#    "dmc_reacher_easy"
)

SEEDS=(7 8 9)

# Determine current task/seed from SLURM_ARRAY_TASK_ID
INDEX=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( INDEX / 3 ))
SEED_IDX=$(( INDEX % 3 ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Run training
echo "Running TASK=$TASK | AGENT=$AGENT | SEED=$SEED"

TASK_SHORT=${TASK#dmc_} 

python scripts/train_dreamer.py \
  --configs dmc_vision videogpt_prior_rb \
  --task=$TASK \
  --reward_model=dmc_clen16_fskip4 \
  --reward_model_use_ot=False \
  --reward_model_ot_path="/home/zyang/XViper/OT/checkpoints/mse/${TASK_SHORT}/SN_TN_64/T_8999.pth" \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1

