#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0-2
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
#AGENT="Viper_3_seeds"
#AGENT="Xiper_ot_R_fid_High"
#AGENT="XViper_separateOT_expert_data_best_ot_ground"
#AGENT="OracleViper"
AGENT="TPIL_body"

# Define tasks and seeds
TASKS=(
    "dmc_cartpole_balance"
#   "dmc_quadruped_walk"
#    "dmc_quadruped_run"
#    "dmc_cup_catch"
    "dmc_cheetah_run"
#"dmc_finger_spin"
#    "dmc_pointmass_easy"
#"dmc_hopper_stand"
#"dmc_walker_walk"
    "dmc_cartpole_swingup"
#    "dmc_pendulum_swingup"
#    "dmc_reacher_easy"
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

TASK_SHORT=${TASK#dmc_} 

OT_TYPE="R" # OT pretrained model is used. ER: expert and random data; R: random data only; P: p2e data only

python scripts/train_dreamer.py \
  --configs dmc_vision motion_prior \
  --task=$TASK \
  --reference_dir="./tpil_data/dmc_tpil/${TASK}" \
  --tpil=True \
  --reward_model_use_ot=False \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1

