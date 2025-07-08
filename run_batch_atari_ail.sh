#!/bin/bash
# Modified by Zhao Yang, originally from Edward Hu

#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=28:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0-8
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
AGENT="XAIL"

# Define tasks and seeds
TASKS=(
    "atari_pong"
    "atari_kangaroo"
    "atari_boxing"
)

SEEDS=(0 1 2)

# Determine current task/seed from SLURM_ARRAY_TASK_ID
INDEX=$SLURM_ARRAY_TASK_ID
TASK_IDX=$(( INDEX / 3 ))
SEED_IDX=$(( INDEX % 3 ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Run training
echo "Running TASK=$TASK | AGENT=$AGENT | SEED=$SEED"

TASK_SHORT=${TASK#dmc_} 

OT_TYPE="R" # OT pretrained model is used. ER: expert and random data; R: random data only; P: p2e data only

python scripts/train_dreamer.py \
  --configs dmc_vision motion_prior \
  --task=$TASK \
  --reference_dir="./xail_data/${TASK}" \
  --reward_model_use_ot=False \
  --reward_model_ot_path="/home/zyang2/XViper/OT/checkpoints_${OT_TYPE}/mse/${TASK_SHORT}/SN_TN_64/T_999.pth" \
  --reward_model_ot_type=${OT_TYPE} \
  --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1

