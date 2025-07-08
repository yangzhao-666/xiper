#!/bin/bash                                                                                  
#modified by Zhao Yang, originally from Edward Hu
 
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_h100
#SBATCH --output=/home/zyang2/logdir/slurm/output/%j.out

conda deactivate
conda deactivate
conda deactivate
conda activate viper

module load cuDNN/8.9.2.26-CUDA-12.1.1
module load FFmpeg/6.0-GCCcore-12.3.0

SEED=$SLURM_ARRAY_TASK_ID

export MUJOCO_GL="egl"

python scripts/train_videogpt.py -o viper_rl_data/self_trained_checkpoints/atari_videogpt_l16_s1 -c viper_rl/configs/videogpt/atari.yaml>"test_videogpt.out" 2>&1
