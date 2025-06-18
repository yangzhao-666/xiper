#!/bin/bash                                                                                  
#modified by Zhao Yang, originally from Edward Hu
 
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1 
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=0
#SBATCH --output=/home/zyang/logdir/slurm/output/%j.out
 
conda deactivate
conda deactivate
conda deactivate
conda activate ot

module load cuDNN/8.9.2.26-CUDA-12.1.1
module load FFmpeg/6.0-GCCcore-12.3.0

SEED=$SLURM_ARRAY_TASK_ID


python train_ot.py>"test_${SEED}.out" 2>&1
