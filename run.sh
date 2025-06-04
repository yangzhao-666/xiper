#!/bin/bash                                                                                  
#modified by Zhao Yang, originally from Edward Hu
 
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1 
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --array=4
#SBATCH --output=/home/zyang/logdir/slurm/output/%j.out
 
conda deactivate
conda deactivate
conda deactivate
conda activate viper

module load cuDNN/8.9.2.26-CUDA-12.1.1
module load FFmpeg/6.0-GCCcore-12.3.0

SEED=$SLURM_ARRAY_TASK_ID

export MUJOCO_GL="egl"

AGENT="Drv3"
#AGENT="Viper"
#AGENT="OViper"
#AGENT="XViper"

#TASK="dmc_cartpole_balance"
#TASK="dmc_quadruped_walk"
#TASK="dmc_cup_catch"
#TASK="dmc_cheetah_run"
#TASK="dmc_finger_spin"
#TASK="dmc_pointmass_easy"
TASK="dmc_hopper_stand"
#TASK="dmc_walker_walk"
#TASK="dmc_cartpole_swingup"
#TASK="dmc_pendulum_swingup"

python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=$TASK --reward_model=dmc_clen16_fskip4 --reward_model_use_ot=False --task_behavior=Greedy --logdir=./logdir/${TASK}/${AGENT}/${SEED}>"test_${SEED}.out" 2>&1
