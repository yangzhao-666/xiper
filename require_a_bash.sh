#!/bin/bash
srun --partition gpu_a100 --gpus 1 -t 8:00:00 --cpus-per-gpu=16 --gpus-per-task=1 --pty bash
#srun --partition gpu_h100 --gpus 1 -t 48:00:00 --cpus-per-gpu=4 --gpus-per-task=1 --pty bash
#srun --partition cbuild -t 48:00:00 --pty bash
