import os
import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import PngImagePlugin
import gc
import wandb
import json

# Plotting and utilities
from tqdm import tqdm
from IPython.display import clear_output

# Local modules
from src import distributions
from src.resnet2 import ResNet_D
from src.unet import UNet
from src.tools import (
    unfreeze, freeze, weights_init_D,
    load_dataset, get_pushed_loader_stats,
    get_loader_stats, fig2data, fig2img
)
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_random_images, plot_images

# For loading HDF5 datasets
import h5py
import cv2
import re
import pickle
import tensorflow as tf

# Allow large PNG metadata
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# Default training params
T_ITERS = 10
f_LR = T_LR = 1e-4
IMG_SIZE = 64
BATCH_SIZE = 64
PLOT_INTERVAL = 100
CPKT_INTERVAL = 1000
MAX_STEPS = 10001
SEED = 0x000000
COST = 'mse'
DEVICE_IDS = [0]

def train_ot_model(task_config):
    task = task_config["name"]
    DATASET1, DATASET1_PATH = task_config["DATASET1"], task_config["DATASET1_PATH"]
    DATASET2, DATASET2_PATH = task_config["DATASET2"], task_config["DATASET2_PATH"]

    EXP_NAME = f'{task}_T{T_ITERS}_{COST}_{IMG_SIZE}'
    OUTPUT_PATH = f'./checkpoints_sim2real/{COST}/{task}/{DATASET1}_{DATASET2}_{IMG_SIZE}/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    config = dict(
        DATASET1=DATASET1,
        DATASET2=DATASET2,
        T_ITERS=T_ITERS,
        f_LR=f_LR,
        T_LR=T_LR,
        BATCH_SIZE=BATCH_SIZE,
        IMG_SIZE=IMG_SIZE,
        COST=COST
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialize wandb session for this task
    wandb.init(
        project="strong_ot_R",
        name=EXP_NAME,
        config=config,
        reinit=True
    )

    ####### add atari mask ######
    if 'atari' in task:
        atari_mask_path = '/home/zyang2/XViper/viper_rl_data/checkpoints/atari_vqgan/mask_map.pkl'
        mask_map = pickle.load(tf.io.gfile.GFile(atari_mask_path, 'rb'))
        task_key = task[6:]  # Strip 'atari_' prefix
        mask = mask_map[task_key]
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1) # (3, 64, 64)
        mask = mask.unsqueeze(0) # # (1, 3, 64, 64)
        mask = mask.to('cuda:0')
        mask = mask.to(torch.float32)
    else:
        mask = 1

    # ------------------------- create stats file ----------
    DATASET_LIST = [(DATASET1, DATASET1_PATH), (DATASET2, DATASET2_PATH)]
    for DATASET, DATASET_PATH in DATASET_LIST:
        sampler, test_sampler = load_dataset(DATASET, DATASET_PATH, img_size=IMG_SIZE)
        mu, sigma = get_loader_stats(test_sampler.loader)
        stats = {'mu': mu.tolist(), 'sigma': sigma.tolist()}
        stats_path = f'./stats/{task}/'
        os.makedirs(stats_path, exist_ok=True)
        filename = f'{stats_path}{DATASET}_{IMG_SIZE}_test.json'
        with open(filename, 'w') as fp:
            json.dump(stats, fp)
    del sampler, test_sampler
    filename = f'./stats/{task}/{DATASET2}_{IMG_SIZE}_test.json'
    with open(filename, 'r') as fp:
        data_stats = json.load(fp)
        mu_data, sigma_data = data_stats['mu'], data_stats['sigma']

    # ------------------------- ending create stats file ----------


    X_sampler, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE)
    Y_sampler, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE)

    torch.cuda.empty_cache(); gc.collect()

    f = ResNet_D(IMG_SIZE, nc=3).cuda()
    f.apply(weights_init_D)

    T = UNet(3, 3, base_factor=48).cuda()
    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)

    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    X_test_fixed = X_test_sampler.sample(10)
    Y_test_fixed = Y_test_sampler.sample(10)

    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)

    for step in tqdm(range(MAX_STEPS)):
        # T optimization
        unfreeze(T); freeze(f)
        for t_iter in range(T_ITERS):
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE)
            T_X = T(X)
            if COST == 'mse':
                T_loss = F.mse_loss(X, T_X).mean() - f(T_X).mean()
            else:
                raise Exception('Unknown COST')
            T_loss.backward(); T_opt.step()
        del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()

        # f optimization
        freeze(T); unfreeze(f)
        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            T_X = T(X)
        Y = Y_sampler.sample(BATCH_SIZE)
        f_opt.zero_grad()
        f_loss = f(T_X).mean() - f(Y).mean()
        f_loss.backward(); f_opt.step()
        wandb.log({f'f_loss': f_loss.item()}, step=step)
        del f_loss, Y, X, T_X; gc.collect(); torch.cuda.empty_cache()

        if step % PLOT_INTERVAL == 0:
            clear_output(wait=True)
            fig, _ = plot_images(X_fixed, Y_fixed, T)
            wandb.log({'Fixed Images': [wandb.Image(fig2img(fig))]}, step=step)
            fig, _ = plot_random_images(X_sampler, Y_sampler, T)
            wandb.log({'Random Images': [wandb.Image(fig2img(fig))]}, step=step)
            fig, _ = plot_images(X_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Test Images': [wandb.Image(fig2img(fig))]}, step=step)
            fig, _ = plot_random_images(X_test_sampler, Y_test_sampler, T)
            wandb.log({'Random Test Images': [wandb.Image(fig2img(fig))]}, step=step)

        if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
            print('Computing FID')
            mu, sigma = get_pushed_loader_stats(T, X_test_sampler.loader)
            fid = calculate_frechet_distance(mu, sigma, mu_data, sigma_data)
            wandb.log({'FID': fid}, step=step)
            torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'T_{step}.pth'))
            torch.save(f.state_dict(), os.path.join(OUTPUT_PATH, f'f_{step}.pth'))

    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (e.g., reacher_easy)")
    args = parser.parse_args()

    TASK_CONFIGS = {
        "real_sim": {
            "name": "real_sim",
            # Source: real images (translate FROM)
            "DATASET1": "SN",
            "DATASET1_PATH": "/home/zyang2/XViper/ot_data/sim2real/real.h5",
            # Target: sim images (translate TO)
            "DATASET2": "TN",
            "DATASET2_PATH": "/home/zyang2/XViper/ot_data/sim2real/sim.h5",
        },
        "close_view": {
            "name": "close_view",
            # Source: real images (translate FROM)
            "DATASET1": "SN",
            "DATASET1_PATH": "/home/zyang2/XViper/ot_data/sim2real/close_view/real_random.h5",
            # Target: sim images (translate TO)
            "DATASET2": "TN",
            "DATASET2_PATH": "/home/zyang2/XViper/ot_data/sim2real/close_view/sim_random.h5",
        },
    }

    if args.task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{args.task_name}'. Available: {list(TASK_CONFIGS.keys())}")

    task = TASK_CONFIGS[args.task_name]

    train_ot_model(task)


