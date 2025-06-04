import os
import torch
import json
import argparse
import gc

from src.unet import UNet
from src.tools import load_dataset, get_pushed_loader_stats, get_loader_stats
from src.fid_score import calculate_frechet_distance

def load_stats(stats_path, device):
    with open(stats_path, 'r') as f:
        data = json.load(f)
    mu = torch.tensor(data['mu']).to(device)
    sigma = torch.tensor(data['sigma']).to(device)
    return mu, sigma

def evaluate_fid(model_path, task_name, dataset2_path, dataset3_path, img_size=64, batch_size=64, device='cuda:0'):
    assert torch.cuda.is_available()
    torch.cuda.set_device(device)

    # Load the evaluation (new) source dataset
    _, eval_sampler = load_dataset("TN", dataset3_path, img_size=img_size)
    # Load the target dataset (used for training)
    _, target_sampler = load_dataset("TN", dataset2_path, img_size=img_size)

    # Load trained transformation model T
    T = UNet(3, 3, base_factor=48).to(device)
    T.load_state_dict(torch.load(model_path, map_location=device))
    T.eval()

    # Compute statistics directly from the DataLoader
    mu_target, sigma_target = get_loader_stats(target_sampler.loader)

    # Compute stats for T(dataset3)
    print("Computing pushed stats for T(dataset3)...")
    mu_eval, sigma_eval = get_pushed_loader_stats(T, eval_sampler.loader)

    # Compute FID
    fid = calculate_frechet_distance(mu_eval, sigma_eval, mu_target, sigma_target)
    print(f"FID Score (T(dataset3) vs dataset2): {fid:.4f}")

    gc.collect()
    torch.cuda.empty_cache()
    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='dmc_cheetah_run')
    parser.add_argument("--model_path", type=str, default='./checkpoints_R/mse/cheetah_run/SN_TN_64/T_8999.pth')
    #parser.add_argument("--dataset2_path", type=str, default='/home/zyang2/XViper/ot_data/dmc_random_orange/cheetah_run/dmc_cheetah_run_50.h5')
    parser.add_argument("--dataset2_path", type=str, default='./eval/dataset/cheetah_expert_orange.h5')
    #parser.add_argument("--dataset3_path", type=str, default='/home/zyang2/XViper/ot_data/dmc_random_gray/cheetah_run/dmc_cheetah_run_50.h5')
    parser.add_argument("--dataset3_path", type=str, default='./eval/dataset/cheetah_expert_gray.h5')
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    evaluate_fid(
        model_path=args.model_path,
        task_name=args.task_name,
        dataset2_path=args.dataset2_path,
        dataset3_path=args.dataset3_path,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

