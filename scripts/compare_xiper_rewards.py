"""Compare VIPER rewards on expert (VideoGPT training data) vs random (sim data).

Both data sources are in the sim domain (same as VideoGPT training), so no OT needed.

Usage:
    python scripts/compare_xiper_rewards.py \
        --videogpt_path viper_rl_data/self_trained_checkpoints/reach_controller_videogpt \
        --vqgan_path viper_rl_data/self_trained_checkpoints/reach_controller_vqgan_gpus \
        --expert_dir viper_rl_data/datasets/reach_controller/train \
        --random_h5 ot_data/sim2real/close_view/sim_random.h5 \
        --n_episodes 5 \
        --output compare_xiper_rewards.json
"""

import os
import sys
import argparse
import glob
import json
import cv2
import numpy as np
import h5py

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel


def load_episodes_from_mp4s(mp4_dir, n_episodes, seq_len):
    """Load n_episodes from mp4 files, each with at least seq_len frames."""
    mp4_files = sorted(glob.glob(os.path.join(mp4_dir, '*.mp4')))
    episodes = []
    for path in mp4_files:
        if len(episodes) >= n_episodes:
            break
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) < seq_len:
            print(f'  Skipping {os.path.basename(path)}: {len(frames)} frames < {seq_len}')
            continue
        episodes.append((os.path.basename(path), frames))
    return episodes


def load_episodes_from_h5(h5_path, n_episodes, seq_len):
    """Load n_episodes by splitting the h5 image array into chunks of seq_len."""
    with h5py.File(h5_path, 'r') as f:
        all_images = f['images'][:]  # (N, H, W, C)
    episodes = []
    total = len(all_images)
    for i in range(0, total, seq_len):
        if len(episodes) >= n_episodes:
            break
        chunk = all_images[i:i + seq_len]
        if len(chunk) < seq_len:
            break
        episodes.append((f'h5_chunk_{i}_{i+seq_len}', list(chunk)))
    return episodes


def score_episodes(reward_model, episodes, use_ot):
    """Run the reward model on each episode. Returns per-episode (mean, per_step) rewards."""
    original_use_ot = reward_model.use_ot
    reward_model.use_ot = use_ot
    results = []
    for name, frames in episodes:
        seq = [{'image': frame, 'is_first': (i == 0)} for i, frame in enumerate(frames)]
        try:
            viper_seq = reward_model(seq)
        except Exception as e:
            print(f'  Error on {name}: {e}')
            continue
        rewards = [float(step['density']) for step in viper_seq]
        results.append({
            'name': name,
            'mean_reward': float(np.mean(rewards)),
            'per_step': rewards,
        })
        print(f'  {name}: mean={np.mean(rewards):.4f}, '
              f'min={np.min(rewards):.4f}, max={np.max(rewards):.4f}, '
              f'steps={len(rewards)}')
    reward_model.use_ot = original_use_ot
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videogpt_path', type=str,
                        default='viper_rl_data/self_trained_checkpoints/reach_controller_videogpt')
    parser.add_argument('--vqgan_path', type=str,
                        default='viper_rl_data/self_trained_checkpoints/reach_controller_vqgan_gpus')
    parser.add_argument('--task', type=str, default='dmc_reach_controller')
    parser.add_argument('--expert_dir', type=str,
                        default='viper_rl_data/datasets/reach_controller/train')
    parser.add_argument('--random_h5', type=str,
                        default='ot_data/sim2real/close_view/sim_random.h5')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--output', type=str, default='compare_xiper_rewards.json')
    args = parser.parse_args()

    # Load reward model (no OT — both data sources are in sim domain)
    print('Loading reward model...')
    reward_model = VideoGPTRewardModel(
        task=args.task,
        videogpt_path=args.videogpt_path,
        vqgan_path=args.vqgan_path,
        minibatch_size=2,
        encoding_minibatch_size=32,
        compute_joint=True,
        use_ot=False,
    )
    seq_len = reward_model.seq_len_steps

    # Load episodes
    print(f'\nLoading expert episodes from {args.expert_dir} (need seq_len>={seq_len})...')
    expert_episodes = load_episodes_from_mp4s(args.expert_dir, args.n_episodes, seq_len)
    print(f'  Loaded {len(expert_episodes)} expert episodes')

    print(f'\nLoading random episodes from {args.random_h5} (chunks of {seq_len})...')
    random_episodes = load_episodes_from_h5(args.random_h5, args.n_episodes, seq_len)
    print(f'  Loaded {len(random_episodes)} random episodes')

    # Score with VIPER (no OT — same domain)
    print('\n--- VIPER rewards (no OT, same domain) ---')
    print('Expert:')
    expert_viper = score_episodes(reward_model, expert_episodes, use_ot=False)
    print('Random:')
    random_viper = score_episodes(reward_model, random_episodes, use_ot=False)

    # Save results
    results = {
        'seq_len': seq_len,
        'expert_viper': expert_viper,
        'random_viper': random_viper,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
