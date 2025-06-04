"""Visualize OT translation: original (left) vs translated (right) side-by-side video.

Usage:
    python scripts/visualize_ot.py
    python scripts/visualize_ot.py --npz_path <path> --ot_path <path> --n_episodes 10
"""

import os
import sys
import argparse
import random
import numpy as np
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viper_rl.videogpt.reward_models.ot import OT


def main():
    parser = argparse.ArgumentParser(description='Visualize OT translation')
    parser.add_argument('--npz_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'real_random_images.npz'),
                        help='Path to images .npz file')
    parser.add_argument('--ot_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'OT', 'checkpoints_sim2real', 'mse', 'close_view', 'SN_TN_64', 'T_2999.pth'),
                        help='Path to OT checkpoint')
    parser.add_argument('--ep_len', type=int, default=100,
                        help='Steps per episode')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of random episodes to include')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'ot_comparison_10eps.mp4'),
                        help='Output video path')
    args = parser.parse_args()

    # Load images
    print(f'Loading images from {args.npz_path}')
    obs_images = np.load(args.npz_path)['obs_images']
    n_eps = len(obs_images) // args.ep_len
    print(f'  {len(obs_images)} images, {n_eps} episodes, {args.ep_len} steps each')

    # Load OT model
    print(f'Loading OT model from {args.ot_path}')
    ot_model = OT(args.ot_path)

    # Pick random episodes
    random.seed(args.seed)
    ep_idxs = sorted(random.sample(range(n_eps), min(args.n_episodes, n_eps)))
    print(f'Selected episodes: {ep_idxs}')

    # Build side-by-side frames
    frames = []
    for ep in ep_idxs:
        start = ep * args.ep_len
        end = start + args.ep_len
        original = obs_images[start:end]
        translated = np.array(ot_model.translate(original))

        for i in range(args.ep_len):
            combined = np.concatenate([original[i], translated[i]], axis=1)
            frames.append(combined)

    frames = np.array(frames)
    print(f'Video: {frames.shape} ({len(ep_idxs)} episodes, {args.ep_len} steps each)')

    imageio.mimsave(args.output, frames, fps=args.fps)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
