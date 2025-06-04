"""Precompute all XIPER reward pipeline ingredients for visualization.

For each selected episode, saves per-window (8-frame) ingredients:
  1. original frames (uint8)
  2. OT-translated frames (uint8)
  3. VideoGPT predicted frames given open_loop_ctx context frames (uint8)
  4. per-step original rewards, VIPER rewards, XIPER rewards

Supports two dataset types:
  --dataset_type random   SB3-format pkl + npz images (default)
  --dataset_type expert   List-of-dicts pkl + zip of PNGs

Usage (on a GPU node):
    # Random (default):
    python scripts/precompute_xiper_ingredients.py

    # Expert:
    python scripts/precompute_xiper_ingredients.py --dataset_type expert

Output: viper_rl_data/lynx/xiper_ingredients.npz
"""

import os, sys, pickle, argparse
import numpy as np

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from scripts.label_replay_buffer import (
    _CompatUnpickler, load_images_from_zip,
)
from viper_rl.videogpt.models import AE, load_videogpt
from viper_rl.videogpt.reward_models.ot import OT
from viper_rl.videogpt import sampler as vgpt_sampler


def _load_random(args):
    """Load SB3-format replay buffer + npz images."""
    print('Loading replay buffer (SB3 format)...')
    with open(args.pkl_path, 'rb') as f:
        rb = _CompatUnpickler(f).load()
    n_samples = int(rb.pos if not rb.full else rb.buffer_size)
    rewards_all = rb.rewards[:n_samples].flatten()
    dones = rb.dones[:n_samples].flatten().astype(bool)
    viper_rewards_all = rb.viper_rewards[:n_samples].flatten() if hasattr(rb, 'viper_rewards') else None
    xiper_rewards_all = rb.xiper_rewards[:n_samples].flatten() if hasattr(rb, 'xiper_rewards') else None

    print('Loading images...')
    images = np.load(args.npz_path)['obs_images']
    return rewards_all, dones, viper_rewards_all, xiper_rewards_all, images


def _load_expert(args):
    """Load list-of-dicts replay buffer + zip images."""
    print('Loading replay buffer (expert list-of-dicts format)...')
    with open(args.pkl_path, 'rb') as f:
        data_list = _CompatUnpickler(f).load()
    if args.max_samples is not None and args.max_samples < len(data_list):
        data_list = data_list[:args.max_samples]
    n_samples = len(data_list)
    print(f'  {n_samples} samples')

    rewards_all = np.array([d['reward'] for d in data_list], dtype=np.float32)
    dones = np.array([d['done'] for d in data_list], dtype=bool)
    viper_rewards_all = np.array(
        [d.get('viper_reward', float('nan')) for d in data_list], dtype=np.float32)
    xiper_rewards_all = np.array(
        [d.get('xiper_reward', float('nan')) for d in data_list], dtype=np.float32)

    if np.all(np.isnan(viper_rewards_all)):
        viper_rewards_all = None
    if np.all(np.isnan(xiper_rewards_all)):
        xiper_rewards_all = None

    image_paths = [d['image_path'] for d in data_list]
    print(f'Loading images from {args.images_zip}...')
    images = load_images_from_zip(args.images_zip, image_paths)
    return rewards_all, dones, viper_rewards_all, xiper_rewards_all, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='random',
                        choices=['random', 'expert'])
    parser.add_argument('--pkl_path', type=str, default=None)
    parser.add_argument('--npz_path', type=str,
                        default='viper_rl_data/lynx/real_random_images.npz')
    parser.add_argument('--images_zip', type=str,
                        default='viper_rl_data/lynx/images_real_noisy_expert.zip')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--vqgan_path', type=str,
                        default='viper_rl_data/self_trained_checkpoints/reach_controller_vqgan_gpus')
    parser.add_argument('--videogpt_path', type=str,
                        default='viper_rl_data/self_trained_checkpoints/reach_controller_videogpt_seqlen2')
    parser.add_argument('--ot_path', type=str,
                        default='OT/checkpoints_sim2real/mse/close_view/SN_TN_64/T_2999.pth')
    parser.add_argument('--episodes', type=str, default=None,
                        help='Comma-separated episode indices (e.g. "2,24,29,94,120"). '
                             'If not set, randomly picks --n_episodes.')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--n_windows_per_ep', type=int, default=5,
                        help='Number of 8-frame windows to sample per episode')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str,
                        default='viper_rl_data/lynx/xiper_ingredients.npz')
    args = parser.parse_args()

    if args.pkl_path is None:
        if args.dataset_type == 'random':
            args.pkl_path = 'viper_rl_data/lynx/real_random_replay_buffer_sb3.pkl'
        else:
            args.pkl_path = 'viper_rl_data/lynx/replay_buffer_direct_20260301_072152_noisy_expert.pkl'

    # --- Load data ---
    if args.dataset_type == 'random':
        rewards_all, dones, viper_rewards_all, xiper_rewards_all, images = _load_random(args)
    else:
        rewards_all, dones, viper_rewards_all, xiper_rewards_all, images = _load_expert(args)

    # --- Split into episodes ---
    ep_boundaries = np.where(dones)[0]
    ep_starts = np.concatenate([[0], ep_boundaries[:-1] + 1])
    ep_ends = ep_boundaries + 1
    n_episodes_total = len(ep_starts)
    print(f'{n_episodes_total} episodes found')

    # --- Pick episodes ---
    if args.episodes is not None:
        chosen_eps = np.array([int(x) for x in args.episodes.split(',')])
    else:
        rng = np.random.RandomState(args.seed)
        chosen_eps = rng.choice(n_episodes_total, size=min(args.n_episodes, n_episodes_total), replace=False)
    chosen_eps.sort()
    print(f'Selected episodes: {chosen_eps}')

    # --- Load models ---
    device = jax.devices()[0]
    print(f'JAX device: {device}')

    print('Loading VQGAN...')
    ae = AE(path=args.vqgan_path, mode='jit')
    ae.ae_vars = jax.device_put(ae.ae_vars, device)

    # Check for mask
    mask = None
    if ae.mask_map is not None:
        for key in ae.mask_map:
            if 'reach' in key:
                mask = ae.mask_map[key].astype(np.uint8)
                print(f'Loaded mask for task "{key}", shape={mask.shape}')
                break
    if mask is None:
        print('No mask found.')

    print('Loading VideoGPT...')
    model, variables, class_map = load_videogpt(args.videogpt_path, ae=ae, replicate=False)
    variables = jax.device_put(variables, device)
    seq_len = model.config.seq_len  # 8
    open_loop_ctx = getattr(model.config, 'open_loop_ctx', 1)
    print(f'  seq_len={seq_len}, open_loop_ctx={open_loop_ctx}')

    gpt_sampler = vgpt_sampler.VideoGPTSampler(model, mode='jit')

    print('Loading OT model...')
    ot_model = OT(args.ot_path)

    # --- Helper: process images the same way the reward model does ---
    def to_model_input(img_batch_uint8):
        """uint8 NHWC -> float32 [-1, 1] for VQGAN."""
        x = jnp.array(img_batch_uint8, dtype=jnp.float32)
        if mask is not None:
            x = x * jnp.array(mask, dtype=jnp.float32)
        return x / 127.5 - 1.0

    def decode_to_uint8(decoded):
        """float32 [-1,1] -> uint8 [0,255]."""
        return np.array(jnp.clip(decoded * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)

    # --- Full-episode OT translations (for GIF visualization) ---
    full_ep_list = []
    max_ep_len = int((ep_ends - ep_starts).max())
    for ep_i in chosen_eps:
        s, e = int(ep_starts[ep_i]), int(ep_ends[ep_i])
        ep_len = e - s
        ep_orig = images[s:e]  # (ep_len, H, W, C)
        print(f'  OT-translating full episode {ep_i} ({ep_len} frames)...')
        ep_ot = np.array(ot_model.translate(ep_orig))  # (ep_len, H, W, C)

        # VideoGPT predictions for the full episode (rolling windows)
        print(f'  VideoGPT predicting full episode {ep_i}...')
        ep_pred = np.zeros_like(ep_ot)
        # First open_loop_ctx frames: copy from OT-translated (no prediction possible)
        ot_input = to_model_input(ep_ot)
        ep_pred[:open_loop_ctx] = ep_ot[:open_loop_ctx]
        # Roll through the episode in steps of (seq_len - open_loop_ctx)
        stride = seq_len - open_loop_ctx
        for win_start in range(0, ep_len - open_loop_ctx, stride):
            win_end = min(win_start + seq_len, ep_len)
            win_len = win_end - win_start
            # Encode the OT-translated window
            win_input = jnp.expand_dims(ot_input[win_start:win_end], axis=0)
            if win_len < seq_len:
                # Pad to seq_len for the model
                pad_len = seq_len - win_len
                win_input = jnp.pad(win_input, ((0,0),(0,pad_len),(0,0),(0,0),(0,0)))
            win_enc = ae.encode(win_input)
            batch = dict(
                encodings=win_enc,
                label=jnp.array([0], dtype=jnp.int32),
            )
            pred_enc = gpt_sampler(
                variables, batch=batch, log_tqdm=False,
                seed=0, open_loop_ctx=open_loop_ctx, decode=False,
            )
            pred_dec = ae.decode(pred_enc)
            pred_uint8 = decode_to_uint8(pred_dec[0])
            # Only write the predicted (non-context) frames
            pred_start = win_start + open_loop_ctx
            pred_end = min(pred_start + stride, ep_len)
            ep_pred[pred_start:pred_end] = pred_uint8[open_loop_ctx:open_loop_ctx + (pred_end - pred_start)]

        # Pad to max_ep_len so we can stack into one array
        if ep_len < max_ep_len:
            pad = np.zeros((max_ep_len - ep_len,) + ep_orig.shape[1:], dtype=np.uint8)
            ep_orig = np.concatenate([ep_orig, pad])
            ep_ot = np.concatenate([ep_ot, pad])
            ep_pred = np.concatenate([ep_pred, pad])
        full_ep_list.append(dict(ep_idx=ep_i, ep_len=ep_len,
                                 original=ep_orig, ot_translated=ep_ot,
                                 videogpt_pred=ep_pred))
    print(f'Full-episode translations done for {len(full_ep_list)} episodes')

    # --- Collect per-window ingredients ---
    results = []

    for ep_i in chosen_eps:
        s, e = int(ep_starts[ep_i]), int(ep_ends[ep_i])
        ep_len = e - s
        if ep_len < seq_len:
            print(f'  Episode {ep_i} too short ({ep_len} < {seq_len}), skipping')
            continue

        # Pick window start indices spread across the episode
        max_start = ep_len - seq_len
        if args.n_windows_per_ep >= max_start + 1:
            win_starts = np.arange(0, max_start + 1)
        else:
            win_starts = np.linspace(0, max_start, args.n_windows_per_ep, dtype=int)
        # Remove duplicates
        win_starts = np.unique(win_starts)

        for w_off in win_starts:
            w_start = s + w_off
            w_end = w_start + seq_len
            global_idxs = np.arange(w_start, w_end)

            # 1. Original frames
            orig_frames = images[w_start:w_end]  # (8, H, W, C)

            # 2. OT translation (real -> sim domain)
            ot_translated = np.array(ot_model.translate(orig_frames))  # (8, H, W, C) uint8

            # 3. VQGAN encode the OT-translated frames (needed for VideoGPT)
            ot_model_input = to_model_input(ot_translated)
            ot_model_input_5d = jnp.expand_dims(ot_model_input, axis=0)  # (1, 8, H, W, C)
            encodings = ae.encode(ot_model_input_5d)  # (1, 8, h, w)

            # 4. VideoGPT prediction: given first frame as context,
            #    autoregressively predict the remaining 7 frames.
            #    This shows what the model "expects" to see — the reward
            #    is the log-likelihood of the actual sequence under this
            #    autoregressive model.
            batch = dict(
                encodings=encodings,
                label=jnp.array([0], dtype=jnp.int32),
            )
            predicted_encodings = gpt_sampler(
                variables, batch=batch, log_tqdm=False,
                seed=0, open_loop_ctx=open_loop_ctx, decode=False,
            )
            predicted_decoded = ae.decode(predicted_encodings)  # (1, 8, H, W, C)
            videogpt_pred = decode_to_uint8(predicted_decoded[0])  # (8, H, W, C) uint8

            # 5. Rewards for these frames
            r_orig = rewards_all[w_start:w_end]
            r_viper = viper_rewards_all[w_start:w_end] if viper_rewards_all is not None else np.full(seq_len, np.nan)
            r_xiper = xiper_rewards_all[w_start:w_end] if xiper_rewards_all is not None else np.full(seq_len, np.nan)

            results.append(dict(
                ep_idx=ep_i,
                window_offset=w_off,
                global_idxs=global_idxs,
                original=orig_frames,
                ot_translated=ot_translated,
                videogpt_pred=videogpt_pred,
                reward_orig=r_orig,
                reward_viper=r_viper,
                reward_xiper=r_xiper,
            ))

        print(f'  Episode {ep_i}: {len(win_starts)} windows collected')

    # --- Save ---
    n_win = len(results)
    print(f'\nTotal windows: {n_win}')

    # Stack into arrays
    np.savez_compressed(
        args.output,
        # Per-window data
        ep_idxs=np.array([r['ep_idx'] for r in results]),
        window_offsets=np.array([r['window_offset'] for r in results]),
        global_idxs=np.stack([r['global_idxs'] for r in results]),
        original=np.stack([r['original'] for r in results]),
        ot_translated=np.stack([r['ot_translated'] for r in results]),
        videogpt_pred=np.stack([r['videogpt_pred'] for r in results]),
        reward_orig=np.stack([r['reward_orig'] for r in results]),
        reward_viper=np.stack([r['reward_viper'] for r in results]),
        reward_xiper=np.stack([r['reward_xiper'] for r in results]),
        seq_len=np.array(seq_len),
        open_loop_ctx=np.array(open_loop_ctx),
        # Full-episode data (for GIFs)
        full_ep_idxs=np.array([f['ep_idx'] for f in full_ep_list]),
        full_ep_lens=np.array([f['ep_len'] for f in full_ep_list]),
        full_ep_original=np.stack([f['original'] for f in full_ep_list]),
        full_ep_ot_translated=np.stack([f['ot_translated'] for f in full_ep_list]),
        full_ep_videogpt_pred=np.stack([f['videogpt_pred'] for f in full_ep_list]),
    )
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
