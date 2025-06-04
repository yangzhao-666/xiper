"""Label a replay buffer with XIPER rewards.

Supports two dataset types:
  --dataset_type random   SB3-format pkl + npz images (default)
  --dataset_type expert   List-of-dicts pkl + zip of PNGs

Usage:
    # Random (SB3 format):
    python scripts/label_replay_buffer.py --dataset_type random --use_ot

    # Expert (list-of-dicts format, first 5k samples):
    python scripts/label_replay_buffer.py --dataset_type expert --use_ot \
        --pkl_path viper_rl_data/lynx/replay_buffer_direct_20260301_072152_noisy_expert.pkl \
        --images_zip viper_rl_data/lynx/images_real_noisy_expert.zip \
        --max_samples 5000
"""

import os
import sys
import re
import argparse
import pickle
import zipfile
import numpy as np
from PIL import Image
import io
import tqdm

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel


class _DummyRNG:
    """Placeholder that accepts any state without error."""
    def __setstate__(self, state):
        pass
    def __getstate__(self):
        return {}


def _stub_bit_generator_ctor(*args, **kwargs):
    return _DummyRNG()


def _stub_generator_ctor(*args, **kwargs):
    return _DummyRNG()


class _SB3ReplayBuffer:
    """Generic container for SB3-style replay buffer attributes."""
    pass


class _CompatUnpickler(pickle.Unpickler):
    """Handle numpy 2.x pickles in numpy 1.x environments."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        # Stub out all numpy random pickle helpers that break across versions
        if module == 'numpy.random._pickle':
            if name == '__bit_generator_ctor':
                return _stub_bit_generator_ctor
            if name == '__generator_ctor':
                return _stub_generator_ctor
        # Handle _DummyRNG saved from a previous labeling run
        if name == '_DummyRNG':
            return _DummyRNG
        # Handle SB3ReplayBuffer saved from conversion script
        if name in ('SB3ReplayBuffer', '_SB3ReplayBuffer'):
            return _SB3ReplayBuffer
        return super().find_class(module, name)


def load_replay_buffer(pkl_path):
    """Load a SB3 replay buffer pickle file."""
    with open(pkl_path, 'rb') as f:
        rb = _CompatUnpickler(f).load()
    return rb


def load_images(npz_path):
    """Load visual observations from .npz file.

    Returns obs_images and next_obs_images, both (N, H, W, C) uint8.
    """
    data = np.load(npz_path)
    obs_images = data['obs_images']
    next_obs_images = data['next_obs_images']
    return obs_images, next_obs_images


def load_images_from_zip(zip_path, image_paths):
    """Load images from a zip file, matched by filename to image_paths.

    Args:
        zip_path: Path to .zip containing PNG images.
        image_paths: List of original absolute paths from the pkl entries.

    Returns:
        obs_images: (N, H, W, C) uint8 array.
    """
    # Build a lookup: basename -> zip member name
    with zipfile.ZipFile(zip_path) as z:
        members = {os.path.basename(n): n for n in z.namelist() if n.endswith('.png')}
        images = []
        for p in tqdm.tqdm(image_paths, desc='Loading images from zip'):
            basename = os.path.basename(p)
            img_data = z.read(members[basename])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            images.append(np.array(img))
    return np.stack(images)


def build_episodes(rb, obs_images):
    """Split the flat replay buffer into per-episode sequences.

    Each episode is a list of dicts with keys: 'image', 'is_first'.
    Returns list of (episode_seq, global_indices) tuples.
    """
    n_samples = int(rb.pos if not rb.full else rb.buffer_size)
    dones = rb.dones[:n_samples].flatten()

    episodes = []
    ep_seq = []
    ep_idxs = []

    for i in range(n_samples):
        step = {
            'image': obs_images[i],
            'is_first': len(ep_seq) == 0,
        }
        ep_seq.append(step)
        ep_idxs.append(i)

        if dones[i]:
            episodes.append((ep_seq, ep_idxs))
            ep_seq = []
            ep_idxs = []

    # Remaining partial episode
    if ep_seq:
        episodes.append((ep_seq, ep_idxs))

    return episodes


def build_episodes_from_list(data_list, obs_images):
    """Split a list-of-dicts replay buffer into per-episode sequences.

    Args:
        data_list: List of dicts with 'done' key.
        obs_images: (N, H, W, C) uint8 array.

    Returns list of (episode_seq, global_indices) tuples.
    """
    episodes = []
    ep_seq = []
    ep_idxs = []

    for i in range(len(data_list)):
        step = {
            'image': obs_images[i],
            'is_first': len(ep_seq) == 0,
        }
        ep_seq.append(step)
        ep_idxs.append(i)

        if data_list[i]['done']:
            episodes.append((ep_seq, ep_idxs))
            ep_seq = []
            ep_idxs = []

    if ep_seq:
        episodes.append((ep_seq, ep_idxs))

    return episodes


def label_rewards(reward_model, episodes):
    """Run the reward model on each episode and collect per-step rewards.

    Returns a dict mapping global buffer index -> xiper reward value.
    """
    reward_map = {}

    for ep_seq, ep_idxs in tqdm.tqdm(episodes, desc='Labeling episodes'):
        if len(ep_seq) < reward_model.seq_len_steps:
            print(f'  Skipping short episode (len={len(ep_seq)}, '
                  f'need>={reward_model.seq_len_steps})')
            continue

        # reward_model.__call__ runs compute_reward + process_seq
        # process_seq truncates the first (seq_len_steps - 1) frames
        viper_seq = reward_model(ep_seq)
        truncated = reward_model.seq_len_steps - 1

        for j, step in enumerate(viper_seq):
            global_idx = ep_idxs[truncated + j]
            reward_map[global_idx] = float(step['density'])

    return reward_map


def _label_and_store_sb3(rb, obs_images, n_samples, reward_model, use_ot):
    """Label an SB3-format replay buffer and attach reward arrays."""
    episodes = build_episodes(rb, obs_images)
    print(f'Found {len(episodes)} episodes')

    if use_ot:
        print('Computing XIPER rewards (with OT)...')
        xiper_map = label_rewards(reward_model, episodes)
        print(f'  XIPER labeled {len(xiper_map)}/{n_samples} steps')

        xiper_rewards = np.full((rb.buffer_size, 1), np.nan, dtype=np.float32)
        for idx, rew in xiper_map.items():
            xiper_rewards[idx, 0] = rew
        rb.xiper_rewards = xiper_rewards

        print('Computing VIPER rewards (without OT)...')
        episodes_fresh = build_episodes(rb, obs_images)
        reward_model.use_ot = False
        viper_map = label_rewards(reward_model, episodes_fresh)
        reward_model.use_ot = True
        print(f'  VIPER labeled {len(viper_map)}/{n_samples} steps')

        viper_rewards = np.full((rb.buffer_size, 1), np.nan, dtype=np.float32)
        for idx, rew in viper_map.items():
            viper_rewards[idx, 0] = rew
        rb.viper_rewards = viper_rewards
    else:
        print('Computing VIPER rewards (no OT)...')
        viper_map = label_rewards(reward_model, episodes)
        print(f'  VIPER labeled {len(viper_map)}/{n_samples} steps')

        viper_rewards = np.full((rb.buffer_size, 1), np.nan, dtype=np.float32)
        for idx, rew in viper_map.items():
            viper_rewards[idx, 0] = rew
        rb.viper_rewards = viper_rewards

    # Print summary
    if hasattr(rb, 'xiper_rewards'):
        labeled = np.sum(~np.isnan(rb.xiper_rewards[:n_samples]))
        print(f'  xiper_rewards: {labeled}/{n_samples} labeled')
    if hasattr(rb, 'viper_rewards'):
        labeled = np.sum(~np.isnan(rb.viper_rewards[:n_samples]))
        print(f'  viper_rewards: {labeled}/{n_samples} labeled')

    # Replace dummy RNG placeholders
    def _fix_rng(obj, visited=None):
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        if not hasattr(obj, '__dict__'):
            return
        for k, v in obj.__dict__.items():
            if isinstance(v, _DummyRNG):
                setattr(obj, k, np.random.default_rng())
            else:
                _fix_rng(v, visited)
    _fix_rng(rb)
    return rb


def _label_and_store_expert(data_list, obs_images, n_samples, reward_model, use_ot):
    """Label a list-of-dicts replay buffer and attach reward fields."""
    episodes = build_episodes_from_list(data_list, obs_images)
    print(f'Found {len(episodes)} episodes')

    if use_ot:
        print('Computing XIPER rewards (with OT)...')
        xiper_map = label_rewards(reward_model, episodes)
        print(f'  XIPER labeled {len(xiper_map)}/{n_samples} steps')

        print('Computing VIPER rewards (without OT)...')
        episodes_fresh = build_episodes_from_list(data_list, obs_images)
        reward_model.use_ot = False
        viper_map = label_rewards(reward_model, episodes_fresh)
        reward_model.use_ot = True
        print(f'  VIPER labeled {len(viper_map)}/{n_samples} steps')

        for i in range(n_samples):
            data_list[i]['xiper_reward'] = xiper_map.get(i, float('nan'))
            data_list[i]['viper_reward'] = viper_map.get(i, float('nan'))
    else:
        print('Computing VIPER rewards (no OT)...')
        viper_map = label_rewards(reward_model, episodes)
        print(f'  VIPER labeled {len(viper_map)}/{n_samples} steps')

        for i in range(n_samples):
            data_list[i]['viper_reward'] = viper_map.get(i, float('nan'))

    # Print summary
    labeled_v = sum(1 for d in data_list if not np.isnan(d.get('viper_reward', float('nan'))))
    print(f'  viper_reward: {labeled_v}/{n_samples} labeled')
    if use_ot:
        labeled_x = sum(1 for d in data_list if not np.isnan(d.get('xiper_reward', float('nan'))))
        print(f'  xiper_reward: {labeled_x}/{n_samples} labeled')

    return data_list


def main():
    parser = argparse.ArgumentParser(
        description='Label replay buffer with XIPER/VIPER rewards')
    parser.add_argument('--dataset_type', type=str, default='random',
                        choices=['random', 'expert'],
                        help='Dataset type: random (SB3 pkl+npz) or expert (list-of-dicts pkl+zip)')
    parser.add_argument('--pkl_path', type=str, default=None,
                        help='Path to replay buffer .pkl file')
    parser.add_argument('--npz_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'real_random_images.npz'),
                        help='Path to visual observations .npz file (random mode)')
    parser.add_argument('--images_zip', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'images_real_noisy_expert.zip'),
                        help='Path to images .zip file (expert mode)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to use (expert mode)')
    parser.add_argument('--task', type=str, default='lynx_reach',
                        help='Task name in domain_task format (e.g. lynx_reach)')
    parser.add_argument('--videogpt_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'self_trained_checkpoints', 'reach_controller_videogpt_seqlen2'),
                        help='Path to VideoGPT checkpoint directory')
    parser.add_argument('--vqgan_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'self_trained_checkpoints', 'reach_controller_vqgan_gpus'),
                        help='Path to VQGAN checkpoint directory')
    parser.add_argument('--use_ot', action='store_true',
                        help='Enable OT domain translation')
    parser.add_argument('--ot_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'OT', 'checkpoints_sim2real', 'mse', 'close_view', 'SN_TN_64', 'T_2999.pth'),
                        help='Path to OT model checkpoint')
    parser.add_argument('--minibatch_size', type=int, default=2,
                        help='Minibatch size for VideoGPT likelihood')
    parser.add_argument('--encoding_minibatch_size', type=int, default=32,
                        help='Minibatch size for VQGAN encoding')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output .pkl path (default: overwrite input)')
    args = parser.parse_args()

    if args.use_ot and args.ot_path is None:
        parser.error('--ot_path is required when --use_ot is set')

    # Set default pkl_path based on dataset_type
    if args.pkl_path is None:
        if args.dataset_type == 'random':
            args.pkl_path = os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'real_random_replay_buffer_sb3.pkl')
        else:
            args.pkl_path = os.path.join(os.path.dirname(__file__), '..', 'viper_rl_data', 'lynx', 'replay_buffer_direct_20260301_072152_noisy_expert.pkl')

    # Load reward model
    print(f'Loading reward model (use_ot={args.use_ot})...')
    reward_model = VideoGPTRewardModel(
        task=args.task,
        videogpt_path=args.videogpt_path,
        vqgan_path=args.vqgan_path,
        minibatch_size=args.minibatch_size,
        encoding_minibatch_size=args.encoding_minibatch_size,
        compute_joint=True,
        use_ot=args.use_ot,
        ot_path=args.ot_path,
    )

    if args.dataset_type == 'random':
        # SB3 format: pkl (buffer object) + npz (images)
        print(f'Loading replay buffer from {args.pkl_path}')
        rb = load_replay_buffer(args.pkl_path)
        n_samples = int(rb.pos if not rb.full else rb.buffer_size)
        print(f'  buffer_size={rb.buffer_size}, pos={rb.pos}, '
              f'full={rb.full}, valid_samples={n_samples}')

        print(f'Loading images from {args.npz_path}')
        obs_images, _ = load_images(args.npz_path)
        print(f'  obs_images: {obs_images.shape}')

        rb = _label_and_store_sb3(rb, obs_images, n_samples, reward_model, args.use_ot)

        output_path = args.output_path or args.pkl_path
        print(f'Saving to {output_path}')
        with open(output_path, 'wb') as f:
            pickle.dump(rb, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Expert format: pkl (list of dicts) + zip (PNG images)
        print(f'Loading expert replay buffer from {args.pkl_path}')
        with open(args.pkl_path, 'rb') as f:
            data_list = _CompatUnpickler(f).load()
        print(f'  Total entries: {len(data_list)}')

        if args.max_samples is not None and args.max_samples < len(data_list):
            data_list = data_list[:args.max_samples]
            print(f'  Truncated to first {args.max_samples} samples')

        n_samples = len(data_list)
        image_paths = [d['image_path'] for d in data_list]

        print(f'Loading images from {args.images_zip}')
        obs_images = load_images_from_zip(args.images_zip, image_paths)
        print(f'  obs_images: {obs_images.shape}')

        data_list = _label_and_store_expert(data_list, obs_images, n_samples, reward_model, args.use_ot)

        output_path = args.output_path or args.pkl_path
        print(f'Saving to {output_path}')
        with open(output_path, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')


if __name__ == '__main__':
    main()
