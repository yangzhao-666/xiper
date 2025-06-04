import os
import numpy as np
import h5py
from tqdm import tqdm

def load_good_sequences(sequence_dir, min_ep_len=200, min_rew=800, max_rew=2000, max_ep=100):
    """
    Load good episodes from .npz files in a directory, based on total reward.
    """
    sequences = []
    loaded = 0
    for fname in sorted(os.listdir(sequence_dir)):
        if not fname.endswith('.npz'):
            continue

        path = os.path.join(sequence_dir, fname)
        try:
            data = np.load(path)
            images = data['image']          # shape: (N, 64, 64, 3)
            rewards = data['reward']         # shape: (N,)
            total_rew = np.sum(rewards)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        if len(images) < min_ep_len:
            continue
        if not (min_rew <= total_rew <= max_rew):
            continue

        sequences.append(images)
        loaded += 1
        if loaded >= max_ep:
            break

    print(f"Loaded {loaded} good episodes from {sequence_dir}")
    return sequences

def save_sequences_to_h5(sequences, output_path, key='frames'):
    """
    Save image frames from sequences to an HDF5 file under the given key.
    """
    all_frames = []
    for seq in sequences:
        all_frames.append(seq)  # shape (T, H, W, C)

    if not all_frames:
        print("No valid sequences to save.")
        return

    all_frames = np.concatenate(all_frames, axis=0)  # (N, H, W, C)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset(key, data=all_frames, compression='gzip')
    
    print(f"Saved {all_frames.shape[0]} frames to {output_path} under key '{key}'")

def main():
    # Configuration
    sequence_dir = '/home/zyang2/XViper/logdir/dmc_cheetah_run/Drv3/0/replay'  # TODO: update this path
    output_h5_path = './eval/dataset/cheetah_expert_orange.h5'

    # Load and filter good sequences
    good_sequences = load_good_sequences(
        sequence_dir=sequence_dir,
        min_ep_len=200,
        min_rew=800,
        max_rew=2000,
        max_ep=50
    )

    # Save to HDF5
    save_sequences_to_h5(good_sequences, output_path=output_h5_path)

if __name__ == '__main__':
    main()

