import os
import numpy as np
import h5py
from glob import glob
import zipfile
import imageio
import pickle
import tensorflow as tf

# Directory containing task folders
tasks_root = '/home/zyang2/XViper/logdir/' # for extracting from replay buffer
#tasks_root = '/home/zyang/XViper/viper_rl_data/datasets/dmc/' # extracting from downloaded
# Output directory to store per-task HDF5 files
output_root = './atari_expert_gray/'

# Create output_root if it doesn't exist
os.makedirs(output_root, exist_ok=True)

#for task in sorted(os.listdir(tasks_root)): # directly take all folders in the dir
for task in ['atari_kangaroo']:
    if 'atari' in task:
        atari_mask_path = '/home/zyang2/XViper/viper_rl_data/checkpoints/atari_vqgan/mask_map.pkl'
        mask_map = pickle.load(tf.io.gfile.GFile(atari_mask_path, 'rb'))
        task_key = task[6:]  # Strip 'atari_' prefix
        mask = mask_map[task_key]
    else:
        mask = 1

    print('processing {}...'.format(task))
    task_path = os.path.join(tasks_root, task)
    drv3_path = os.path.join(task_path, 'Drv3_gray', '2', 'replay')
    #drv3_path = os.path.join(task_path, 'train')
    #drv3_path = os.path.join(task_path, 'P2E', '0', 'replay')

    # Skip tasks without Drv3/0/replay
    if not os.path.isdir(drv3_path):
        continue

    # Find all npz files in replay directory
    npz_files = glob(os.path.join(drv3_path, '*.npz'))
    if not npz_files:
        continue

    # Sort by timestamp embedded in filename (ISO format prefix)
    # e.g. '20250610T183410F....-1024.npz'
    def extract_timestamp(fname):
        base = os.path.basename(fname)
        # timestamp is before first 'F'
        ts = base.split('F', 1)[0]
        # convert to a sortable string (YYYYMMDDThhmmss)
        return ts

    npz_files.sort(key=extract_timestamp)
    latest = npz_files[-500:] # take the last 50
    
    ##### take 50 evenly ######
    #total_files = len(npz_files)
    #num_samples = 50
    # Get 50 evenly spaced indices
    #indices = np.linspace(0, total_files - 1, num_samples, dtype=int)
    # Select the corresponding files
    #latest = [npz_files[i] for i in indices]

    # Load frames from latest files
    frames_list = []
    count = 0

    for file in latest:
        # Skip empty or non-zip files
        if os.path.getsize(file) == 0 or not zipfile.is_zipfile(file):
            print(f"Warning: skipping invalid file '{file}'")
            continue
        try:
            data = np.load(file, allow_pickle=True)
        except Exception as e:
            print(f"Warning: could not load '{file}': {e}")
            continue

        # Extract array
        if isinstance(data, np.lib.npyio.NpzFile):
            # Try default key then fallback
            arr = data.get('image')
            arr = arr * mask
            if arr is None:
                try:
                    arr = next(iter(data.values()))
                except StopIteration:
                    print(f"Warning: no arrays in '{file}'")
                    continue
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            print(f"Warning: unknown data type in '{file}'")
            continue

        frames_list.append(arr)
        count += 1


    # Concatenate along time axis
    all_frames = np.concatenate(frames_list, axis=0)

    # Ensure per-task output folder
    task_output_dir = os.path.join(output_root, task)
    os.makedirs(task_output_dir, exist_ok=True)

    # Save to HDF5
    out_path = os.path.join(task_output_dir, f'{task}_latest{count}.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('frames', data=all_frames, compression='gzip')

    # Save as MP4 video at 400 fps
    mp4_path = os.path.join(task_output_dir, f'{task}_latest500.mp4')
    imageio.mimsave(mp4_path, all_frames, fps=400)

    print(f"Task '{task}': saved {all_frames.shape[0]} frames to {out_path}")

