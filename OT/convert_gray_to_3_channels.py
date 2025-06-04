import h5py
import numpy as np

# Path to your input and output HDF5 files
input_path = '/home/zyang2/XViper/ot_data/atari_expert_500/atari_expert_gray/atari_boxing/atari_boxing_latest500.h5'
output_path = '/home/zyang2/XViper/ot_data/atari_expert_500/atari_expert_gray/atari_boxing/atari_boxing_latest500_3channels.h5'

# Open the input HDF5 file
with h5py.File(input_path, 'r') as f_in:
    # Load the grayscale frames (N, 64, 64, 1)
    gray_frames = f_in['frames'][:]

    # Convert to RGB by repeating the channel 3 times
    rgb_frames = np.repeat(gray_frames, repeats=3, axis=-1)  # Shape becomes (N, 64, 64, 3)

# Save the new array to a new HDF5 file
with h5py.File(output_path, 'w') as f_out:
    f_out.create_dataset('frames', data=rgb_frames, compression="gzip")

print(f"Converted {gray_frames.shape[0]} frames to RGB and saved to {output_path}")

