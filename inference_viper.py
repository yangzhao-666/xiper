import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.append(os.path.dirname(os.path.abspath("")))

import tqdm
import numpy as np

#import notebook_utils as nbu
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

from PIL import Image
import numpy as np

def gif_to_seq(gif_path, camera_key='image') -> list:
    img = Image.open(gif_path)
    frames = []

    try:
        while True:
            frame = img.convert('RGB')  # Ensure it's in RGB
            frame_array = np.array(frame)  # shape (H, W, 3)
            frames.append(frame_array)
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of GIF

    seq = []
    for i, frame_array in enumerate(frames):
        timestep = {
            camera_key: frame_array,
            'is_first': i == 0
        }
        seq.append(timestep)

    return seq


print('Available reward models:', LOAD_REWARD_MODEL_DICT.keys())

rm_key = 'dmc_clen16_fskip4'

reward_model = LOAD_REWARD_MODEL_DICT[rm_key](task='dmc_cartpole_balance', minibatch_size=2, encoding_minibatch_size=32, compute_joint=True, use_ot=True, ot_path="/home/zyang/NeuralOptimalTransport/checkpoints/mse/dmc/SN_TN_64/0_7999.pt")

# Directory containing .gif files
gif_dir = './good_bad_videos/dmc_cartpole_balance/bad_videos/'
reward_list = []

# Process each .gif
for file_name in os.listdir(gif_dir):
    if file_name.endswith('.gif'):
        gif_path = os.path.join(gif_dir, file_name)
        print(f"Processing: {gif_path}")

        seq = gif_to_seq(gif_path)
        viper_seq = reward_model(seq)

        rewards = [s['density'] for s in viper_seq]
        reward_list.append(rewards[:800])

# Save the full reward list
np.save('bad_cartpole_balance_rewards_ot.npy', reward_list)
print("Saved all rewards.")
