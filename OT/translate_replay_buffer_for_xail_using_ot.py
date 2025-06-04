import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from tqdm import tqdm
from src.unet import UNet
import argparse
import imageio

def load_model(checkpoint_path, device='cuda:0'):
    model = UNet(3, 3, base_factor=48).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess(imgs):
    imgs = imgs.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    imgs = imgs.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return torch.tensor(imgs)

def postprocess(tensor):
    imgs = tensor.detach().cpu().numpy()
    imgs = (imgs + 1.0) * 127.5
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    imgs = imgs.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    return imgs

def write_video(imgs, out_path, fps=24):
    h, w, c = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for img in imgs:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()

def write_gif(imgs, out_path, fps=24):
    duration = 1 / fps
    imageio.mimsave(out_path, imgs, duration=duration)

def process_npz_directory(model, input_dir, output_dir, device, fps=24, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    for file_name in tqdm(npz_files, desc="Processing .npz files"):
        path = os.path.join(input_dir, file_name)
        data = np.load(path, allow_pickle=True)
        
        original_imgs = data['image']
        input_tensor = preprocess(original_imgs).to(device)
        
        outputs = []
        with torch.no_grad():
            for i in range(0, len(input_tensor), batch_size):
                batch = input_tensor[i:i+batch_size]
                out = model(batch)
                outputs.append(postprocess(out))
        processed_imgs = np.concatenate(outputs, axis=0)

        # Save original + processed GIFs
        base = os.path.splitext(file_name)[0]

        # Update only the 'image' key, keep others intact
        updated_dict = {key: (processed_imgs if key == 'image' else data[key]) for key in data.files}
        np.savez(path, **updated_dict)
    write_gif(original_imgs, os.path.join(output_dir, f'{base}_original.gif'), fps)
    write_gif(processed_imgs, os.path.join(output_dir, f'{base}_processed.gif'), fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/zyang2/XViper/OT/checkpoints_dmc_bigger_xail/mse/cartpole_balance/SN_TN_64/T_5999.pth')
    parser.add_argument('--input_dir', type=str, default='/home/zyang2/XViper/xail_data/dmc_cartpole_balance_body/')
    parser.add_argument('--output_dir', type=str, default='/home/zyang2/XViper/xail_data/')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model...")
    model = load_model(args.model_path, device)
    print("Running inference...")
    process_npz_directory(model, args.input_dir, args.output_dir, device, args.fps, args.batch_size)
    print("Done!")

