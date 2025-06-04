import torch
import torch.nn.functional as F
import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm
from unet import UNet
import argparse

def load_model(checkpoint_path, device='cuda:0'):
    model = UNet(3, 3, base_factor=48).cuda()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_hdf5_images(hdf5_path, key='frames'):
    with h5py.File(hdf5_path, 'r') as f:
        data = f[key][:]
    return data  # shape: (N, H, W, C)

def preprocess(imgs):
    # Normalize to [-1, 1] assuming input is uint8 [0, 255]
    imgs = imgs.astype(np.float32) / 127.5 - 1.0
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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    T = load_model(args.model_path, device)

    # Load data
    print("Loading HDF5 images...")
    input_images = load_hdf5_images(args.input_hdf5, key=args.key)
    input_tensor = preprocess(input_images).to(device)

    # Run inference (batched)
    print("Running inference...")
    batch_size = args.batch_size
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(input_tensor), batch_size)):
            batch = input_tensor[i:i+batch_size]
            out = T(batch)
            outputs.append(postprocess(out))
    outputs = np.concatenate(outputs, axis=0)

    # Save video
    print(f"Writing output video to {args.output_video}...")
    write_video(outputs, args.output_video, fps=args.fps)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/zyang/NeuralOptimalTransport/checkpoints/mse/dmc/SN_TN_64/0_7999.pt')
    parser.add_argument('--input_hdf5', type=str, default='/home/zyang/viper_rl/ot_data/target.h5')
    parser.add_argument('--key', type=str, default='frames', help='Key in HDF5 file')
    parser.add_argument('--output_video', type=str, default='ot_translated_expert.mp4')
    parser.add_argument('--fps', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(args)

