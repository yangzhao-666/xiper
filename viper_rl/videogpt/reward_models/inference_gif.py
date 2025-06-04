import torch
import torch.nn.functional as F
import numpy as np
import imageio
import os
from tqdm import tqdm
from unet import UNet
import argparse

def load_model(checkpoint_path, device='cuda:0'):
    model = UNet(3, 3, base_factor=48).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_gif(gif_path):
    frames = imageio.mimread(gif_path)
    return np.array(frames)  # shape: (N, H, W, C)

def preprocess(imgs):
    imgs = imgs.astype(np.float32) / 127.5 - 1.0
    imgs = imgs.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return torch.tensor(imgs)

def postprocess(tensor):
    imgs = tensor.detach().cpu().numpy()
    imgs = (imgs + 1.0) * 127.5
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    imgs = imgs.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    return imgs

def save_gif(imgs, out_path, fps=10):
    imageio.mimsave(out_path, imgs, fps=fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to input GIF', default='./gray_boxing.gif')
    parser.add_argument('--checkpoint', help='Path to model checkpoint', default='/home/zyang/NeuralOptimalTransport/checkpoints/mse/atari/SN_TN_64/0_999.pt')
    parser.add_argument('--output', help='Path to output GIF', default='output.gif')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)

    imgs = load_gif(args.input)
    input_tensor = preprocess(imgs).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_imgs = postprocess(output_tensor)
    save_gif(output_imgs, args.output)

if __name__ == '__main__':
    main()
