import torch
import numpy as np
from PIL import Image, ImageSequence
import argparse
from tqdm import tqdm
from src.unet import UNet

def load_model(checkpoint_path, device='cuda:0'):
    model = UNet(3, 3, base_factor=48).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def gif_to_frames(gif_path, image_size=(64, 64)):
    img = Image.open(gif_path)
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(img):
        rgb = frame.convert("RGB").resize(image_size)
        frames.append(np.array(rgb))
        durations.append(frame.info.get('duration', 40))  # default to 40ms if not found

    return np.stack(frames), durations

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

def save_as_gif(frames, durations, output_path):
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0
    )

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = load_model(args.model_path, device)

    print("Loading GIF and extracting frames...")
    input_images, durations = gif_to_frames(args.input_gif, image_size=(64, 64))
    input_tensor = preprocess(input_images).to(device)

    print("Running inference...")
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(input_tensor), args.batch_size)):
            batch = input_tensor[i:i+args.batch_size]
            out = model(batch)
            outputs.append(postprocess(out))
    outputs = np.concatenate(outputs, axis=0)

    print(f"Saving processed GIF to {args.output_gif}...")
    save_as_gif(outputs, durations, args.output_gif)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/zyang2/XViper/OT/checkpoints_R/mse/atari_boxing/SN_TN_64/T_7999.pth")
    parser.add_argument('--input_gif', type=str, default="/home/zyang2/XViper/analysis_videos/atari_boxing/good_videos/good_1.gif")
    parser.add_argument('--output_gif', type=str, default="/home/zyang2/XViper/analysis_videos/atari_boxing/translated_good_videos/good_1.gif")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(args)

