"""Prepare reach_controller dataset by symlinking mp4 files into train/test splits."""

import os
import random
from pathlib import Path

# Paths
SRC_DIR = Path("/home/zyang2/XViper/ot_data/reach_controller")
DST_DIR = Path("viper_rl_data/datasets/reach_controller")
TRAIN_DIR = DST_DIR / "train"
TEST_DIR = DST_DIR / "test"

# Reproducible split
SEED = 42
TRAIN_RATIO = 0.9

# Get all mp4 files
mp4_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".mp4")])
print(f"Found {len(mp4_files)} mp4 files in {SRC_DIR}")

# Shuffle deterministically
random.seed(SEED)
random.shuffle(mp4_files)

# Split
split_idx = int(len(mp4_files) * TRAIN_RATIO)
train_files = mp4_files[:split_idx]
test_files = mp4_files[split_idx:]

print(f"Train: {len(train_files)}, Test: {len(test_files)}")

# Create symlinks
for f in train_files:
    src = SRC_DIR / f
    dst = TRAIN_DIR / f
    if not dst.exists():
        os.symlink(src, dst)

for f in test_files:
    src = SRC_DIR / f
    dst = TEST_DIR / f
    if not dst.exists():
        os.symlink(src, dst)

print("Done. Symlinks created.")
print(f"  Train: {len(os.listdir(TRAIN_DIR))} files")
print(f"  Test:  {len(os.listdir(TEST_DIR))} files")
