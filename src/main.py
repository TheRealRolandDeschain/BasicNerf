#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# file: main.py
# Created on : 2023-10-07
# by : Johannes Zellinger
#
#
#
# --- imports -----------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch_device)
    torch.device(torch_device)

    data = np.load("data/tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    print(f"Images shape: {images.shape}")
    print(f"Poses shape: {poses.shape}")
    print(f"Focal length: {focal}")

    # height, width = images.shape[1:3]
    # near, far = 2., 6.

    # n_training = 100
    testimg_idx = 101
    testimg, testpose = images[testimg_idx], poses[testimg_idx]

    print("Pose")
    print(testpose)
    plt.imshow(testimg)
    plt.savefig("results/asdf.png")


if __name__ == "__main__":
    main()
