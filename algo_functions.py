import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch_dct as dct
from torchmetrics.image import PeakSignalNoiseRatio
import os
from tqdm import tqdm

from model_unet_dist import BFBatchNorm2d

from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from deepinv.physics.generator import (
    GaussianMaskGenerator,
    RandomMaskGenerator,
    EquispacedMaskGenerator,
)
import cv2

import deepinv as dinv


def get_physics(args):
    """Build student/teacher (and optional mentor) physics operators based on args.physics."""
    if args.physics == "cs":  # single-pixel
        if args.cr_teacher == 1:
            physics_t = []
            for cr in args.crs_expert_teacher:
                mt = int(args.n**2 * cr)
                one_physics = dinv.physics.SinglePixelCamera(
                    m=mt, img_shape=(1, args.n, args.n), device=args.device, fast=True
                )
                physics_t.append(one_physics)
        else:
            mt = int(args.n**2 * args.cr_teacher)
            physics_t = dinv.physics.SinglePixelCamera(
                m=mt, img_shape=(1, args.n, args.n), device=args.device, fast=True
            )
        if hasattr(args, 'cr_mentor'):
            if args.cr_mentor == 1:
                physics_m = []
                for cr in args.crs_expert_mentor:
                    mm = int(args.n**2 * cr)
                    one_physics = dinv.physics.SinglePixelCamera(
                        m=mm, img_shape=(1, args.n, args.n), device=args.device, fast=True
                    )
                    physics_m.append(one_physics)
            else:
                mm = int(args.n**2 * args.cr_mentor)
                physics_m = dinv.physics.SinglePixelCamera(
                    m=mm, img_shape=(1, args.n, args.n), device=args.device, fast=True
                )
        ms = int(args.n**2 * args.cr_student)
        physics_s = dinv.physics.SinglePixelCamera(
            m=ms, img_shape=(1, args.n, args.n), device=args.device, fast=True
        )
    elif args.physics == "fmri":
        mask_path_s = (
            f"masks/mask_s_{args.mask_s}_as_{args.acceleration_s}_{args.n}.pth"
        )
        mask_path_t = (
            f"masks/mask_t_{args.mask_t}_at_{args.acceleration_t}_n_{args.n}.pth"
        )
        if os.path.exists(mask_path_s) and os.path.exists(mask_path_t):
            mask_s = torch.load(mask_path_s)
            mask_t = torch.load(mask_path_t)
        else:  # Generate and cache masks
            if args.mask_s == "gaussian":
                mask_s = GaussianMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_s
                ).step()["mask"]
            elif args.mask_s == "uniform":
                mask_s = EquispacedMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_s
                ).step()["mask"]
            elif args.mask_s == "random":
                mask_s = RandomMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_s
                ).step()["mask"]
            if args.mask_t == "gaussian":
                mask_t = GaussianMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_t
                ).step()["mask"]
            elif args.mask_s == "uniform":
                mask_t = EquispacedMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_t
                ).step()["mask"]
            elif args.mask_t == "random":
                mask_t = RandomMaskGenerator(
                    (args.n, args.n), acceleration=args.acceleration_t
                ).step()["mask"]
            torch.save(mask_s, mask_path_s)
            torch.save(mask_t, mask_path_t)
        physics_s = dinv.physics.MRI(mask=mask_s, device=args.device)
        physics_t = dinv.physics.MRI(mask=mask_t, device=args.device)
    elif args.physics == "sr":  # Super-resolution
        filter = dinv.physics.blur.gaussian_blur()
        physics_s = dinv.physics.Downsampling(
            factor=args.srf_student,
            filter=filter,
            device=args.device,
            img_size=(args.c, args.n, args.n),
        )
        physics_t = dinv.physics.Downsampling(
            factor=args.srf_teacher,
            filter=filter,
            device=args.device,
            img_size=(args.c, args.n, args.n),
        )
    if hasattr(args, 'cr_mentor'):
        return physics_s, physics_t, physics_m
    else:
        return physics_s, physics_t

def get_device(preferred_device):
    """Select CUDA device if available else exit (placeholder)."""
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        if preferred_device < total_gpus:
            return torch.device(f"cuda:{preferred_device}")
        else:
            exit()
    else:
        exit()

def conv_block(ch_in, ch_out, batch_norm=True, bias=True, circular_padding=False, biasfree=False):
    """Two-layer Conv2d block with optional batch norm / bias-free batch norm."""
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                padding_mode="circular" if circular_padding else "zeros",
            ),
            (
                BFBatchNorm2d(ch_out, use_bias=bias)
                if biasfree
                else nn.BatchNorm2d(ch_out)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            (
                BFBatchNorm2d(ch_out, use_bias=bias)
                if biasfree
                else nn.BatchNorm2d(ch_out)
            ),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                padding_mode="circular" if circular_padding else "zeros",
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.ReLU(inplace=True),
        )

class PNGImageDataset(Dataset):
    """Dataset to load grayscale PNG images from directory."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [
            img for img in os.listdir(image_dir) if img.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.fromarray(np.array(cv2.imread(img_path))).convert("L")
        if self.transform:
            image = self.transform(image)
        return image


def hadamard(n):
    """Recursive Hadamard matrix generator (power-of-two size)."""
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])


def cake_cutting_seq(i, p):
    """Return i-th sequence for cake-cutting ordering with period p."""
    step = -i * (-1) ** (np.mod(i, 2))
    if np.mod(i, 2) == 1:
        seq = list(range(i, i * p + 1, step))
    else:
        seq = list(range(i * p, i - 1, step))
    return seq


def rgb2gray(rgb):
    """Convert RGB ndarray to grayscale (float)."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def normalize_img(x):
    """Normalize tensor/array to [0,1]."""
    val_min = x.min(); val_max = x.max()
    return (x - val_min) / (val_max - val_min)


def rescale(input_tensor, new_min, new_max):
    """Rescale tensor to new [new_min, new_max] range."""
    current_min = torch.min(input_tensor)
    current_max = torch.max(input_tensor)
    return (input_tensor - current_min) / (current_max - current_min) * (
        new_max - new_min
    ) + new_min


def cake_cutting_order(n):
    """Return permutation implementing cake-cutting sampling order for n elements."""
    p = int(np.sqrt(n))
    seq = [cake_cutting_seq(i, p) for i in range(1, p + 1)]
    seq = [item for sublist in seq for item in sublist]
    return np.argsort(seq)

def save_npy_metric(file, metric_name):
    """Save metric array to .npy with supplied name."""
    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)


def compute_gradient(theta_old, y, A):
    """Gradient for sparse DCT formulation (fashioned for 28x28)."""
    psi_theta = dct.idct_2d(theta_old.view(-1, 1, 28, 28)).reshape(-1, 784)
    grad = (psi_theta @ A.T - y) @ A
    grad = grad.reshape(-1, 1, 28, 28)
    grad = dct.dct_2d(grad).reshape(-1, 784)
    return grad

class AverageMeter(object):
    """Keeps running average of a scalar metric."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AddGaussianNoiseSNR:
    """Add AWGN at specified SNR (per-image)."""
    def __init__(self, snr):
        self.snr = snr
    def __call__(self, images):
        device = images.device
        if images.min() < 0:
            images = images * 0.5 + 0.5  # Assume in [-1,1] → [0,1]
        signal_power = torch.mean(images**2, dim=(1,2,3), keepdim=True)
        snr_linear = 10 ** (self.snr / 10)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(images, device=device) * noise_std
        noisy_images = images + noise
        return torch.clamp(noisy_images, 0.0, 1.0)