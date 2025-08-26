import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch_dct as dct
from torchmetrics.image import PeakSignalNoiseRatio
import os
from tqdm import tqdm
from algo_functions import *
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
import deepinv as dinv
from deepinv.utils.demo import load_dataset, load_degradation
from pathlib import Path
from deepinv.datasets import SimpleFastMRISliceDataset
from torchvision.datasets import CelebA
from torch.utils.data import Subset


class SimpleImageLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [
            f
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(".png")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("L")  # Convert image to grayscale

        if self.transform:
            image = self.transform(image)

        return image, 0  # Returns a dummy label since there are no classes

def get_dataloaders(args):
    """Return train/test dataloaders for selected dataset in args.dataset."""
    transform = transforms.Compose(
        [transforms.Resize((args.n, args.n), antialias=True),
        transforms.ToTensor()]
    )

    if args.dataset == "mnist":
        trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True,
        )
        testset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False
        )

    elif args.dataset == "celeba":
        trainset = datasets.CelebA(
            root="./data",
            split="train",
            download=False,
            transform=transform
        )

        trainset = Subset(trainset, list(range(20000)))  # Subsample for speed

        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )

        testset = datasets.CelebA(
            root="./data",
            split="test",
            download=False,
            transform=transform
        )

        testset = Subset(testset, list(range(2000)))

        testloader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False
        )

    elif args.dataset == "BSDS500":

        train_dir = r"data\BSDS500\train"
        trainloadr = PNGImageDataset(train_dir, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainloadr, batch_size=args.batch_size, shuffle=False
        )
        test_dir = r"data\BSDS500\test"
        testloadr = PNGImageDataset(test_dir, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testloadr, batch_size=args.batch_size, shuffle=False
        )
    elif args.dataset == "ct":

        dataset = PNGImageDataset(args.dataset_path, transform=transform)
        # Split the dataset into training, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        trainset, valset, testset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        # Create data loaders for training, validation, and test sets
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    elif args.dataset == "fastmri":

        transform = transforms.Compose([transforms.Resize(args.n, antialias=True)])

        # train_dataset_name = "fastmri_knee_singlecoil"
        dataset_path = Path("data/fastmri_knee_singlecoil")

        train_dataset = SimpleFastMRISliceDataset(
            root_dir=dataset_path, transform=transform, download=True, 
            train=True, train_percent = 0.8
        )

        test_dataset = SimpleFastMRISliceDataset(
            root_dir=dataset_path, transform=transform, download=False, 
            train=False, train_percent = 0.8
        )
        
        print(len(train_dataset), len(test_dataset))

        trainloader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True
        )
        testloader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
        )


    return trainloader, testloader