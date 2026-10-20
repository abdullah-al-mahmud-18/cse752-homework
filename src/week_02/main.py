import math, random
import numpy as np
from torchvision import datasets, transforms
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

SEED = 0
BS = 128
EPOCHS, EPOCHS_SMALL = 25, 8
N_TR, N_TE, N_SMALL = 8000, 2000, 1000

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

tr = datasets.FashionMNIST(root='./data', train=True, download=True)
te = datasets.FashionMNIST(root='./data', train=False, download=True)
Xtr = tr.data[:N_TR].float().div(255).unsqueeze(1)
ytr = tr.targets[:N_TR]
Xte = te.data[:N_TE].float().div(255).unsqueeze(1)
yte = te.targets[:N_TE]
FULL = (Xtr, ytr)
SMALL = (Xtr[:N_SMALL], ytr[:N_SMALL])
TEST = (Xte, yte)



def make_mlp():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model

def main():
    print("ANYTHING")
    # print(f"Number of training samples: {len(train)}")
    # print(f"Shape of the dataset: {train.data.shape}")
    # print(f"Class names: {train.classes}")
    # # print(train[0][0])  # Print the first sample (image tensor)
    # print(f"Shape of the first sample: {train[0][0].shape}")
    # print(f"Label of the first sample: {train[0][1]}")