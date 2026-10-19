from torchvision import datasets, transforms
import torch.nn as nn

tfm = transforms.ToTensor()
train = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)

def make_mlp():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model

def main():
    print(f"Number of training samples: {len(train)}")
    print(f"Shape of the dataset: {train.data.shape}")
    print(f"Class names: {train.classes}")
    # print(train[0][0])  # Print the first sample (image tensor)
    print(f"Shape of the first sample: {train[0][0].shape}")
    print(f"Label of the first sample: {train[0][1]}")