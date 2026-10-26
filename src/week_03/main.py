import torch
from torchvision import datasets, transforms

accelerator = torch.accelerator.current_accelerator()
DEVICE = accelerator.type if accelerator is not None else "cpu"

# Batch Size
BS = 128

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
    ])

    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train, test

def print_data_info(train, test):
    print(f"Train dataset size: {len(train)}")
    print(f"Test dataset size: {len(test)}")
    print(f"Number of classes: {len(train.classes)}")
    print(f"Classes: {train.classes}")


def main():
    train, test = load_data()
    print_data_info(train, test)

if __name__ == "__main__":
    main()