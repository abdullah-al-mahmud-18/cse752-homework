import torch
from torchvision import datasets, transforms

accelerator = torch.accelerator.current_accelerator()
DEVICE = accelerator.device if accelerator is not None else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
])

train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def main():
    print("Running week 03 main.py")

if __name__ == "__main__":
    main()