import random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

accelerator = torch.accelerator.current_accelerator()
DEVICE = accelerator.type if accelerator is not None else "cpu"

SEED = 0
# Batch Size
BS = 128
EPOCHS = 10
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train, test

def load_aug_train():
    # augmentation on the train set only; Normalize comes last
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def print_data_info(train, test):
    print(f"Train dataset size: {len(train)}")
    print(f"Test dataset size: {len(test)}")
    print(f"Number of classes: {len(train.classes)}")
    print(f"Classes: {train.classes}")

def make_loader(data, shuffle):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(data, batch_size=BS, shuffle=shuffle, generator=g, num_workers=2)

def n_params(model):
    return sum(p.numel() for p in model.parameters())

# ---------- Step 1: sanity-check a single convolution ----------

def conv_sanity(loader):
    x, _ = next(iter(loader))
    print("Batch shape:", tuple(x.shape))              # (B, 3, 32, 32)
    for s in (1, 2):
        out = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=s)(x)
        h = (32 + 2 * 1 - 3) // s + 1
        print(f"stride={s}: out={tuple(out.shape)}  "
              f"H' = floor((H + 2p - k)/s) + 1 = floor((32 + 2 - 3)/{s}) + 1 = {h}")

# ---------- Step 2: MLP baseline ----------

def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

# ---------- Step 3: small CNN ----------

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.f(x)

# Optional stretch: third block with 128 channels, pooled to 4x4
class SmallCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
            nn.Flatten(),
            nn.Linear(128*4*4, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.f(x)

def param_counts():
    mlp, cnn, cnn3 = make_mlp(), SmallCNN(), SmallCNN3()
    print(f"MLP params:       {n_params(mlp):,}")
    print(f"SmallCNN params:  {n_params(cnn):,}")
    print(f"  of which conv:  {n_params(cnn.f[:10]):,}  (rest is the 4096->128 head)")
    print(f"SmallCNN3 params: {n_params(cnn3):,}")
    print("Parameter sharing: each 3x3 filter reuses the same few weights at every "
          "spatial position, so conv layers cost 9*C_in*C_out weights regardless of image size.")

# ---------- Step 4: training helper ----------

@torch.no_grad()
def accuracy(model, dl):
    model.eval()
    correct = n = 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n

def train_eval(model, opt, loader_tr, loader_te, epochs, name=""):
    lossf = nn.CrossEntropyLoss()                    # takes raw logits, no softmax in the model
    h = {"train_loss": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, y in loader_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        h["train_loss"].append(tot / n)
        h["test_acc"].append(accuracy(model, loader_te))
        print(f"{name:10s} epoch {ep + 1:2d}  train loss={h['train_loss'][-1]:.3f}  "
              f"test acc={h['test_acc'][-1]:.3f}")
    return h

def run(make_model, train_data, test_data, epochs, name=""):
    set_seed()                                       # same init + shuffling every run
    model = make_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    h = train_eval(model, opt, make_loader(train_data, True),
                   make_loader(test_data, False), epochs, name)
    return model, h

def mlp_vs_cnn(train, test):
    _, mlp_h = run(make_mlp, train, test, EPOCHS, "MLP")
    cnn, cnn_h = run(SmallCNN, train, test, EPOCHS, "SmallCNN")

    ep = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, h in [("MLP", mlp_h), ("SmallCNN", cnn_h)]:
        axes[0].plot(ep, h["train_loss"], marker="o", label=name)
        axes[1].plot(ep, h["test_acc"], marker="o", label=name)
    axes[0].set_ylabel("train loss"); axes[1].set_ylabel("test accuracy")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle("MLP vs CNN on CIFAR-10 (Adam 1e-3)")
    plt.tight_layout(); plt.savefig("src/week_03/w03_mlp_vs_cnn.png", dpi=150)
    print(f"Final test acc: MLP={mlp_h['test_acc'][-1]:.3f}  CNN={cnn_h['test_acc'][-1]:.3f}")
    return cnn, cnn_h

# ---------- Step 5: feature-map inspection ----------

@torch.no_grad()
def feature_maps(model, test, cls="cat"):
    idx = next(i for i, t in enumerate(test.targets) if t == test.class_to_idx[cls])
    x, _ = test[idx]
    model.eval()
    conv1 = model.f[0]
    maps = conv1(x.unsqueeze(0).to(DEVICE))[0].cpu()  # (32, 32, 32)
    w = conv1.weight.cpu()                             # (32, 3, 3, 3)

    img = x * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    fig, axes = plt.subplots(2, 9, figsize=(14, 3.6))
    axes[0, 0].imshow(img.permute(1, 2, 0).clamp(0, 1)); axes[0, 0].set_title(cls)
    axes[1, 0].axis("off")
    for c in range(8):
        axes[0, c + 1].imshow(maps[c], cmap="gray"); axes[0, c + 1].set_title(f"map {c}")
        k = w[c]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)  # rescale filter to [0, 1] for display
        axes[1, c + 1].imshow(k.permute(1, 2, 0)); axes[1, c + 1].set_title(f"filter {c}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("First conv layer: activations (top) and 3x3 filters (bottom)")
    plt.tight_layout(); plt.savefig("src/week_03/w03_maps.png", dpi=150)

# ---------- Step 6: receptive field ----------

def receptive_field(layers=(("conv3x3", 3, 1), ("conv3x3", 3, 1), ("pool2x2", 2, 2))):
    # RF_l = RF_{l-1} + (k_l - 1) * prod_{i<l} s_i
    rf, jump = 1, 1
    print("RF_0 = 1 (a single input pixel)")
    for l, (name, k, s) in enumerate(layers, 1):
        new_rf = rf + (k - 1) * jump
        print(f"RF_{l} = {rf} + ({k} - 1) * {jump} = {new_rf}   [{name}, stride {s}]")
        rf, jump = new_rf, jump * s
    print(f"Receptive field after the first pool: {rf} x {rf} pixels")
    return rf

# ---------- Step 7: light augmentation ----------

def augmented_cnn(test, cnn_h):
    _, aug_h = run(SmallCNN, load_aug_train(), test, EPOCHS, "CNN+aug")
    print(f"CNN test acc: no aug={cnn_h['test_acc'][-1]:.3f}  "
          f"flip+crop={aug_h['test_acc'][-1]:.3f}")
    return aug_h

# ---------- Optional stretch: third conv block ----------

def third_block(train, test, cnn_h):
    _, h3 = run(SmallCNN3, train, test, EPOCHS, "SmallCNN3")
    print(f"Test acc: 2 blocks={cnn_h['test_acc'][-1]:.3f}  3 blocks={h3['test_acc'][-1]:.3f}")
    return h3

def main():
    train, test = load_data()
    print_data_info(train, test)
    conv_sanity(make_loader(train, False))
    param_counts()
    cnn, cnn_h = mlp_vs_cnn(train, test)
    feature_maps(cnn, test)
    receptive_field()
    augmented_cnn(test, cnn_h)
    third_block(train, test, cnn_h)

if __name__ == "__main__":
    main()
