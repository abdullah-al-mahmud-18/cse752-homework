import math, random
import numpy as np
from torchvision import datasets, transforms
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import matplotlib.pyplot as plt

SEED = 0
BS = 128
EPOCHS, EPOCHS_SMALL = 25, 8
N_TR, N_TE, N_SMALL = 8000, 2000, 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def load_data():
    tr = datasets.FashionMNIST(root='./data', train=True, download=True)
    te = datasets.FashionMNIST(root='./data', train=False, download=True)
    Xtr = tr.data[:N_TR].float().div(255).unsqueeze(1)
    ytr = tr.targets[:N_TR]
    Xte = te.data[:N_TE].float().div(255).unsqueeze(1)
    yte = te.targets[:N_TE]
    full = (Xtr, ytr)
    small = (Xtr[:N_SMALL], ytr[:N_SMALL])
    test = (Xte, yte)
    return full, small, test

def make_loader(data, shuffle):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(TensorDataset(*data), batch_size=BS, shuffle=shuffle, generator=g)

# ---------- Step 1: model factories ----------

def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def make_mlp_dropout(p=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(128, 10)
    )

def make_mlp_bn():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# ---------- Step 2: training helper ----------

@torch.no_grad()
def accuracy(model, dl):
    model.eval()
    correct = n = 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n

def train_eval(model, opt, loader_tr, loader_te, epochs, sched=None):
    lossf = nn.CrossEntropyLoss()
    h = {"train_loss": [], "train_acc": [], "test_acc": [], "lr": []}
    for _ in range(epochs):
        model.train()
        h["lr"].append(opt.param_groups[0]["lr"])   # lr used during this epoch
        tot = n = 0
        for x, y in loader_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        if sched is not None:
            sched.step()                             # once per epoch
        h["train_loss"].append(tot / n)
        h["train_acc"].append(accuracy(model, loader_tr))
        h["test_acc"].append(accuracy(model, loader_te))
    return h

def run(make_model, make_opt, train_data, test_data, epochs, make_sched=None):
    set_seed()                                       # same init + shuffling every run
    model = make_model().to(DEVICE)
    opt = make_opt(model.parameters())
    sched = make_sched(opt) if make_sched else None
    return train_eval(model, opt, make_loader(train_data, True),
                      make_loader(test_data, False), epochs, sched)

SGD      = lambda p: torch.optim.SGD(p, lr=0.1)
SGD_MOM  = lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)
SGD_NES  = lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)
ADAM     = lambda p: torch.optim.Adam(p, lr=1e-3)
STEP     = lambda o: StepLR(o, step_size=5, gamma=0.1)

# ---------- Step 3: optimizer sweep ----------

def optimizer_sweep(full, test):
    opt_runs = {
        "SGD (0.1)":            run(make_mlp, SGD,     full, test, EPOCHS),
        "SGD+mom (0.05)":       run(make_mlp, SGD_MOM, full, test, EPOCHS),
        "SGD+mom+Nesterov":     run(make_mlp, SGD_NES, full, test, EPOCHS),
        "Adam (1e-3)":          run(make_mlp, ADAM,    full, test, EPOCHS),
    }
    plt.figure(figsize=(6, 4))
    for name, h in opt_runs.items():
        plt.plot(range(1, EPOCHS + 1), h["train_loss"], marker="o", label=name)
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Optimizer sweep"); plt.tight_layout(); plt.savefig("src/week_02/w02_optimizers.png", dpi=150)
    for name, h in opt_runs.items():
        print(f"{name:20s} epoch1 loss={h['train_loss'][0]:.3f}  "
              f"epoch1 test acc={h['test_acc'][0]:.3f}  final test acc={h['test_acc'][-1]:.3f}")
    return opt_runs

# ---------- Step 4: LR schedules (SGD+momentum, lr=0.1) ----------

def lr_schedules(full, test):
    SGD_MOM_01 = lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)
    sched_runs = {
        "constant": run(make_mlp, SGD_MOM_01, full, test, EPOCHS),
        "StepLR":   run(make_mlp, SGD_MOM_01, full, test, EPOCHS, STEP),
        "Cosine":   run(make_mlp, SGD_MOM_01, full, test, EPOCHS,
                        lambda o: CosineAnnealingLR(o, T_max=EPOCHS)),
    }
    for name, h in sched_runs.items():
        print(f"{name:9s} final test acc={h['test_acc'][-1]:.4f}  last-epoch lr={h['lr'][-1]:.5f}")
    return sched_runs

# ---------- Steps 5-6: overfitting and dropout (1,000 images, 25 epochs) ----------

def overfit_and_dropout(small, test):
    small_base = run(make_mlp,         ADAM, small, test, EPOCHS_SMALL)
    small_drop = run(make_mlp_dropout, ADAM, small, test, EPOCHS_SMALL)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, h) in zip(axes, [("Adam, no regularizer", small_base),
                                     ("Adam + Dropout 0.5", small_drop)]):
        ax.plot(h["train_acc"], label="train"); ax.plot(h["test_acc"], label="test")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes[0].set_ylabel("accuracy"); axes[0].legend()
    plt.tight_layout(); plt.savefig("src/week_02/w02_overfit.png", dpi=150)
    return small_base, small_drop

# ---------- Step 7: BatchNorm (SGD) on the 8,000 set ----------

def batchnorm(full, small, test, plain):
    bn_01 = run(make_mlp_bn, SGD,                          full, test, EPOCHS)
    bn_03 = run(make_mlp_bn, lambda p: torch.optim.SGD(p, lr=0.3), full, test, EPOCHS)
    print("Plain SGD 0.1 loss by epoch:", [round(v, 3) for v in plain["train_loss"]])
    print("BN    SGD 0.1 loss by epoch:", [round(v, 3) for v in bn_01["train_loss"]])
    print("BN    SGD 0.3 loss by epoch:", [round(v, 3) for v in bn_03["train_loss"]],
          "| diverged:", not math.isfinite(bn_03["train_loss"][-1]))

    # BN on the small set with Adam, so it is comparable with the overfit baseline
    return run(make_mlp_bn, ADAM, small, test, EPOCHS_SMALL)

# ---------- Step 8: ablation table ----------

def ablation_table(full, test, opt_runs, small_base, small_drop, small_bn):
    adam_step = run(make_mlp, ADAM, full, test, EPOCHS, STEP)
    rows = [
        ("SGD",               opt_runs["SGD (0.1)"],      "8k, 8 ep, lr 0.1"),
        ("SGD+mom",           opt_runs["SGD+mom (0.05)"], "8k, 8 ep, lr 0.05"),
        ("Adam",              opt_runs["Adam (1e-3)"],    "8k, 8 ep"),
        ("Adam + step decay", adam_step,                  "8k, 8 ep, x0.1 at ep 5"),
        ("Small-data Adam",   small_base,                 "1k, 25 ep, overfit baseline"),
        ("+ Dropout 0.5",     small_drop,                 "1k, 25 ep"),
        ("+ BatchNorm",       small_bn,                   "1k, 25 ep, Adam"),
    ]
    print("\n| Setup | Train acc | Test acc | Gap | Note |\n|---|---|---|---|---|")
    for name, h, note in rows:
        a, b = h["train_acc"][-1], h["test_acc"][-1]
        print(f"| {name} | {a:.3f} | {b:.3f} | {a - b:.3f} | {note} |")

# ---------- Optional stretch: AdamW weight decay vs dropout-only ----------

def adamw_stretch(small, test, small_base, small_drop):
    adamw = run(make_mlp, lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-4),
                small, test, EPOCHS_SMALL)
    for name, h in [("Adam baseline", small_base), ("Dropout only", small_drop), ("AdamW wd=1e-4", adamw)]:
        a, b = h["train_acc"][-1], h["test_acc"][-1]
        print(f"{name:15s} train={a:.3f} test={b:.3f} gap={a - b:.3f}")

def main():
    full, small, test = load_data()
    opt_runs = optimizer_sweep(full, test)
    lr_schedules(full, test)
    small_base, small_drop = overfit_and_dropout(small, test)
    small_bn = batchnorm(full, small, test, opt_runs["SGD (0.1)"])
    ablation_table(full, test, opt_runs, small_base, small_drop, small_bn)
    adamw_stretch(small, test, small_base, small_drop)

if __name__ == "__main__":
    main()
