"""Steps 4-6: augmentation, U-Net (resnet34/imagenet), Dice+BCE, Adam 1e-4, early stopping, best-weights."""
import argparse, json, time
import numpy as np, torch, albumentations as A
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--patience", type=int, default=7)
ap.add_argument("--limit", type=int, default=0, help="use only N training slices (smoke test)")
a = ap.parse_args()
torch.manual_seed(SEED); np.random.seed(SEED)

# Same geometric transform hits scan and mask; brightness/contrast touches the scan only (albumentations does this).
aug = A.Compose([
    A.HorizontalFlip(), A.VerticalFlip(),
    A.Rotate(limit=15, interpolation=1, mask_interpolation=0, border_mode=0, p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
])


class DS(Dataset):
    def __init__(self, name, train):
        d = np.load(OUT / f"{name}.npz"); self.x, self.y, self.train = d["x"], d["y"], train
        if train and a.limit:
            idx = np.random.permutation(len(self.x))[:a.limit]
            self.x, self.y = self.x[idx], self.y[idx]

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        img, m = self.x[i].astype(np.float32), self.y[i]
        if self.train:
            r = aug(image=img, mask=m); img, m = np.clip(r["image"], 0, 1), r["mask"]
        return to_tensor(img), torch.from_numpy(m.astype(np.float32))[None]


tl = DataLoader(DS("train", True), a.batch, shuffle=True, num_workers=0)
vl = DataLoader(DS("val", False), a.batch, num_workers=0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, loss_fn = build_model().to(dev), DiceBCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
hist, best, bad = {"train": [], "val": []}, 1e9, 0
print(f"device={dev} train={len(tl.dataset)} val={len(vl.dataset)}", flush=True)

for ep in range(a.epochs):
    t0 = time.time(); model.train(); tot = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad()
        l = loss_fn(model(x), y); l.backward(); opt.step(); tot += l.item() * len(x)
    tr = tot / len(tl.dataset)
    model.eval(); tot = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(dev), y.to(dev); tot += loss_fn(model(x), y).item() * len(x)
    va = tot / len(vl.dataset); hist["train"].append(tr); hist["val"].append(va)
    star = ""
    if va < best:
        best, bad, star = va, 0, " *saved"; torch.save(model.state_dict(), OUT / "best_model.pth")
    else:
        bad += 1
    print(f"epoch {ep+1}/{a.epochs} train={tr:.4f} val={va:.4f} ({time.time()-t0:.0f}s){star}", flush=True)
    json.dump(hist, open(OUT / "history.json", "w"))
    if bad >= a.patience:
        print("early stopping"); break

plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
plt.xlabel("epoch"); plt.ylabel("Dice+BCE loss"); plt.legend(); plt.savefig(OUT / "loss_curve.png", dpi=100)
