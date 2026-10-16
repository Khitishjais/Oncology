"""Step 7: test-set Dice/IoU + 20 side-by-side visualisations."""
import numpy as np, torch, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

d = np.load(OUT / "test.npz"); X, Y = d["x"], d["y"]
model = build_model(pretrained=False)
model.load_state_dict(torch.load(OUT / "best_model.pth", map_location="cpu")); model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(X), 32):
        xb = torch.stack([to_tensor(x.astype(np.float32)) for x in X[i:i+32]])
        preds.append((torch.sigmoid(model(xb))[:, 0] > 0.5).numpy())
P = np.concatenate(preds)
sc = np.array([dice_iou(p, y) for p, y in zip(P, Y)]); has = Y.reshape(len(Y), -1).any(1)
rep = (f"Test slices: {len(Y)} ({has.sum()} with tumor)\n"
       f"All slices      Dice={sc[:,0].mean():.4f}  IoU={sc[:,1].mean():.4f}\n"
       f"Tumor slices    Dice={sc[has,0].mean():.4f}  IoU={sc[has,1].mean():.4f}\n"
       f"(empty-vs-empty slices scored as 1.0 in 'All slices')\n")
print(rep); (OUT / "evaluation_report.txt").write_text(rep)

random.seed(SEED); idx = random.sample(range(len(Y)), 20)
fig, ax = plt.subplots(20, 3, figsize=(7, 46))
for r, i in enumerate(idx):
    for c, (im, t, cm) in enumerate([(X[i].astype(np.float32), "Scan", None), (Y[i], "True", "gray"),
                                     (P[i], f"Pred (Dice {sc[i,0]:.2f})", "gray")]):
        ax[r, c].imshow(im, cmap=cm); ax[r, c].set_title(t, fontsize=7); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "sample_predictions.png", dpi=80)
