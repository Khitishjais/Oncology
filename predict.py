"""Usage: python predict.py scan.tif [out_mask.png]
Loads the v4 5-fold cross-validation ensemble (5 x U-Net/ResNet50) + tuned threshold,
uses flip test-time augmentation, and writes a black/white tumor mask.
Needs outputs/fold0_model.pth .. fold4_model.pth and outputs/postprocess_v4.json."""
import sys, json, numpy as np, tifffile, torch
from PIL import Image
from common import *

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "predicted_mask.png"

cfg = json.load(open(OUT / "postprocess_v4.json"))          # {'encoder','n_folds','threshold','min_area','fold_files'}
img = tifffile.imread(src).astype(np.float32) / 255.0
x = to_tensor(img)[None]

probs = []
for fname in cfg["fold_files"]:
    model = build_model(pretrained=False, encoder=cfg["encoder"])
    model.load_state_dict(torch.load(OUT / fname, map_location="cpu")); model.eval()
    with torch.no_grad():
        p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(3))).flip(3)
             + torch.sigmoid(model(x.flip(2))).flip(2)) / 3
    probs.append(p[0, 0].numpy())

prob = np.mean(probs, axis=0)
m = prob > cfg["threshold"]
if m.sum() < cfg["min_area"]:
    m[:] = False
Image.fromarray((m * 255).astype(np.uint8)).save(dst)
print(f"saved {dst}, tumor pixels: {int(m.sum())}")
