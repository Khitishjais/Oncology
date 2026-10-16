"""Step 1: count data and plot scan/mask pairs."""
import random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

pairs = collect_pairs()
patients = {p for p, _, _ in pairs}
nonempty = [x for x in pairs if load_pair(x[1], x[2])[1].any()]
print(f"patients={len(patients)}  slices={len(pairs)}  with tumor={len(nonempty)}  empty={len(pairs)-len(nonempty)}")

random.seed(SEED)
sample = random.sample(nonempty, 6) + random.sample(pairs, 2)
fig, ax = plt.subplots(len(sample), 2, figsize=(5, 2.5 * len(sample)))
for i, (_, s, m) in enumerate(sample):
    img, mask = load_pair(s, m)
    ax[i, 0].imshow(img); ax[i, 1].imshow(mask, cmap="gray")
    ax[i, 0].set_title(s.stem, fontsize=6); ax[i, 0].axis("off"); ax[i, 1].axis("off")
plt.tight_layout(); plt.savefig(OUT / "explore.png", dpi=100)
