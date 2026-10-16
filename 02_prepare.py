"""Steps 2-3: load, normalise, binarise, split BY PATIENT (70/15/15, seed 42), save .npz."""
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from common import *

pairs = collect_pairs()
pids = sorted({p for p, _, _ in pairs})
train_p, rest = train_test_split(pids, test_size=0.30, random_state=SEED)
val_p, test_p = train_test_split(rest, test_size=0.50, random_state=SEED)
split = {**{p: "train" for p in train_p}, **{p: "val" for p in val_p}, **{p: "test" for p in test_p}}
assert not (set(train_p) & set(val_p) or set(train_p) & set(test_p) or set(val_p) & set(test_p))

buckets = {k: ([], [], []) for k in ("train", "val", "test")}
for pid, s, m in tqdm(pairs):
    img, mask = load_pair(s, m)
    X, Y, N = buckets[split[pid]]
    X.append(img.astype(np.float16)); Y.append(mask.astype(np.uint8)); N.append(s.stem)

for k, (X, Y, N) in buckets.items():
    np.savez(OUT / f"{k}.npz", x=np.stack(X), y=np.stack(Y), names=np.array(N))
    print(f"{k}: patients={len({n.rsplit('_',1)[0] for n in N})} slices={len(X)} tumor slices={sum(y.any() for y in Y)}")
