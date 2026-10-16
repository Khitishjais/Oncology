"""Generates colab_train_v6.ipynb: pushes for a higher tumor-slice Dice using two NEW diverse models
(stronger encoder + Focal-Tversky-loss architecture), then builds the biggest ensemble yet by combining
them with the existing v4 5-fold models (loaded, not retrained). Same train(77)/val(16)/test(17) patient
split as v1/v2/v3/v5 for a clean apples-to-apples comparison.

Honest framing (see PROCESS_LOG.md Stage 12): v1->v4 already moved tumor Dice 0.62 -> 0.79 and PROCESS_LOG's
own v4 conclusion says further tweaks are expected to give only marginal (low single-digit %) gains from
here. This notebook is that next marginal attempt, not a promise of reaching 0.95 - published results on
this exact dataset top out around 0.82-0.90.
"""
import json

v2 = json.load(open("colab_train_v2.ipynb"))
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# TumorVision v6 - stronger encoder + Focal Tversky, mega-ensemble with v4
Runtime > Change runtime type > **T4 GPU**. Needs `MyDrive/oncology/archive.zip` and, for the mega-ensemble
cell, `MyDrive/oncology/fold0_model.pth` .. `fold4_model.pth` (from v4).

**Goal:** push tumor-slice Dice as far as realistically possible past v4's 0.7867, by adding two models that
are diverse from everything already trained:
1. `m6_effnet` - **EfficientNet-B4** encoder (stronger/deeper than the ResNet50 used everywhere else so far),
   Dice+BCE loss.
2. `m6_unetpp_ft` - **Unet++** architecture (already used in v3) + **Focal Tversky loss** (already tried alone
   in v5) - an architecture+loss combination not tried before.

Both trained on the same 77/16/17 patient split as v1/v2/v3/v5 (seed 42) so they are a clean comparison.
Each is saved to Drive as soon as it finishes (re-running skips finished models, same as v3).

**Honest expectation (see PROCESS_LOG.md Stage 12):** v1->v4 already moved tumor Dice from 0.62 to 0.79;
that stage's own conclusion was that further tweaks give only marginal (low single-digit percent) gains.
This is that attempt - realistic target is ~0.80-0.83, not 0.95. Published results on this exact dataset
(LGG MRI Segmentation) top out around 0.82-0.90.
""")
# reuse v2 cells: GPU check, pip, mount+unzip, data prep (indices 1..6 of the v2 notebook)
cells.extend(v2["cells"][1:7])

md("## 3. Shared setup: losses, augmentation, loaders, metrics")
code("""
import os, torch, torch.nn.functional as F, albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
random.seed(SEED)

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_tensor(img): return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())

aug = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                 A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, interpolation=1,
                                    mask_interpolation=0, border_mode=0, p=0.6),
                 A.RandomBrightnessContrast(0.15, 0.15, p=0.5)])

class DS(Dataset):
    def __init__(self, x, y, train): self.x, self.y, self.train = x, y, train
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        img, m = self.x[i].astype(np.float32), self.y[i]
        if self.train:
            r = aug(image=img, mask=m); img, m = np.clip(r['image'], 0, 1), r['mask']
        return to_tensor(img), torch.from_numpy(m.astype(np.float32))[None]

class DiceBCE(torch.nn.Module):
    def forward(self, lg, t):
        p = torch.sigmoid(lg); inter = (p * t).sum((1, 2, 3))
        dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
        return F.binary_cross_entropy_with_logits(lg, t) + (1 - dice).mean()

class FocalTverskyLoss(torch.nn.Module):
    \"\"\"Same as v5: beta > alpha punishes missed tumor (false negative) more than over-drawing one,
    gamma > 1 focuses training on the harder (small/tricky) examples.\"\"\"
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, bce_weight=0.3):
        super().__init__(); self.alpha, self.beta, self.gamma, self.bce_weight = alpha, beta, gamma, bce_weight
    def forward(self, lg, t):
        p = torch.sigmoid(lg)
        tp = (p * t).sum((1, 2, 3)); fp = ((1 - t) * p).sum((1, 2, 3)); fn = (t * (1 - p)).sum((1, 2, 3))
        tversky = (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)
        ftl = ((1 - tversky) ** self.gamma).mean()
        return ftl + self.bce_weight * F.binary_cross_entropy_with_logits(lg, t)

def batch_dice(pr, t):
    inter = (pr & t).flatten(1).sum(1).float(); tot = (pr.flatten(1).sum(1) + t.flatten(1).sum(1)).float()
    return torch.where(tot == 0, torch.ones_like(inter), 2 * inter / tot.clamp(min=1))

def dice_iou(p, t):
    p, t = p.astype(bool), t.astype(bool); tot = p.sum() + t.sum()
    if tot == 0: return 1.0, 1.0
    i = (p & t).sum(); return 2 * i / tot, i / (p | t).sum()

def postprocess(prob, thr, min_area):
    m = prob > thr
    m[m.reshape(len(m), -1).sum(1) < min_area] = False
    return m

def evaluate(P, Y):
    sc = np.array([dice_iou(p, y) for p, y in zip(P, Y)]); has = Y.reshape(len(Y), -1).any(1)
    return sc, has

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ARCH = {'Unet': smp.Unet, 'UnetPlusPlus': smp.UnetPlusPlus}
LOSS = {'dicebce': DiceBCE, 'focaltversky': FocalTverskyLoss}
def make(arch, enc, pretrained=True):
    return ARCH[arch](enc, encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=1)

Xtr, Ytr = D['train']
has_tumor = Ytr.reshape(len(Ytr), -1).any(1)
w = np.where(has_tumor, 2.0, 1.0)     # tumor slices ~2x as often
vl = DataLoader(DS(*D['val'], False), 32, num_workers=2)
""")

md("## 4. Train the 2 new models (each saved to Drive right after it finishes; skipped on re-run if already saved)")
code("""
MODELS = [
    dict(name='m6_effnet',    arch='Unet',         enc='efficientnet-b4', loss='dicebce',      seed=7,   file='best_model_m6_effnet.pth'),
    dict(name='m6_unetpp_ft', arch='UnetPlusPlus', enc='resnet50',        loss='focaltversky',  seed=123, file='best_model_m6_unetpp_ft.pth'),
]
EPOCHS, PATIENCE, BS, LR = 45, 10, 16, 2e-4

def train_model(cfg):
    path = f"{DRIVE}/{cfg['file']}"
    if os.path.exists(path):
        print('found', cfg['file'], '-> skipping training'); return
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    tl = DataLoader(DS(Xtr, Ytr, True), BS, sampler=WeightedRandomSampler(w, len(Xtr), replacement=True), num_workers=2)
    model = make(cfg['arch'], cfg['enc']).to(dev); loss_fn = LOSS[cfg['loss']]()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(dev == 'cuda'))
    best, bad, hist = -1, 0, []
    print('=== training', cfg['name'], cfg['arch'], cfg['enc'], cfg['loss'])
    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                lg = model(x)
            l = loss_fn(lg.float(), y); scaler.scale(l).backward(); scaler.step(opt); scaler.update(); tot += l.item() * len(x)
        sched.step(); tr = tot / len(tl.dataset)
        model.eval(); dices, tum = [], []
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                    lg = model(x)
                dices.append(batch_dice(torch.sigmoid(lg.float()) > 0.5, y > 0.5)); tum.append(y.flatten(1).sum(1) > 0)
        dices, tum = torch.cat(dices), torch.cat(tum)
        d_all, d_tum = dices.mean().item(), dices[tum].mean().item(); score = 0.5 * (d_all + d_tum); star = ''
        hist.append(dict(epoch=ep + 1, train=tr, val_dice_all=d_all, val_dice_tumor=d_tum))
        if score > best:
            best, bad, star = score, 0, ' *saved'; torch.save(model.state_dict(), path)
        else: bad += 1
        print(f"epoch {ep+1}/{EPOCHS} train={tr:.4f} valDice(all)={d_all:.3f} valDice(tumor)={d_tum:.3f} ({time.time()-t0:.0f}s){star}")
        if bad >= PATIENCE: print('early stopping'); break
    json.dump(hist, open(f"{DRIVE}/history_{cfg['name']}.json", 'w'))
    del model, opt; torch.cuda.empty_cache()

for cfg in MODELS: train_model(cfg)
""")

md("""## 5. Mega-ensemble: the 2 new models + the v4 5-fold models (loaded, not retrained)
Threshold/min_area chosen on the 16-patient VALIDATION set only - test is not touched until the next cell.""")
code("""
@torch.no_grad()
def predict_probs(model, X):
    out = []
    for i in range(0, len(X), 32):
        xb = torch.stack([to_tensor(x.astype(np.float32)) for x in X[i:i+32]]).to(dev)
        p = (torch.sigmoid(model(xb)) + torch.sigmoid(model(xb.flip(3))).flip(3) + torch.sigmoid(model(xb.flip(2))).flip(2)) / 3
        out.append(p[:, 0].float().cpu())
    return torch.cat(out).numpy()

Xv, Yv = D['val']; Xt, Yt = D['test']
PV, PT = {}, {}
for cfg in MODELS:
    m = make(cfg['arch'], cfg['enc'], pretrained=False).to(dev)
    m.load_state_dict(torch.load(f"{DRIVE}/{cfg['file']}")); m.eval()
    PV[cfg['name']], PT[cfg['name']] = predict_probs(m, Xv), predict_probs(m, Xt)
    del m; torch.cuda.empty_cache()

# v4's 5 cross-validation fold models (trained on 93 patients, none of which is the 17-patient test set)
for i in range(5):
    m = smp.Unet('resnet50', encoder_weights=None, in_channels=3, classes=1).to(dev)
    m.load_state_dict(torch.load(f'{DRIVE}/fold{i}_model.pth')); m.eval()
    name = f'v4_fold{i}'
    PV[name], PT[name] = predict_probs(m, Xv), predict_probs(m, Xt)
    del m; torch.cuda.empty_cache()

all_names = list(PV.keys())
PV['ensemble_all'] = np.mean([PV[n] for n in all_names], 0); PT['ensemble_all'] = np.mean([PT[n] for n in all_names], 0)
# also try just the 2 new models + v4 mean (in case adding all 5 folds separately over-weights v4's family)
v4_mean_v, v4_mean_t = np.mean([PV[f'v4_fold{i}'] for i in range(5)], 0), np.mean([PT[f'v4_fold{i}'] for i in range(5)], 0)
PV['ensemble_balanced'] = np.mean([PV['m6_effnet'], PV['m6_unetpp_ft'], v4_mean_v], 0)
PT['ensemble_balanced'] = np.mean([PT['m6_effnet'], PT['m6_unetpp_ft'], v4_mean_t], 0)

def tune(P, Y):
    best_cfg, best_s = None, -1
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for min_area in (0, 30, 60, 100, 200):
            sc, has = evaluate(postprocess(P, thr, min_area), Y)
            s = 0.5 * (sc[:, 0].mean() + sc[has, 0].mean())
            if s > best_s: best_s, best_cfg = s, (thr, min_area)
    return best_cfg, best_s

print('VALIDATION scores (mean of all-slice Dice and tumor-slice Dice) - model/ensemble is CHOSEN here, not on test:')
candidates = ['m6_effnet', 'm6_unetpp_ft', 'ensemble_all', 'ensemble_balanced']
val_res = {}
for k in candidates:
    cfg_k, s = tune(PV[k], Yv); val_res[k] = (cfg_k, s); print(f'  {k:20s} thr/min_area={cfg_k}  score={s:.4f}')
chosen = max(val_res, key=lambda k: val_res[k][1])
print('CHOSEN on validation:', chosen, '  <- compare to v4 alone (out-of-fold 0.8493) and v3 (0.8693)')
""")

md("## 6. Evaluate ONCE on the unseen test patients (chosen model/ensemble only)")
code("""
thr, min_area = val_res[chosen][0]
M = postprocess(PT[chosen], thr, min_area)
sc, has = evaluate(M, Yt); pix_acc = (M == Yt.astype(bool)).mean()
rep = (f"Test slices: {len(Yt)} ({has.sum()} with tumor) | chosen: {chosen} | threshold={thr}, min_area={min_area}, TTA on\\n"
       f"All slices      Dice={sc[:,0].mean():.4f}  IoU={sc[:,1].mean():.4f}   (empty-vs-empty scored 1.0)\\n"
       f"Tumor slices    Dice={sc[has,0].mean():.4f}  IoU={sc[has,1].mean():.4f}   <- compare to v4's 0.7867 and v3's 0.7922\\n"
       f"Pixel accuracy  {pix_acc*100:.3f}%   (misleading on its own)\\n")
print(rep); open(f'{DRIVE}/evaluation_report_v6.txt', 'w').write(rep)
json.dump({'chosen': chosen, 'threshold': thr, 'min_area': min_area,
           'models': [c for c in MODELS] + [dict(name=f'v4_fold{i}', file=f'fold{i}_model.pth') for i in range(5)]},
          open(f'{DRIVE}/postprocess_v6.json', 'w'), indent=1)

random.seed(SEED)
idx = random.sample(list(np.where(has)[0]), 14) + random.sample(list(np.where(~has)[0]), 6)
fig, ax = plt.subplots(20, 3, figsize=(7, 46))
for r, i in enumerate(idx):
    for c, (im, t, cm) in enumerate([(Xt[i].astype(np.float32), 'Scan', None), (Yt[i], 'True mask', 'gray'),
                                     (M[i], f'Predicted (Dice {sc[i,0]:.2f})', 'gray')]):
        ax[r, c].imshow(im, cmap=cm); ax[r, c].set_title(t, fontsize=7); ax[r, c].axis('off')
plt.tight_layout(); plt.savefig(f'{DRIVE}/sample_predictions_v6.png', dpi=80); plt.show()
plt.figure(figsize=(6, 3)); plt.hist(sc[has, 0], bins=20); plt.xlabel('Dice on tumor slices'); plt.ylabel('count')
plt.savefig(f'{DRIVE}/dice_hist_v6.png'); plt.show()
print('Saved *_v6 files to', DRIVE)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("colab_train_v6.ipynb", "w"), indent=1)
print("wrote colab_train_v6.ipynb")
