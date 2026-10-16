"""Shared helpers for the TumorVision LGG segmentation pipeline."""
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42


def collect_pairs():
    """Walk patient folders -> list of (patient_id, scan_path, mask_path)."""
    pairs = []
    for pdir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for scan in sorted(pdir.glob("*.tif")):
            if scan.stem.endswith("_mask"):
                continue
            mask = scan.with_name(scan.stem + "_mask.tif")
            if mask.exists():
                pairs.append((pdir.name, scan, mask))
    return pairs


def load_pair(scan_path, mask_path):
    img = tifffile.imread(scan_path).astype(np.float32) / 255.0   # (256,256,3) in 0-1
    mask = (tifffile.imread(mask_path) > 0).astype(np.float32)    # (256,256) binary
    return img, mask


def dice_iou(pred, true, eps=1e-7):
    """Per-image Dice / IoU on binary arrays. Both empty -> 1.0 (correct 'no tumor')."""
    p, t = pred.astype(bool), true.astype(bool)
    inter = (p & t).sum()
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0, 1.0
    return 2 * inter / (total + eps), inter / ((p | t).sum() + eps)


class DiceBCELoss(torch.nn.Module):
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return bce + (1 - dice).mean()


def build_model(pretrained=True, encoder="resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)   # raw logits; sigmoid applied at inference


# ImageNet stats: the pretrained encoder expects inputs normalised this way (after /255)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img):
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())
