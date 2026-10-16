"""
TumorVision backend API.

Serves the trained v4 (5-fold cross-validation ensemble) model behind an HTTP API,
so it can be called by a hospital system, a demo page, or a script - not just from
inside a notebook. Also serves a small built-in HTML page at "/" for a quick manual demo.

Run:
    D:\\oncology\\.venv\\Scripts\\python -m uvicorn backend.app:app --reload --port 8000
    (run this from D:\\oncology, so "backend" is importable as a package)

Then open http://127.0.0.1:8000 in a browser for the demo page,
or http://127.0.0.1:8000/docs for the interactive API docs.
"""
import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from common import OUT, build_model, to_tensor  # noqa: E402

app = FastAPI(title="TumorVision API", description="Brain MRI tumor segmentation (LGG dataset, 5-fold U-Net ensemble)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS = []       # loaded once at startup
CFG = {}          # postprocess_v4.json: encoder, threshold, min_area, fold_files
DATASET_DIR = ROOT / "data"   # local LGG dataset (110 patients, full per-patient slice stacks) - used for the 3D volume view
_PATIENT_IDS = None            # lazily populated by get_patient_ids()


@app.on_event("startup")
def load_models():
    global CFG
    cfg_path = OUT / "postprocess_v4.json"
    if not cfg_path.exists():
        print(f"WARNING: {cfg_path} not found - /predict will fail until model files are in outputs/")
        return
    CFG = json.load(open(cfg_path))
    for fname in CFG["fold_files"]:
        weights = OUT / fname
        if not weights.exists():
            print(f"WARNING: missing {weights}")
            continue
        m = build_model(pretrained=False, encoder=CFG["encoder"])
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        m.eval()
        MODELS.append(m)
    print(f"Loaded {len(MODELS)}/{len(CFG.get('fold_files', []))} fold models "
          f"(threshold={CFG.get('threshold')}, min_area={CFG.get('min_area')})")


def read_scan(raw: bytes, filename: str) -> np.ndarray:
    """Bytes -> (256,256,3) float32 array in [0,1]. Accepts .tif (as used by the dataset) or common images."""
    if filename.lower().endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(io.BytesIO(raw))
    else:
        arr = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

    if arr.ndim == 2:                       # grayscale -> 3 channels
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[:2] != (256, 256):
        arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((256, 256)))
    return arr.astype(np.float32) / 255.0


@torch.no_grad()
def predict_mask(img: np.ndarray) -> np.ndarray:
    """Runs the 5-model ensemble with flip test-time augmentation. Returns a (256,256) bool mask."""
    x = to_tensor(img)[None]
    probs = []
    for model in MODELS:
        p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(3))).flip(3)
             + torch.sigmoid(model(x.flip(2))).flip(2)) / 3
        probs.append(p[0, 0].numpy())
    prob = np.mean(probs, axis=0)
    mask = prob > CFG["threshold"]
    if mask.sum() < CFG["min_area"]:
        mask[:] = False
    return mask


def to_png_b64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scan with the predicted tumor region tinted red, for a doctor-readable view."""
    base = (img * 255).astype(np.uint8).copy()
    red = np.zeros_like(base); red[..., 0] = 255
    alpha = 0.45
    base[mask] = (base[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)
    return base


# ---------------------------------------------------------------------------
# 3D volume view - reconstructs a whole patient's slice stack into one volume
# using the SAME 2D 5-fold ensemble as /predict (run once per slice), so this
# reuses the already-verified model rather than needing the (not yet trained)
# 3D U-Net. No physical (mm/cm) units anywhere below: this dataset's .tif files
# carry no pixel-spacing/slice-thickness metadata (checked against data/README.md
# and data/data.csv - neither has it), so all measurements are honestly reported
# in pixels (in-plane) and slice-count (depth), never invented mm/cm figures.
# ---------------------------------------------------------------------------

def get_patient_ids() -> list:
    """Patient IDs available in the local dataset (data/<patient_id>/, scan+mask pairs present)."""
    global _PATIENT_IDS
    if _PATIENT_IDS is None:
        _PATIENT_IDS = sorted(
            p.name for p in DATASET_DIR.iterdir()
            if p.is_dir() and next(p.glob("*_mask.tif"), None) is not None
        )
    return _PATIENT_IDS


def load_patient_stack(pid: str):
    """One patient's full slice stack, in correct NUMERIC z-order (filenames sort lexicographically,
    which is the wrong order for stacking - '_11' would sort before '_2').
    Returns scan (D,256,256,3) float32 in [0,1] and the original slice indices (for display)."""
    pdir = DATASET_DIR / pid
    ordered = []
    for scan in pdir.glob("*.tif"):
        if scan.stem.endswith("_mask"):
            continue
        mask = scan.with_name(scan.stem + "_mask.tif")
        if not mask.exists():
            continue
        idx = int(scan.stem[len(pid) + 1:])   # "<patient_id>_<slice_index>.tif" -> slice_index
        ordered.append((idx, scan))
    ordered.sort(key=lambda t: t[0])
    import tifffile
    arr = np.stack([tifffile.imread(s).astype(np.float32) / 255.0 for _, s in ordered])
    return arr, [idx for idx, _ in ordered]


@torch.no_grad()
def predict_volume_batch(scan_stack: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """Runs the 5-model ensemble + flip TTA across every slice of a patient stack (batched, not
    one-slice-at-a-time, so a ~35-88 slice patient takes seconds rather than tens of API calls).
    scan_stack: (D,256,256,3) float32 in [0,1]. Returns (D,256,256) bool mask - same threshold and
    per-slice min-area rule as /predict, so results for any single slice match /predict exactly."""
    D = scan_stack.shape[0]
    prob_sum = np.zeros((D, 256, 256), dtype=np.float32)
    for start in range(0, D, batch_size):
        chunk = scan_stack[start:start + batch_size]
        x = torch.stack([to_tensor(im) for im in chunk])
        for model in MODELS:
            p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(3))).flip(3)
                 + torch.sigmoid(model(x.flip(2))).flip(2)) / 3
            prob_sum[start:start + batch_size] += p[:, 0].numpy()
    prob = prob_sum / len(MODELS)
    mask = prob > CFG["threshold"]
    flat_area = mask.reshape(D, -1).sum(1)
    mask[flat_area < CFG["min_area"]] = False
    return mask


def volume_stats(mask: np.ndarray) -> dict:
    """Pixel/slice-based size stats - deliberately NOT in mm/cm (see module note above)."""
    D, H, W = mask.shape
    tumor_voxels = int(mask.sum())
    stats = {
        "total_slices": D,
        "slices_with_tumor": int(mask.reshape(D, -1).any(1).sum()),
        "tumor_voxel_count": tumor_voxels,
        "tumor_volume_percent_of_scanned_stack": round(100 * tumor_voxels / mask.size, 4) if mask.size else 0.0,
        "extent": None,
    }
    if tumor_voxels > 0:
        zs, ys, xs = np.nonzero(mask)
        stats["extent"] = {
            "depth_slices": int(zs.max() - zs.min() + 1),
            "height_px": int(ys.max() - ys.min() + 1),
            "width_px": int(xs.max() - xs.min() + 1),
        }
    return stats


def tumor_pointcloud(mask: np.ndarray, max_points: int = 15000) -> list:
    """Downsampled (z,y,x) voxel coordinates of the tumor, for the 3D scatter view."""
    zs, ys, xs = np.nonzero(mask)
    n = len(zs)
    if n == 0:
        return []
    if n > max_points:
        stride = int(np.ceil(n / max_points))
        zs, ys, xs = zs[::stride], ys[::stride], xs[::stride]
    return [[int(z), int(y), int(x)] for z, y, x in zip(zs, ys, xs)]


def representative_slices(scan_stack: np.ndarray, mask: np.ndarray, slice_indices: list) -> dict:
    """A handful of labelled overlay images (largest tumor slice, first/last tumor slice, or a
    middle slice if no tumor was found) - 2D context to go with the 3D view / report."""
    D = mask.shape[0]
    areas = mask.reshape(D, -1).sum(1)
    tumor_idx = np.nonzero(areas)[0]
    picks = {}
    if len(tumor_idx):
        picks["Largest tumor slice"] = int(tumor_idx[np.argmax(areas[tumor_idx])])
        picks["First tumor-containing slice"] = int(tumor_idx[0])
        picks["Last tumor-containing slice"] = int(tumor_idx[-1])
    else:
        picks["Middle slice (no tumor detected)"] = D // 2
    out = {}
    for label, i in picks.items():
        overlay = make_overlay(scan_stack[i], mask[i])
        out[label] = {"slice_index": slice_indices[i], "tumor_pixels": int(areas[i]),
                      "overlay_png": to_png_b64(overlay)}
    return out


_ANALYSIS_CACHE = {}   # pid -> analysis dict; a patient takes ~1 min on CPU, so each is computed once per process
_ANALYSIS_CACHE_MAX = 12


def analyze_patient(pid: str) -> dict:
    """Runs the ensemble over a patient's whole stack once and keeps everything the 3D view and the
    report need (so the /volume page and the standalone report never repeat the inference)."""
    if pid not in _ANALYSIS_CACHE:
        scan_stack, slice_indices = load_patient_stack(pid)
        mask = predict_volume_batch(scan_stack)
        points = tumor_pointcloud(mask)
        if len(_ANALYSIS_CACHE) >= _ANALYSIS_CACHE_MAX:
            _ANALYSIS_CACHE.pop(next(iter(_ANALYSIS_CACHE)))
        _ANALYSIS_CACHE[pid] = {
            "shape": {"depth_slices": mask.shape[0], "height_px": mask.shape[1], "width_px": mask.shape[2]},
            "stats": volume_stats(mask),
            "points": points,
            "tumor_voxels": int(mask.sum()),
            "reps": representative_slices(scan_stack, mask, slice_indices),
        }
    return _ANALYSIS_CACHE[pid]


@app.get("/health")
def health():
    return {
        "status": "ok" if MODELS else "model not loaded",
        "models_loaded": len(MODELS),
        "encoder": CFG.get("encoder"),
        "threshold": CFG.get("threshold"),
        "min_area": CFG.get("min_area"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODELS:
        raise HTTPException(503, "Model not loaded - check that outputs/fold*_model.pth and postprocess_v4.json exist")
    raw = await file.read()
    try:
        img = read_scan(raw, file.filename or "scan.tif")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    mask = predict_mask(img)
    overlay = make_overlay(img, mask)
    tumor_pixels = int(mask.sum())

    return JSONResponse({
        "filename": file.filename,
        "tumor_detected": tumor_pixels > 0,
        "tumor_pixel_count": tumor_pixels,
        "tumor_area_percent": round(100 * tumor_pixels / mask.size, 3),
        "original_png": to_png_b64((img * 255).astype(np.uint8)),   # re-encoded so the browser can always render it (source may be .tif)
        "mask_png": to_png_b64((mask * 255).astype(np.uint8)),
        "overlay_png": to_png_b64(overlay),
        "model_info": {"encoder": CFG.get("encoder"), "n_folds": len(MODELS), "test_dice_tumor": 0.7867},
    })


CALIBRATION_NOTE = (
    "Uncalibrated: measurements are in pixels (in-plane) and slice-count (depth), NOT millimetres or "
    "centimetres. This dataset's .tif files carry no pixel-spacing / slice-thickness metadata, so no "
    "physical unit conversion is applied anywhere in this report - inventing one would risk misleading "
    "whoever reads it. If real spacing is known from the original DICOM/PACS study, convert manually."
)


@app.get("/dataset/patients")
def dataset_patients():
    """Patient IDs available in the local LGG dataset, for the 3D volume view / report (data/<patient_id>/)."""
    return {"patients": get_patient_ids()}


@app.get("/dataset/patients/{pid}/volume")
def dataset_patient_volume(pid: str):
    if pid not in get_patient_ids():
        raise HTTPException(404, f"Unknown patient id '{pid}'")
    if not MODELS:
        raise HTTPException(503, "Model not loaded - check that outputs/fold*_model.pth and postprocess_v4.json exist")
    a = analyze_patient(pid)
    return JSONResponse({
        "patient_id": pid,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "shape": a["shape"],
        "stats": a["stats"],
        "calibration": CALIBRATION_NOTE,
        "pointcloud": a["points"],
        "pointcloud_downsampled_from": a["tumor_voxels"] if len(a["points"]) < a["tumor_voxels"] else None,
        "representative_slices": a["reps"],
        "model_info": {"encoder": CFG.get("encoder"), "n_folds": len(MODELS), "test_dice_tumor": 0.7867},
    })


REPORT_ROW = """
      <div class="rep-card">
        <img src="{png}" alt="{label}">
        <div class="rep-cap"><b>{label}</b><br>Slice #{idx} &middot; {px:,} tumor px</div>
      </div>"""

REPORT_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TumorVision report - {pid}</title>
<style>
  :root{{ --ink:#0f1629; --muted:#5b6478; --line:#dfe3ea; --danger:#b3261e; --danger-bg:#fdecea; --brand:#3730a3; }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
       font-size:14px;line-height:1.5;padding:2rem clamp(1rem,4vw,3rem) 3rem;max-width:900px;margin:0 auto}}
  h1{{font-size:1.4rem;margin:.2rem 0}}
  .sub{{color:var(--muted);font-size:.85rem;margin-bottom:1.2rem}}
  .meta{{display:flex;flex-wrap:wrap;gap:.5rem 1.6rem;font-size:.85rem;color:var(--muted);margin-bottom:1.2rem}}
  .meta b{{color:var(--ink)}}
  .disclaimer{{background:var(--danger-bg);border:1px solid #f3c6c2;color:var(--danger);border-radius:10px;
              padding:.9rem 1.1rem;font-size:.85rem;font-weight:600;margin-bottom:1.4rem}}
  h2{{font-size:1rem;border-bottom:2px solid var(--line);padding-bottom:.35rem;margin:1.6rem 0 .8rem}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  td{{padding:.4rem .2rem;border-bottom:1px solid var(--line)}}
  td:first-child{{color:var(--muted)}}
  td:last-child{{text-align:right;font-weight:700}}
  .calib-note{{font-size:.78rem;color:var(--muted);background:#f6f7fb;border:1px solid var(--line);border-radius:8px;
              padding:.6rem .8rem;margin-top:.6rem}}
  .rep-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.8rem;margin-top:.4rem}}
  .rep-card{{border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  .rep-card img{{width:100%;aspect-ratio:1/1;object-fit:contain;background:#000;display:block}}
  .rep-cap{{padding:.5rem .6rem;font-size:.76rem;color:var(--muted)}}
  .verify{{border:1.5px solid var(--ink);border-radius:10px;padding:1.1rem 1.3rem;margin-top:.6rem}}
  .verify .row{{display:flex;gap:1.4rem;margin:.7rem 0;flex-wrap:wrap}}
  .field{{flex:1;min-width:200px}}
  .field label{{display:block;font-size:.78rem;color:var(--muted);margin-bottom:.3rem}}
  .line{{border:none;border-bottom:1px solid var(--ink);height:1.6rem}}
  .checks{{display:flex;gap:1.4rem;font-size:.88rem;margin:.4rem 0}}
  .checks span{{display:flex;align-items:center;gap:.4rem}}
  .box{{width:14px;height:14px;border:1.5px solid var(--ink);border-radius:3px;display:inline-block}}
  .notes-lines{{border-bottom:1px solid var(--line);height:1.7rem}}
  .btn-print{{background:var(--brand);color:#fff;border:none;padding:.6rem 1.1rem;border-radius:8px;font-weight:700;
             cursor:pointer;font-size:.85rem}}
  footer{{margin-top:2rem;font-size:.76rem;color:var(--muted);text-align:center}}
  @media print{{ .no-print{{display:none}} body{{padding:0 1cm}} }}
</style></head>
<body>
  <div class="no-print" style="text-align:right;margin-bottom:.6rem">
    <button class="btn-print" onclick="window.print()">Print / Save as PDF</button>
  </div>

  <h1>TumorVision - AI-Assisted Segmentation Report</h1>
  <div class="sub">Volumetric summary reconstructed from a 2D 5-fold U-Net ensemble run across every slice of the patient's scan.</div>
  <div class="meta">
    <span>Patient / case ID: <b>{pid}</b></span>
    <span>Generated: <b>{generated}</b></span>
    <span>Model: <b>{encoder}, {n_folds}-fold ensemble</b></span>
    <span>Test-set Dice (tumor slices): <b>0.7867</b></span>
  </div>

  <div class="disclaimer">
    This report is produced by a research prototype AI model (not an FDA/CE-cleared diagnostic device) and
    is NOT a diagnosis. It must be reviewed and verified by a qualified radiologist before being relied upon
    or forwarded to a treating physician.
  </div>

  <h2>Volume summary</h2>
  <table>
    <tr><td>Total slices in scan</td><td>{total_slices}</td></tr>
    <tr><td>Slices with tumor detected</td><td>{slices_with_tumor}</td></tr>
    <tr><td>Tumor voxel count</td><td>{tumor_voxel_count:,}</td></tr>
    <tr><td>Tumor volume (% of scanned stack)</td><td>{tumor_volume_percent}%</td></tr>
    <tr><td>Extent - width (x)</td><td>{width_px}</td></tr>
    <tr><td>Extent - height (y)</td><td>{height_px}</td></tr>
    <tr><td>Extent - depth (z, slices)</td><td>{depth_slices}</td></tr>
  </table>
  <div class="calib-note">{calibration_note}</div>

  <h2>Representative slices</h2>
  <div class="rep-grid">{rep_cards}
  </div>

  <h2>Radiologist verification</h2>
  <div class="verify">
    <div class="row">
      <div class="field"><label>Reviewed by (name)</label><div class="line"></div></div>
      <div class="field"><label>Date</label><div class="line"></div></div>
    </div>
    <div class="checks">
      <span><span class="box"></span> Findings confirmed</span>
      <span><span class="box"></span> Findings amended</span>
      <span><span class="box"></span> Findings rejected</span>
    </div>
    <div class="field" style="margin-top:.6rem"><label>Notes / comments</label>
      <div class="notes-lines"></div><div class="notes-lines"></div><div class="notes-lines"></div>
    </div>
    <div class="row" style="margin-top:.8rem">
      <div class="field"><label>Signature</label><div class="line"></div></div>
    </div>
  </div>

  <footer>TumorVision &middot; brain-MRI segmentation research proof-of-concept &middot; not for clinical use without radiologist sign-off</footer>
</body></html>"""


@app.get("/dataset/patients/{pid}/report", response_class=HTMLResponse)
def dataset_patient_report(pid: str):
    if pid not in get_patient_ids():
        raise HTTPException(404, f"Unknown patient id '{pid}'")
    if not MODELS:
        raise HTTPException(503, "Model not loaded - check that outputs/fold*_model.pth and postprocess_v4.json exist")
    a = analyze_patient(pid)
    stats = a["stats"]
    extent = stats["extent"] or {"width_px": "-", "height_px": "-", "depth_slices": "-"}
    reps = a["reps"]
    rep_cards = "".join(REPORT_ROW.format(png=v["overlay_png"], label=k, idx=v["slice_index"], px=v["tumor_pixels"])
                         for k, v in reps.items())
    html = REPORT_TEMPLATE.format(
        pid=pid, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        encoder=CFG.get("encoder"), n_folds=len(MODELS),
        total_slices=stats["total_slices"], slices_with_tumor=stats["slices_with_tumor"],
        tumor_voxel_count=stats["tumor_voxel_count"],
        tumor_volume_percent=stats["tumor_volume_percent_of_scanned_stack"],
        width_px=extent["width_px"], height_px=extent["height_px"], depth_slices=extent["depth_slices"],
        calibration_note=CALIBRATION_NOTE, rep_cards=rep_cards,
    )
    return HTMLResponse(html)


FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
           "%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E"
           "%3Cstop offset='0' stop-color='%236366f1'/%3E%3Cstop offset='1' stop-color='%2306b6d4'/%3E%3C/linearGradient%3E"
           "%3C/defs%3E%3Crect width='64' height='64' rx='16' fill='url(%23g)'/%3E"
           "%3Ccircle cx='32' cy='30' r='14' fill='none' stroke='white' stroke-width='4'/%3E"
           "%3Ccircle cx='32' cy='30' r='5' fill='white'/%3E%3C/svg%3E")

DEMO_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TumorVision</title>
<link rel="icon" href="__FAVICON__">
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --surface-2:#f8fafc; --border:#e6e9f0; --text:#0f1629; --muted:#667085;
  --brand:#4f46e5; --brand-2:#06b6d4; --brand-dark:#4338ca; --brand-soft:#eef1ff;
  --ok:#0e9f6e; --ok-soft:#e5f9f0; --danger:#e11d48; --danger-soft:#fde8ee;
  --r-sm:10px; --r-md:16px; --r-lg:22px;
  --shadow-sm:0 1px 2px rgba(16,24,40,.05);
  --shadow-md:0 4px 14px -4px rgba(16,24,40,.10), 0 1px 3px rgba(16,24,40,.06);
  --shadow-lg:0 24px 48px -16px rgba(16,24,40,.18);
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0a0e1a; --surface:#121826; --surface-2:#0d1220; --border:#232b3d; --text:#eaeef7; --muted:#8b96ab;
         --brand:#818cf8; --brand-2:#22d3ee; --brand-dark:#a5b4fc; --brand-soft:#171d33;
         --ok:#34d399; --ok-soft:#0e2a22; --danger:#fb7185; --danger-soft:#301220;
         --shadow-sm:0 1px 2px rgba(0,0,0,.3);
         --shadow-md:0 4px 14px -4px rgba(0,0,0,.5), 0 1px 3px rgba(0,0,0,.4);
         --shadow-lg:0 24px 48px -16px rgba(0,0,0,.6); }
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
     -webkit-font-smoothing:antialiased;position:relative;overflow-x:hidden}
.bg-blob{position:fixed;border-radius:50%;filter:blur(80px);opacity:.35;z-index:0;pointer-events:none}
.b1{width:520px;height:520px;top:-180px;right:-140px;background:radial-gradient(circle,var(--brand),transparent 70%)}
.b2{width:420px;height:420px;bottom:-160px;left:-120px;background:radial-gradient(circle,var(--brand-2),transparent 70%);opacity:.25}
.page{position:relative;z-index:1;max-width:1040px;margin:0 auto;padding:1.4rem 1.25rem 4rem}

.nav{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0 2.2rem}
.brand{display:flex;align-items:center;gap:.6rem}
.brand-mark{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,var(--brand),var(--brand-2));
            display:flex;align-items:center;justify-content:center;flex:none;box-shadow:var(--shadow-sm)}
.brand-mark svg{width:19px;height:19px}
.brand-name{font-weight:800;font-size:1.05rem;letter-spacing:-.01em}
.nav-right{display:flex;align-items:center;gap:1.1rem}
.health{display:flex;align-items:center;gap:.45rem;font-size:.82rem;color:var(--muted);font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;background:var(--muted);position:relative}
.dot.ready{background:var(--ok)}
.dot.ready::after{content:'';position:absolute;inset:-4px;border-radius:50%;border:1.5px solid var(--ok);
                   animation:pulse 1.8s ease-out infinite}
@keyframes pulse{0%{opacity:.7;transform:scale(.6)}100%{opacity:0;transform:scale(1.9)}}
.nav-link{color:var(--muted);text-decoration:none;font-size:.85rem;font-weight:600;border:1px solid var(--border);
          padding:.35rem .75rem;border-radius:999px;transition:color .15s,border-color .15s}
.nav-link:hover{color:var(--brand);border-color:var(--brand)}

.hero{text-align:center;max-width:640px;margin:0 auto 2.2rem}
.hero h1{font-size:clamp(1.6rem,3.4vw,2.35rem);line-height:1.15;letter-spacing:-.02em;margin:0 0 .7rem;font-weight:800}
.hero h1 .accent{background:linear-gradient(135deg,var(--brand),var(--brand-2));-webkit-background-clip:text;
                  background-clip:text;color:transparent}
.hero p{color:var(--muted);font-size:1.02rem;line-height:1.55;margin:0 0 1.1rem}
.hero-badges{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap}
.badge{font-size:.76rem;padding:.32rem .7rem;border-radius:999px;background:var(--brand-soft);color:var(--brand-dark);
       border:1px solid var(--border);font-weight:700;letter-spacing:.01em}
.badge.accent{background:linear-gradient(135deg,var(--brand),var(--brand-2));color:#fff;border:none}

.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);box-shadow:var(--shadow-md);
      padding:1.6rem}
.drop{border:2px dashed var(--border);border-radius:var(--r-md);padding:2.6rem 1rem;text-align:center;cursor:pointer;
      transition:border-color .18s,background .18s,transform .18s}
.drop:hover{border-color:var(--brand);background:var(--brand-soft)}
.drop.drag{border-color:var(--brand);background:var(--brand-soft);transform:scale(1.01)}
.drop-icon{width:52px;height:52px;margin:0 auto .9rem;border-radius:50%;background:var(--brand-soft);
           display:flex;align-items:center;justify-content:center;color:var(--brand)}
.drop-icon svg{width:24px;height:24px}
.drop-main{font-weight:700;font-size:1.02rem}
.drop-sub{color:var(--muted);font-size:.85rem;margin-top:.3rem}
input[type=file]{display:none}

.file-row{display:flex;align-items:center;gap:.9rem;padding:.9rem 1rem;background:var(--surface-2);
          border:1px solid var(--border);border-radius:var(--r-md)}
.thumb{width:48px;height:48px;border-radius:10px;object-fit:cover;background:var(--brand-soft);flex:none;
       display:flex;align-items:center;justify-content:center;color:var(--brand)}
.file-meta{flex:1;min-width:0}
.file-name{font-weight:700;font-size:.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.file-size{color:var(--muted);font-size:.78rem;margin-top:.1rem}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text);padding:.45rem .9rem;
           border-radius:999px;font-size:.82rem;font-weight:700;cursor:pointer;transition:border-color .15s}
.btn-ghost:hover{border-color:var(--brand);color:var(--brand)}

.actions{display:flex;align-items:center;gap:1rem;margin-top:1.2rem}
button#go{background:linear-gradient(135deg,var(--brand),var(--brand-2));color:#fff;border:none;
       padding:.75rem 1.6rem;border-radius:12px;font-size:.95rem;font-weight:700;cursor:pointer;
       display:flex;align-items:center;gap:.55rem;transition:opacity .15s,transform .15s;box-shadow:var(--shadow-sm)}
button#go:hover:not(:disabled){transform:translateY(-1px);box-shadow:var(--shadow-md)}
button#go:disabled{opacity:.4;cursor:not-allowed;transform:none}
.spinner{width:16px;height:16px;border:2.5px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;
         animation:spin .7s linear infinite;display:none}
.spinner.on{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
#status{color:var(--muted);font-size:.87rem;font-weight:600}
#status.err{color:var(--danger)}

.results{opacity:0;transform:translateY(14px);transition:opacity .45s ease,transform .45s ease;pointer-events:none;
         margin-top:2rem;display:none}
.results.show{opacity:1;transform:none;pointer-events:auto}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem;margin-bottom:1.2rem}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);padding:1.1rem;
      box-shadow:var(--shadow-sm);display:flex;align-items:center;gap:.8rem}
.stat-icon{width:38px;height:38px;border-radius:10px;background:var(--brand-soft);color:var(--brand);flex:none;
           display:flex;align-items:center;justify-content:center}
.stat-icon svg{width:19px;height:19px}
.stat .label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:700}
.stat .value{font-size:1.25rem;font-weight:800;margin-top:.15rem;line-height:1.2}
.pill{display:inline-flex;align-items:center;gap:.4rem;padding:.2rem .6rem;border-radius:999px;font-weight:800;font-size:.92rem}
.pill.pos{background:var(--danger-soft);color:var(--danger)}
.pill.neg{background:var(--ok-soft);color:var(--ok)}
.pill .dot2{width:7px;height:7px;border-radius:50%;background:currentColor}

.compare-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);
              box-shadow:var(--shadow-md);padding:1.3rem;margin-bottom:1rem}
.compare-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:.9rem}
.compare-head h3{margin:0;font-size:1rem;font-weight:800}
.compare-hint{color:var(--muted);font-size:.8rem;font-weight:600}
.compare{position:relative;width:100%;aspect-ratio:1/1;max-width:480px;margin:0 auto;border-radius:var(--r-md);
         overflow:hidden;background:#000;--pct:50%;box-shadow:var(--shadow-lg);touch-action:none}
.cmp-img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;user-select:none;
         -webkit-user-drag:none}
.cmp-overlay-wrap{position:absolute;inset:0;clip-path:inset(0 calc(100% - var(--pct)) 0 0)}
.cmp-tag{position:absolute;top:.6rem;font-size:.68rem;font-weight:800;color:#fff;background:rgba(0,0,0,.5);
         padding:.2rem .55rem;border-radius:999px;letter-spacing:.03em;text-transform:uppercase;pointer-events:none}
.cmp-tag.left{left:.6rem}
.cmp-tag.right{right:.6rem}
.cmp-handle{position:absolute;top:0;bottom:0;left:var(--pct);pointer-events:none;transform:translateX(-1px)}
.cmp-handle::before{content:'';position:absolute;top:0;bottom:0;left:0;width:2px;background:#fff;
                     box-shadow:0 0 8px rgba(0,0,0,.4)}
.cmp-handle-grip{position:absolute;top:50%;left:0;transform:translate(-50%,-50%);width:34px;height:34px;
                  border-radius:50%;background:#fff;color:#111;display:flex;align-items:center;justify-content:center;
                  box-shadow:0 3px 10px rgba(0,0,0,.4)}
.cmp-handle-grip svg{width:16px;height:16px}
.cmp-range{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}

.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);box-shadow:var(--shadow-sm);
       overflow:hidden}
.panel .imgbox{background:#000;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}
.panel img{width:100%;height:100%;object-fit:contain}
.panel .cap{padding:.65rem .9rem;font-size:.85rem;font-weight:700;color:var(--muted);
            display:flex;align-items:center;justify-content:space-between}
.dl{color:var(--brand);text-decoration:none;font-size:.78rem;font-weight:800;display:flex;align-items:center;gap:.25rem}
.dl:hover{color:var(--brand-dark)}
.dl svg{width:13px;height:13px}

footer{margin-top:2.6rem;color:var(--muted);font-size:.8rem;text-align:center;line-height:1.6}
footer .disclaimer{display:flex;align-items:center;gap:.55rem;justify-content:center;background:var(--surface);
                    border:1px solid var(--border);border-radius:999px;padding:.6rem 1.1rem;margin:0 auto .9rem;
                    color:var(--text);max-width:fit-content;box-shadow:var(--shadow-sm)}
footer .disclaimer svg{width:15px;height:15px;color:var(--danger);flex:none}
@media (max-width:600px){ .nav-link{display:none} .compare-head{flex-direction:column;align-items:flex-start;gap:.2rem} }
</style></head>
<body>
<div class="bg-blob b1"></div>
<div class="bg-blob b2"></div>
<div class="page">

  <nav class="nav">
    <div class="brand">
      <div class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.4" fill="#fff" stroke="none"/></svg></div>
      <span class="brand-name">TumorVision</span>
    </div>
    <div class="nav-right">
      <span class="health"><span class="dot" id="health-dot"></span><span id="health-text">Connecting…</span></span>
      <a class="nav-link" href="/volume">3D volume view →</a>
      <a class="nav-link" href="/docs" target="_blank" rel="noopener">API docs ↗</a>
    </div>
  </nav>

  <section class="hero">
    <h1>See the <span class="accent">tumor outline</span> the moment you upload a scan</h1>
    <p>Upload a brain MRI slice and TumorVision segments the tumor region using a 5-fold cross-validated U-Net ensemble - trained end to end on the public LGG dataset.</p>
    <div class="hero-badges">
      <span class="badge" id="badge-model">Model loading…</span>
      <span class="badge">LGG MRI dataset · Kaggle</span>
      <span class="badge accent">Test Dice (tumor slices) 0.79</span>
    </div>
  </section>

  <section class="card">
    <div class="drop" id="drop" tabindex="0">
      <input type="file" id="f" accept=".tif,.tiff,.png,.jpg,.jpeg">
      <div class="drop-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 16v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1"/><path d="M7 9l5-5 5 5"/><path d="M12 4v13"/>
        </svg>
      </div>
      <div class="drop-main">Drop an MRI scan here, or click to browse</div>
      <div class="drop-sub">.tif slice from the LGG dataset (also accepts PNG / JPG)</div>
    </div>

    <div class="file-row" id="file-row" style="display:none">
      <div class="thumb" id="thumb-wrap">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" id="thumb-icon">
          <rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/>
        </svg>
        <img id="local-preview" style="display:none;width:100%;height:100%;object-fit:cover;border-radius:10px">
      </div>
      <div class="file-meta">
        <div class="file-name" id="fname"></div>
        <div class="file-size" id="fsize"></div>
      </div>
      <button class="btn-ghost" onclick="clearFile()">Change</button>
    </div>

    <div class="actions">
      <button id="go" onclick="run()" disabled>
        <span class="spinner" id="spin"></span><span id="go-label">Analyze scan</span>
      </button>
      <span id="status"></span>
    </div>
  </section>

  <section class="results" id="results">
    <div class="stats">
      <div class="stat">
        <div class="stat-icon" id="icon-detected"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><circle cx="12" cy="16" r=".5" fill="currentColor"/></svg></div>
        <div><div class="label">Result</div><div class="value" id="stat-detected">—</div></div>
      </div>
      <div class="stat">
        <div class="stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></div>
        <div><div class="label">Tumor pixels</div><div class="value" id="stat-pixels">—</div></div>
      </div>
      <div class="stat">
        <div class="stat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9 15l6-6M9 9h.01M15 15h.01"/></svg></div>
        <div><div class="label">Area of scan</div><div class="value" id="stat-area">—</div></div>
      </div>
    </div>

    <div class="compare-card">
      <div class="compare-head">
        <h3>Original vs. AI overlay</h3>
        <span class="compare-hint">Drag the handle to compare ⟷</span>
      </div>
      <div class="compare" id="compare">
        <img id="cmp-base" class="cmp-img" alt="Original scan">
        <div class="cmp-overlay-wrap" id="cmp-overlay-wrap"><img id="cmp-overlay" class="cmp-img" alt="AI overlay"></div>
        <span class="cmp-tag left">Original</span>
        <span class="cmp-tag right">AI overlay</span>
        <div class="cmp-handle" id="cmp-handle">
          <div class="cmp-handle-grip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 7l-5 5 5 5M16 7l5 5-5 5"/></svg></div>
        </div>
        <input type="range" id="cmp-slider" min="0" max="100" value="50" class="cmp-range" aria-label="Compare slider">
      </div>
    </div>

    <div class="gallery">
      <div class="panel">
        <div class="imgbox"><img id="img-mask"></div>
        <div class="cap"><span>Predicted tumor mask</span>
          <a class="dl" id="dl-mask" download="tumor_mask.png">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 21h16"/></svg>Save</a>
        </div>
      </div>
      <div class="panel">
        <div class="imgbox"><img id="img-overlay2"></div>
        <div class="cap"><span>Overlay (full)</span>
          <a class="dl" id="dl-overlay" download="tumor_overlay.png">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 21h16"/></svg>Save</a>
        </div>
      </div>
    </div>
  </section>

  <footer>
    <div class="disclaimer">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.3 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L14.7 3.86a2 2 0 0 0-3.4 0z"/></svg>
      Research proof-of-concept, trained on the public LGG dataset. Not a diagnostic device and not for clinical use.
    </div>
    TumorVision · brain-MRI segmentation POC
  </footer>
</div>

<script>
const drop = document.getElementById('drop'), input = document.getElementById('f'), go = document.getElementById('go');
let chosen = null;

drop.addEventListener('click', () => input.click());
drop.addEventListener('keydown', e => { if(e.key === 'Enter' || e.key === ' ') input.click(); });
input.addEventListener('change', () => input.files[0] && setFile(input.files[0]));
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

function setFile(file){
  chosen = file;
  drop.style.display = 'none';
  document.getElementById('file-row').style.display = 'flex';
  document.getElementById('fname').textContent = file.name;
  document.getElementById('fsize').textContent = (file.size/1024).toFixed(0) + ' KB';
  const preview = document.getElementById('local-preview');
  if(/\\.(png|jpe?g)$/i.test(file.name)){
    preview.src = URL.createObjectURL(file); preview.style.display = 'block';
    document.getElementById('thumb-icon').style.display = 'none';
  } else {
    preview.style.display = 'none'; document.getElementById('thumb-icon').style.display = 'block';
  }
  go.disabled = false;
  document.getElementById('status').textContent = ''; document.getElementById('status').className = '';
}
function clearFile(){
  chosen = null; input.value = '';
  drop.style.display = 'block'; document.getElementById('file-row').style.display = 'none';
  go.disabled = true;
}

fetch('/health').then(r => r.json()).then(d => {
  const dot = document.getElementById('health-dot'), text = document.getElementById('health-text');
  if(d.models_loaded){ dot.classList.add('ready'); text.textContent = 'API ready'; }
  else { text.textContent = 'Model not loaded'; }
  document.getElementById('badge-model').textContent =
    d.models_loaded ? `${d.models_loaded}-fold U-Net ensemble` : 'Model not loaded';
}).catch(() => { document.getElementById('health-text').textContent = 'API unreachable'; });

const compare = document.getElementById('compare'), slider = document.getElementById('cmp-slider');
slider.addEventListener('input', e => compare.style.setProperty('--pct', e.target.value + '%'));

async function run(){
  if(!chosen) return;
  const status = document.getElementById('status'), spin = document.getElementById('spin'), label = document.getElementById('go-label');
  go.disabled = true; spin.classList.add('on'); status.className = ''; status.textContent = ''; label.textContent = 'Analyzing…';

  const fd = new FormData(); fd.append('file', chosen);
  try {
    const r = await fetch('/predict', {method:'POST', body: fd});
    const d = await r.json();
    if(!r.ok){ throw new Error(d.detail || 'request failed'); }

    document.getElementById('cmp-base').src = d.original_png;
    document.getElementById('cmp-overlay').src = d.overlay_png;
    document.getElementById('img-mask').src = d.mask_png;
    document.getElementById('img-overlay2').src = d.overlay_png;
    document.getElementById('dl-mask').href = d.mask_png;
    document.getElementById('dl-overlay').href = d.overlay_png;
    slider.value = 50; compare.style.setProperty('--pct', '50%');

    const badge = document.getElementById('stat-detected'), icon = document.getElementById('icon-detected');
    badge.innerHTML = d.tumor_detected
      ? '<span class="pill pos"><span class="dot2"></span>Tumor detected</span>'
      : '<span class="pill neg"><span class="dot2"></span>No tumor</span>';
    icon.style.color = d.tumor_detected ? 'var(--danger)' : 'var(--ok)';
    icon.style.background = d.tumor_detected ? 'var(--danger-soft)' : 'var(--ok-soft)';
    document.getElementById('stat-pixels').textContent = d.tumor_pixel_count.toLocaleString();
    document.getElementById('stat-area').textContent = d.tumor_area_percent + '%';

    const results = document.getElementById('results');
    results.style.display = 'block';
    requestAnimationFrame(() => requestAnimationFrame(() => results.classList.add('show')));
    results.scrollIntoView({behavior:'smooth', block:'nearest'});
    status.textContent = '';
  } catch(e) {
    status.textContent = 'Error: ' + e.message; status.className = 'err';
  } finally {
    go.disabled = false; spin.classList.remove('on'); label.textContent = 'Analyze scan';
  }
}
</script>
</body></html>"""

RENDERED_DEMO_HTML = DEMO_HTML.replace("__FAVICON__", FAVICON)


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return RENDERED_DEMO_HTML


VOLUME_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TumorVision - 3D volume view</title>
<link rel="icon" href="__FAVICON__">
<style>
:root{
  --bg:#f6f7fb; --surface:#ffffff; --surface-2:#f8fafc; --border:#e6e9f0; --text:#0f1629; --muted:#667085;
  --brand:#4f46e5; --brand-2:#06b6d4; --brand-dark:#4338ca; --brand-soft:#eef1ff;
  --ok:#0e9f6e; --danger:#e11d48; --danger-soft:#fde8ee; --r-md:16px; --r-lg:22px;
  --shadow-sm:0 1px 2px rgba(16,24,40,.05); --shadow-md:0 4px 14px -4px rgba(16,24,40,.10), 0 1px 3px rgba(16,24,40,.06);
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0a0e1a; --surface:#121826; --surface-2:#0d1220; --border:#232b3d; --text:#eaeef7; --muted:#8b96ab;
         --brand:#818cf8; --brand-2:#22d3ee; --brand-dark:#a5b4fc; --brand-soft:#171d33; --ok:#34d399;
         --danger:#fb7185; --danger-soft:#301220; --shadow-sm:0 1px 2px rgba(0,0,0,.3);
         --shadow-md:0 4px 14px -4px rgba(0,0,0,.5), 0 1px 3px rgba(0,0,0,.4); }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif}
.page{max-width:1040px;margin:0 auto;padding:1.4rem 1.25rem 4rem}
.nav{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0 1.6rem}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:800;font-size:1.05rem}
.brand-mark{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,var(--brand),var(--brand-2));
            display:flex;align-items:center;justify-content:center}
.nav-link{color:var(--muted);text-decoration:none;font-size:.85rem;font-weight:600;border:1px solid var(--border);
          padding:.35rem .75rem;border-radius:999px}
.nav-link:hover{color:var(--brand);border-color:var(--brand)}
h1{font-size:1.4rem;margin:0 0 .3rem}
.sub{color:var(--muted);font-size:.92rem;margin-bottom:1.2rem;max-width:640px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);box-shadow:var(--shadow-md);
      padding:1.4rem;margin-bottom:1rem}
.picker{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center}
select{padding:.55rem .8rem;border-radius:10px;border:1px solid var(--border);background:var(--surface-2);
       color:var(--text);font-size:.9rem;min-width:220px}
button.primary{background:linear-gradient(135deg,var(--brand),var(--brand-2));color:#fff;border:none;
       padding:.6rem 1.3rem;border-radius:10px;font-size:.9rem;font-weight:700;cursor:pointer}
button.primary:disabled{opacity:.4;cursor:not-allowed}
button.ghost{background:transparent;border:1px solid var(--border);color:var(--text);padding:.5rem 1rem;
       border-radius:10px;font-size:.85rem;font-weight:700;cursor:pointer}
#pstatus{color:var(--muted);font-size:.85rem}
#pstatus.err{color:var(--danger)}

.layout{display:grid;grid-template-columns:1.3fr 1fr;gap:1rem}
@media (max-width:800px){ .layout{grid-template-columns:1fr} }
.viewer-wrap{position:relative;background:#05070d;border-radius:var(--r-md);overflow:hidden;aspect-ratio:1/1}
canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}
.viewer-controls{position:absolute;top:.6rem;right:.6rem;display:flex;gap:.4rem}
.vbtn{background:rgba(20,24,38,.75);color:#fff;border:1px solid rgba(255,255,255,.2);border-radius:8px;
     padding:.35rem .6rem;font-size:.74rem;font-weight:700;cursor:pointer}
.viewer-hint{position:absolute;left:.7rem;bottom:.6rem;color:rgba(255,255,255,.55);font-size:.72rem}
.viewer-empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#8b96ab;
              font-size:.9rem;text-align:center;padding:1.5rem}

.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem .8rem;font-size:.85rem;margin-bottom:.8rem}
.stat-grid .k{color:var(--muted)}
.stat-grid .v{font-weight:800;text-align:right}
.calib{font-size:.76rem;color:var(--muted);background:var(--surface-2);border:1px solid var(--border);
      border-radius:8px;padding:.6rem .7rem;margin-top:.2rem}
h3{font-size:.9rem;margin:1.1rem 0 .6rem}
.rep-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem}
.rep-card{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--surface-2)}
.rep-card img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#000;display:block}
.rep-cap{padding:.4rem .5rem;font-size:.7rem;color:var(--muted)}
.report-link{display:inline-flex;align-items:center;gap:.4rem;margin-top:1rem;color:var(--brand);
             text-decoration:none;font-weight:700;font-size:.88rem}
.report .rep-title{font-size:1.3rem;margin:.2rem 0}
.report .sub{max-width:none}
.meta{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;font-size:.85rem;color:var(--muted);margin-bottom:1rem}
.meta b{color:var(--text)}
.disclaimer{background:var(--danger-soft);border:1px solid var(--danger);color:var(--danger);border-radius:10px;
            padding:.9rem 1.1rem;font-size:.85rem;font-weight:600;margin-bottom:1rem}
#r-snap{width:100%;max-width:420px;border-radius:12px;background:#05070d;display:block;aspect-ratio:1/1}
.rtable{width:100%;border-collapse:collapse;font-size:.88rem}
.rtable td{padding:.4rem .2rem;border-bottom:1px solid var(--border)}
.rtable td:first-child{color:var(--muted)}
.rtable td:last-child{text-align:right;font-weight:700}
.verify{border:1.5px solid var(--text);border-radius:10px;padding:1.1rem 1.3rem}
.vrow{display:flex;gap:1.4rem;margin:.7rem 0;flex-wrap:wrap}
.field{flex:1;min-width:200px}
.field label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:.3rem}
.vline{border-bottom:1px solid var(--text);height:1.6rem}
.nline{border-bottom:1px solid var(--border);height:1.7rem}
.checks{display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.88rem;margin:.4rem 0}
.checks span{display:flex;align-items:center;gap:.4rem}
.box{width:14px;height:14px;border:1.5px solid var(--text);border-radius:3px;display:inline-block}
.verify input[type=text],.verify textarea{width:100%;padding:.55rem .7rem;border-radius:8px;border:1px solid var(--border);
  background:var(--surface-2);color:var(--text);font:inherit;font-size:.9rem}
.checks label{display:flex;align-items:center;gap:.4rem;cursor:pointer}
#sig{width:100%;max-width:600px;aspect-ratio:600/160;border:1px dashed var(--muted);border-radius:8px;background:#fff;
     touch-action:none;cursor:crosshair;display:block;margin-bottom:.4rem}
.err-msg{color:var(--danger);font-size:.85rem;margin-left:.8rem}
.stamp{display:inline-block;border:2px solid var(--ok);color:var(--ok);font-weight:800;font-size:.8rem;letter-spacing:.06em;
       padding:.25rem .7rem;border-radius:6px;margin-bottom:.6rem}
.stamp.amended{border-color:#d97706;color:#d97706}
.stamp.rejected{border-color:var(--danger);color:var(--danger)}
.para{font-size:.92rem;line-height:1.6;margin:.3rem 0}
.pre{white-space:pre-wrap}
.snap{width:100%;max-width:300px;border-radius:10px;background:#05070d;display:block;margin:.8rem 0}
.signoff{display:flex;gap:2.5rem;flex-wrap:wrap;margin-top:2rem;align-items:flex-end}
.signoff>div{flex:1;min-width:220px}
#f-sig{height:70px;max-width:100%;display:block;background:#fff;border-radius:4px}
.sig-cap{font-size:.78rem;color:var(--muted);margin-top:.3rem}
#final .rep-grid{grid-template-columns:repeat(3,1fr)}
body.show-final #report{display:none!important}
@media print{
  body:not(.show-final) #final{display:none!important}
  body.show-final #report{display:none!important}
  #f-sig{background:#fff}
}
@media print{
  :root{--bg:#fff;--surface:#fff;--surface-2:#fff;--border:#dfe3ea;--text:#0f1629;--muted:#5b6478;--danger:#b3261e;--danger-soft:#fdecea}
  .no-print,.nav,.picker-card,#results,footer,.page>h1,.page>.sub{display:none!important}
  body{background:#fff}
  .page{padding:0}
  .card{box-shadow:none;border:none;padding:0}
}
footer{margin-top:1.6rem;color:var(--muted);font-size:.78rem;text-align:center;line-height:1.6}
</style></head>
<body>
<div class="page">
  <nav class="nav">
    <div class="brand"><div class="brand-mark"></div>TumorVision - 3D volume</div>
    <a class="nav-link" href="/">&larr; single-slice view</a>
  </nav>
  <h1>3D tumor reconstruction</h1>
  <p class="sub">Picks one patient's full scan (every slice) from the LGG dataset, runs the same 5-fold ensemble
  on each slice, and stacks the results into one volume - so you can see the tumor's real 3D shape, not just
  one cross-section.</p>

  <div class="card picker-card">
    <div class="picker">
      <select id="pid"><option>Loading patients…</option></select>
      <button class="primary" id="go" onclick="analyze()" disabled>Analyze full scan</button>
      <span id="pstatus"></span>
    </div>
  </div>

  <div class="layout" id="results" style="display:none">
    <div class="card">
      <div class="viewer-wrap">
        <canvas id="c"></canvas>
        <div class="viewer-empty" id="viewer-empty">No tumor detected across this patient's scan - nothing to render.</div>
        <div class="viewer-controls">
          <button class="vbtn" id="auto-btn" onclick="toggleAuto()">Auto-rotate: on</button>
          <button class="vbtn" onclick="resetView()">Reset view</button>
        </div>
        <div class="viewer-hint">Drag to rotate &middot; scroll to zoom</div>
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Key findings</h3>
      <div class="stat-grid" id="stats"></div>
      <a class="report-link" href="#report">Full report below &darr;</a>
    </div>
  </div>

  <div class="card report" id="report" style="display:none">
    <div class="no-print" style="text-align:right;margin-bottom:.6rem">
      <button class="primary" onclick="window.print()">Print / Save report as PDF</button>
    </div>
    <h2 class="rep-title">TumorVision - AI-Assisted Segmentation Report</h2>
    <div class="sub">Volumetric summary reconstructed from a 2D 5-fold U-Net ensemble run across every slice of the patient's scan.</div>
    <div class="meta" id="r-meta"></div>
    <div class="disclaimer">This report is produced by a research prototype AI model (not an FDA/CE-cleared diagnostic device) and
    is NOT a diagnosis. It must be reviewed and verified by a qualified radiologist before being relied upon
    or forwarded to a treating physician.</div>

    <h3>3D tumor reconstruction</h3>
    <img id="r-snap" alt="3D tumor reconstruction snapshot">

    <h3>Volume summary</h3>
    <table class="rtable" id="r-table"></table>
    <div class="calib" id="calib"></div>

    <h3>Representative slices</h3>
    <div class="rep-grid" id="reps"></div>

    <h3 class="no-print">Radiologist verification</h3>
    <div class="verify no-print" id="verify-form">
      <div class="vrow">
        <div class="field"><label for="v-name">Reviewed by (full name)</label><input type="text" id="v-name" autocomplete="off" placeholder="Dr. ..."></div>
        <div class="field"><label for="v-role">Designation / registration no. (optional)</label><input type="text" id="v-role" autocomplete="off"></div>
      </div>
      <div class="field"><label>Decision</label>
        <div class="checks">
          <label><input type="radio" name="v-dec" value="confirmed" checked> Findings confirmed</label>
          <label><input type="radio" name="v-dec" value="amended"> Findings amended</label>
          <label><input type="radio" name="v-dec" value="rejected"> Findings rejected</label>
        </div>
      </div>
      <div class="field" style="margin-top:.6rem"><label for="v-notes">Radiologist's observations / amendments <span id="v-notes-req"></span></label>
        <textarea id="v-notes" rows="4" placeholder="e.g. Tumor margin on slices 14-16 slightly over-segmented; no midline shift seen."></textarea></div>
      <div class="field" style="margin-top:.6rem"><label>Signature (draw with mouse, finger or stylus)</label>
        <canvas id="sig" width="600" height="160"></canvas>
        <button type="button" class="ghost" onclick="clearSig()">Clear signature</button></div>
      <div style="margin-top:1rem"><button type="button" class="primary" onclick="verifyReport()">Verify &amp; generate final report</button>
        <span id="v-msg" class="err-msg"></span></div>
    </div>
  </div>

  <div class="card report" id="final" style="display:none">
    <div class="no-print" style="text-align:right;margin-bottom:.6rem">
      <button class="ghost" onclick="editReview()">&larr; Edit review</button>
      <button class="primary" onclick="window.print()">Print / Save final report as PDF</button>
    </div>
    <div class="stamp" id="f-stamp"></div>
    <h2 class="rep-title">Verified Radiology Report - AI-Assisted Brain MRI Tumor Segmentation</h2>
    <div class="meta" id="f-meta"></div>

    <h3>1. Findings (AI-assisted segmentation)</h3>
    <p id="f-findings" class="para"></p>
    <table class="rtable" id="f-table"></table>
    <div class="calib" id="f-calib"></div>
    <img id="f-snap" alt="3D tumor reconstruction" class="snap">
    <div class="rep-grid" id="f-reps"></div>

    <h3>2. Radiologist's verified observations</h3>
    <p class="para"><b>Decision:</b> <span id="f-dec"></span></p>
    <p class="para pre" id="f-notes"></p>

    <h3>3. Impression</h3>
    <p class="para" id="f-impr"></p>

    <div class="signoff">
      <div><img id="f-sig" alt="signature"><div class="vline"></div>
        <div class="sig-cap"><b id="f-name"></b><br><span id="f-role"></span></div></div>
      <div class="sig-date"><div class="vline" style="text-align:center;padding-top:.9rem" id="f-date"></div>
        <div class="sig-cap">Date / time of verification</div></div>
    </div>
    <div class="disclaimer" style="margin-top:1.2rem">Generated with a research prototype AI model (not an FDA/CE-cleared device). The AI output
    forms part of this report only to the extent stated in the radiologist's verified observations above.</div>
  </div>

  <footer>Axes are pixel (width/height) and slice-count (depth) units - not physically calibrated to mm/cm (see calibration note above).
  Research proof-of-concept, not for clinical use.</footer>
</div>

<script>
let CURRENT_PID = null;

fetch('/dataset/patients').then(r => r.json()).then(d => {
  const sel = document.getElementById('pid');
  sel.innerHTML = d.patients.map(p => `<option value="${p}">${p}</option>`).join('');
  document.getElementById('go').disabled = false;
}).catch(() => { document.getElementById('pstatus').textContent = 'Could not load patient list'; document.getElementById('pstatus').className = 'err'; });

async function analyze(){
  const pid = document.getElementById('pid').value;
  const go = document.getElementById('go'), status = document.getElementById('pstatus');
  go.disabled = true; status.className = ''; status.textContent = 'Running ensemble across every slice… (a few seconds)';
  try {
    const r = await fetch(`/dataset/patients/${pid}/volume`);
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'request failed');
    CURRENT_PID = pid;
    LAST = d;
    document.body.classList.remove('show-final'); document.getElementById('final').style.display = 'none';
    renderStats(d);
    renderReps(d.representative_slices);
    document.getElementById('report').style.display = 'block';
    document.getElementById('results').style.display = 'grid';
    if(d.pointcloud.length){
      document.getElementById('viewer-empty').style.display = 'none';
      initViewer(d.pointcloud);
    } else {
      document.getElementById('viewer-empty').style.display = 'flex';
      initViewer([]);
    }
    status.textContent = '';
  } catch(e) {
    status.textContent = 'Error: ' + e.message; status.className = 'err';
  } finally { go.disabled = false; }
}

function renderStats(d){
  const s = d.stats, ex = s.extent || {width_px:'-', height_px:'-', depth_slices:'-'};
  document.getElementById('stats').innerHTML = `
    <div class="k">Tumor detected</div><div class="v">${s.tumor_voxel_count > 0 ? 'Yes' : 'No'}</div>
    <div class="k">Slices with tumor</div><div class="v">${s.slices_with_tumor} / ${s.total_slices}</div>
    <div class="k">Volume (% of stack)</div><div class="v">${s.tumor_volume_percent_of_scanned_stack}%</div>
    <div class="k">Extent (w x h x d)</div><div class="v">${ex.width_px} x ${ex.height_px} x ${ex.depth_slices}</div>`;
  const mi = d.model_info;
  document.getElementById('r-meta').innerHTML = `
    <span>Patient / case ID: <b>${d.patient_id}</b></span>
    <span>Generated: <b>${d.generated}</b></span>
    <span>Model: <b>${mi.encoder}, ${mi.n_folds}-fold ensemble</b></span>
    <span>Test-set Dice (tumor slices): <b>${mi.test_dice_tumor}</b></span>`;
  document.getElementById('r-table').innerHTML = `
    <tr><td>Total slices in scan</td><td>${s.total_slices}</td></tr>
    <tr><td>Slices with tumor detected</td><td>${s.slices_with_tumor}</td></tr>
    <tr><td>Tumor voxel count</td><td>${s.tumor_voxel_count.toLocaleString()}</td></tr>
    <tr><td>Tumor volume (% of scanned stack)</td><td>${s.tumor_volume_percent_of_scanned_stack}%</td></tr>
    <tr><td>Extent - width (x)</td><td>${ex.width_px} px</td></tr>
    <tr><td>Extent - height (y)</td><td>${ex.height_px} px</td></tr>
    <tr><td>Extent - depth (z)</td><td>${ex.depth_slices} slices</td></tr>`;
  document.getElementById('calib').textContent = d.calibration;
}
// the report carries a snapshot of the live 3D view, refreshed whenever the view changes or before printing
function snapshot3d(){
  try { document.getElementById('r-snap').src = canvas.toDataURL('image/png'); } catch(e) {}
}
window.addEventListener('beforeprint', snapshot3d);

// --- radiologist verification -> standard verified report ---
let LAST = null;
const sig = document.getElementById('sig'), sctx = sig.getContext('2d');
let signing = false, signed = false;
sctx.lineWidth = 2.4; sctx.lineCap = 'round'; sctx.lineJoin = 'round'; sctx.strokeStyle = '#0f1629';
function sigPos(e){ const r = sig.getBoundingClientRect(); return [(e.clientX - r.left) * sig.width / r.width, (e.clientY - r.top) * sig.height / r.height]; }
sig.addEventListener('pointerdown', e => { signing = true; sig.setPointerCapture(e.pointerId); const [x,y] = sigPos(e); sctx.beginPath(); sctx.moveTo(x,y); sctx.lineTo(x+.1,y); sctx.stroke(); signed = true; });
sig.addEventListener('pointermove', e => { if(!signing) return; const [x,y] = sigPos(e); sctx.lineTo(x,y); sctx.stroke(); });
sig.addEventListener('pointerup', () => signing = false);
function clearSig(){ sctx.clearRect(0, 0, sig.width, sig.height); signed = false; }
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

document.querySelectorAll('input[name=v-dec]').forEach(r => r.addEventListener('change', () => {
  const dec = document.querySelector('input[name=v-dec]:checked').value;
  document.getElementById('v-notes-req').textContent = dec === 'confirmed' ? '(optional)' : '(required)';
}));
document.getElementById('v-notes-req').textContent = '(optional)';

function verifyReport(){
  const msg = document.getElementById('v-msg'); msg.textContent = '';
  const name = document.getElementById('v-name').value.trim();
  const role = document.getElementById('v-role').value.trim();
  const dec = document.querySelector('input[name=v-dec]:checked').value;
  const notes = document.getElementById('v-notes').value.trim();
  if(!name){ msg.textContent = 'Enter the name of the reviewing radiologist.'; return; }
  if(!signed){ msg.textContent = 'Please sign in the signature box.'; return; }
  if(dec !== 'confirmed' && !notes){ msg.textContent = 'Describe the amendment / reason for rejection.'; return; }

  const d = LAST, s = d.stats, ex = s.extent, now = new Date();
  const when = now.toLocaleString([], {dateStyle:'medium', timeStyle:'short'});
  const label = {confirmed:'FINDINGS CONFIRMED', amended:'FINDINGS AMENDED', rejected:'FINDINGS REJECTED'}[dec];
  const stamp = document.getElementById('f-stamp'); stamp.textContent = 'VERIFIED - ' + label; stamp.className = 'stamp ' + dec;

  document.getElementById('f-meta').innerHTML = `
    <span>Patient / case ID: <b>${esc(d.patient_id)}</b></span>
    <span>AI analysis: <b>${esc(d.generated)}</b></span>
    <span>Model: <b>${esc(d.model_info.encoder)}, ${d.model_info.n_folds}-fold ensemble</b></span>
    <span>Verified: <b>${esc(when)}</b></span>`;

  document.getElementById('f-findings').textContent = s.tumor_voxel_count > 0
    ? `Automated segmentation of the ${s.total_slices}-slice MRI stack identified a tumor region on ${s.slices_with_tumor} slice(s), `
      + `extending approximately ${ex.width_px} x ${ex.height_px} pixels in-plane over ${ex.depth_slices} slice(s) in depth, `
      + `occupying ${s.tumor_volume_percent_of_scanned_stack}% of the scanned stack (${s.tumor_voxel_count.toLocaleString()} voxels). `
      + `Measurements are in pixels / slice count and are not calibrated to physical units.`
    : `Automated segmentation of the ${s.total_slices}-slice MRI stack did not identify a tumor region.`;
  document.getElementById('f-table').innerHTML = document.getElementById('r-table').innerHTML;
  document.getElementById('f-calib').textContent = d.calibration;
  document.getElementById('f-reps').innerHTML = document.getElementById('reps').innerHTML;
  snapshot3d(); document.getElementById('f-snap').src = document.getElementById('r-snap').src;

  document.getElementById('f-dec').textContent = {confirmed:'The radiologist confirms the AI-assisted findings above.',
    amended:'The radiologist confirms the findings with the amendments below.',
    rejected:'The radiologist does not accept the AI-assisted findings.'}[dec];
  document.getElementById('f-notes').textContent = notes || 'No additional observations recorded.';
  document.getElementById('f-impr').textContent = {
    confirmed: s.tumor_voxel_count > 0 ? 'AI-segmented tumor region verified by the reviewing radiologist. Clinical correlation advised.'
                                       : 'No tumor region identified by AI segmentation; verified by the reviewing radiologist. Clinical correlation advised.',
    amended: 'AI-segmented findings reviewed and amended by the radiologist as recorded above. Clinical correlation advised.',
    rejected: 'AI-segmented findings were reviewed and rejected by the radiologist and must not be relied upon.'}[dec];

  document.getElementById('f-sig').src = sig.toDataURL('image/png');
  document.getElementById('f-name').textContent = name;
  document.getElementById('f-role').textContent = role;
  document.getElementById('f-date').textContent = when;

  document.getElementById('final').style.display = 'block';
  document.body.classList.add('show-final');
  document.getElementById('final').scrollIntoView();
}
function editReview(){
  document.body.classList.remove('show-final');
  const r = document.getElementById('report'); if(r.style.display !== 'none') r.scrollIntoView();
}
function renderReps(reps){
  document.getElementById('reps').innerHTML = Object.entries(reps).map(([label, v]) => `
    <div class="rep-card"><img src="${v.overlay_png}" alt="${label}">
      <div class="rep-cap"><b>${label}</b><br>#${v.slice_index} &middot; ${v.tumor_pixels.toLocaleString()} px</div>
    </div>`).join('');
}

// --- Dependency-free 3D point-cloud viewer (plain Canvas2D, no external library) ---
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let points = [], center = [0,0,0], maxRange = 1;
let yaw = 0.6, pitch = -0.35, zoom = 1, autoRotate = true, dragging = false, lastX = 0, lastY = 0;

function fitCanvas(){
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio; canvas.height = rect.height * devicePixelRatio;
}
window.addEventListener('resize', fitCanvas);

function initViewer(pc){
  points = pc; // each point is [z, y, x] voxel coords
  if(points.length){
    const zs = points.map(p=>p[0]), ys = points.map(p=>p[1]), xs = points.map(p=>p[2]);
    const bb = a => [Math.min(...a), Math.max(...a)];
    const [z0,z1] = bb(zs), [y0,y1] = bb(ys), [x0,x1] = bb(xs);
    center = [(z0+z1)/2, (y0+y1)/2, (x0+x1)/2];
    maxRange = Math.max(z1-z0, y1-y0, x1-x0, 1);
  } else { center = [0,0,0]; maxRange = 1; }
  fitCanvas(); resetView(); if(!animating) tick();
  setTimeout(snapshot3d, 400);
}
function resetView(){ yaw = 0.6; pitch = -0.35; zoom = 1; }
function toggleAuto(){ autoRotate = !autoRotate; document.getElementById('auto-btn').textContent = 'Auto-rotate: ' + (autoRotate ? 'on' : 'off'); }

canvas.addEventListener('pointerdown', e => { dragging = true; autoRotate = false; document.getElementById('auto-btn').textContent = 'Auto-rotate: off'; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => { if(!dragging) return; yaw += (e.clientX - lastX) * 0.008; pitch += (e.clientY - lastY) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch)); lastX = e.clientX; lastY = e.clientY; });
canvas.addEventListener('pointerup', () => { dragging = false; snapshot3d(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= (e.deltaY < 0 ? 1.08 : 0.93); zoom = Math.max(0.3, Math.min(5, zoom)); }, {passive:false});

function project(z,y,x, cy,sy,cx,sx, ox,oy, scale){
  const Z = z - center[0], Y = y - center[1], X = x - center[2];
  const x1 = X*cy + Z*sy, z1 = -X*sy + Z*cy;
  const y2 = Y*cx - z1*sx, z2 = Y*sx + z1*cx;
  const fov = 4 * maxRange;
  const f = fov / (fov + z2);
  return {sx: ox + x1*scale*f, sy: oy + y2*scale*f, depth: z2};
}

let animating = false;
function tick(){
  animating = true;
  if(autoRotate) yaw += 0.004;
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#05070d'; ctx.fillRect(0,0,w,h);
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
  const ox = w/2, oy = h/2, scale = (Math.min(w,h)/maxRange) * 0.9 * zoom;

  // bounding wireframe box for orientation
  const hz = (points.length ? maxRange : 1)/2;
  const corners = [];
  for(const dz of [-hz,hz]) for(const dy of [-hz,hz]) for(const dx of [-hz,hz]) corners.push([center[0]+dz, center[1]+dy, center[2]+dx]);
  const proj = corners.map(([z,y,x]) => project(z,y,x, cy,sy,cx,sx, ox,oy, scale));
  const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  ctx.strokeStyle = 'rgba(129,140,248,.35)'; ctx.lineWidth = 1;
  for(const [a,b] of edges){ ctx.beginPath(); ctx.moveTo(proj[a].sx, proj[a].sy); ctx.lineTo(proj[b].sx, proj[b].sy); ctx.stroke(); }

  if(points.length){
    const drawn = points.map(([z,y,x]) => project(z,y,x, cy,sy,cx,sx, ox,oy, scale));
    const order = drawn.map((_,i)=>i).sort((a,b) => drawn[b].depth - drawn[a].depth);
    for(const i of order){
      const p = drawn[i];
      const size = Math.max(1.2, 2.6 * (4*maxRange)/(4*maxRange + p.depth) * devicePixelRatio);
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(p.sx - size/2, p.sy - size/2, size, size);
    }
  }
  requestAnimationFrame(tick);
}
</script>
</body></html>"""

RENDERED_VOLUME_HTML = VOLUME_HTML.replace("__FAVICON__", FAVICON)


@app.get("/volume", response_class=HTMLResponse)
def volume_page():
    return RENDERED_VOLUME_HTML
