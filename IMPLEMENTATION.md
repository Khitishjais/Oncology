# TumorVision - Implementation Reference

Technical reference for the shipped system: how the code is structured, what each
module does, and how to run it. For the story of *how* we got here (decisions,
experiments, dead ends) see `PROCESS_LOG.md`. For the one-page results summary see
`SUMMARY.md`. This document does not repeat those - it documents the implementation
as it stands.

**Shipped model:** v4, a 5-fold cross-validation ensemble of U-Net/ResNet50 models.
Test-set Dice (tumor slices) = 0.7867, IoU = 0.7018, pixel accuracy = 99.55%.

---

## 1. Architecture

```
data/<patient>/*.tif  --(common.py: collect_pairs, load_pair)-->  (256,256,3) scan, (256,256) mask
                                          |
                        02_prepare.py: patient-level 70/15/15 split (seed 42)
                                          |
                 03_train.py / colab_train_v4.ipynb: 5-fold U-Net/ResNet50 training
                                          |
                 outputs/fold0_model.pth .. fold4_model.pth + postprocess_v4.json
                                          |
        +---------------------------------------------------------+
        |                                                          |
predict.py (CLI, one .tif in -> mask.png out)          backend/app.py (FastAPI service)
                                                                    |
                                                        GET /health, POST /predict, GET /
                                                                    |
                                                        browser demo page (self-contained HTML)
```

Both `predict.py` and `backend/app.py` implement the *same* inference recipe
independently (load 5 folds -> flip TTA -> average -> threshold -> min-area filter).
They are kept in sync by hand; there is no shared inference module beyond
`common.py`'s `build_model`/`to_tensor`.

## 2. Data pipeline (`common.py`, `01_explore.py`, `02_prepare.py`)

- **Source:** Kaggle LGG MRI Segmentation dataset, 110 patients, 3,929 slices,
  256x256x3 (pre-contrast/FLAIR/post-contrast channels), paired `<slice>.tif` /
  `<slice>_mask.tif`.
- `collect_pairs()` walks `data/<patient_id>/` and pairs each scan with its mask by
  filename convention (`X.tif` <-> `X_mask.tif`).
- `load_pair()` normalises the scan to `[0,1]` (divide by 255) and binarises the mask
  (`> 0`). Empty-mask (no-tumor) slices are **kept**, not dropped - the model needs
  healthy-tissue examples.
- **Split:** patient-level (not image-level), `sklearn` with `SEED=42` ->
  77 train / 16 val / 17 test patients, held constant across every model version
  (v1-v4) so results are comparable. The 17-patient test set is touched only once,
  at final evaluation.
- **Normalisation for the encoder:** `to_tensor()` applies ImageNet mean/std
  (`MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`) after the `/255` scaling,
  since the encoder backbone is ImageNet-pretrained and expects that distribution.

## 3. Model (`common.py: build_model`, training notebooks)

- **Architecture:** U-Net (`segmentation_models_pytorch.Unet`), configurable encoder.
  Shipped model uses `encoder="resnet50"`, `encoder_weights="imagenet"`,
  `in_channels=3`, `classes=1`. Output is a raw logit map; sigmoid is applied at
  inference, not inside the model.
- **Loss:** `DiceBCELoss` (`common.py`) - `BCEWithLogits` + `(1 - Dice)`, Dice computed
  per-image with a Laplace smoothing term of 1 in numerator/denominator.
- **Training recipe (v4, `colab_train_v4.ipynb`):**
  - 5-fold `KFold(shuffle=True, seed=42)` over the 93 train+val patients (the 17-patient
    test set is excluded from all folds).
  - One U-Net/ResNet50 trained per fold, up to 35 epochs, early stopping (patience 8),
    cosine LR decay 2e-4 -> 1e-6, Adam, mixed precision (fp16), batch 16.
  - Tumor-containing slices oversampled (~2x) per epoch since they are a minority
    (~35% of slices).
  - Augmentation (train only, scan+mask together): horizontal/vertical flip, rotation
    up to 15deg, small shift/zoom (geometric transforms use nearest-neighbour on the
    mask to keep it binary); brightness/contrast jitter on the scan only.
  - Best epoch per fold chosen by validation Dice (mean of all-slice and tumor-slice
    Dice), not loss.
- **Weights:** `outputs/fold0_model.pth` .. `fold4_model.pth` (raw `state_dict`,
  loadable via `build_model(pretrained=False, encoder=...); model.load_state_dict(...)`).

## 4. Inference: ensemble + test-time augmentation

Both `predict.py` and `predict_mask()` in `backend/app.py` run the identical recipe:

1. For each of the 5 fold models, run the image and its horizontal- and
   vertical-flip 3 times, sigmoid each, un-flip the flipped outputs, and average
   the 3 -> one probability map per fold (flip TTA).
2. Average the 5 folds' probability maps -> single ensemble probability map.
3. Threshold at `cfg["threshold"]` (0.5) to get a binary mask.
4. If the predicted tumor area is smaller than `cfg["min_area"]` (100 px), discard it
   entirely (treat as no tumor) - removes small spurious blobs.

Config driving this (`outputs/postprocess_v4.json`):
```json
{
  "encoder": "resnet50",
  "n_folds": 5,
  "fold_files": ["fold0_model.pth", "fold1_model.pth", "fold2_model.pth", "fold3_model.pth", "fold4_model.pth"],
  "threshold": 0.5,
  "min_area": 100
}
```
Threshold and min_area were tuned on the 93-patient out-of-fold predictions only,
never on the test set.

## 5. Backend API (`backend/app.py`)

FastAPI app, CORS open to all origins (local/demo use only - see Section 7).

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | `{status, models_loaded, encoder, threshold, min_area}` - confirms the 5 fold weights loaded at startup. |
| `/predict` | POST | `multipart/form-data`, field `file` (`.tif`/`.tiff` or any Pillow-readable image). Returns JSON (below). |
| `/` | GET | Self-contained HTML demo page (drag-and-drop upload, 3-panel result view, dark mode). No external JS/CSS dependencies. |
| `/dataset/patients` | GET | Lists the 110 local dataset patient IDs, for the 3D volume view / report. |
| `/dataset/patients/{pid}/volume` | GET | Runs the 5-fold ensemble across **every slice** of one patient, stacks the result into a 3D mask, returns volume stats, extent, a tumor-voxel point cloud, and representative slice overlays. See Section 8. |
| `/dataset/patients/{pid}/report` | GET | Print-ready HTML report (disclaimer + volume summary + representative slices + a radiologist verification section: reviewed-by/date/confirmed-amended-rejected/notes/signature). Generation only - sending it onward is manual. |
| `/volume` | GET | 3D volume view page: patient picker, stats panel, dependency-free Canvas2D 3D point-cloud viewer (drag-rotate, zoom, no external library), link to the report. |
| `/docs` | GET | Auto-generated Swagger UI (FastAPI default). |

**`POST /predict` response:**
```json
{
  "filename": "sample_scan.png",
  "tumor_detected": true,
  "tumor_pixel_count": 1459,
  "tumor_area_percent": 2.226,
  "original_png": "data:image/png;base64,...",
  "mask_png": "data:image/png;base64,...",
  "overlay_png": "data:image/png;base64,...",
  "model_info": {"encoder": "resnet50", "n_folds": 5, "test_dice_tumor": 0.7867}
}
```
- `read_scan()` accepts `.tif`/`.tiff` (via `tifffile`) or standard image formats (via
  Pillow), converts grayscale to 3-channel, and resizes anything not already 256x256.
- `overlay_png` is the scan with the predicted tumor region tinted red at 45% alpha
  (`make_overlay()`), for a doctor-readable view.
- Startup (`load_models()`) loads `outputs/postprocess_v4.json` and each listed fold
  file; if the config or a weight file is missing, `/predict` returns `503` and
  `/health` reports `models_loaded: 0` rather than crashing the process.

## 6a. 3D volume view, extent, and radiologist report (`/dataset/*`, `/volume`)

Reuses the shipped 2D 5-fold ensemble - no separate 3D-trained model is needed for
this (that's the pending, unrun `colab_train_3d.ipynb` from Section 3/Stage 13, a
different effort aimed at *improving accuracy*; this feature is about *visualizing*
what the existing model already predicts, across a whole patient rather than one slice).

- **No physical units.** `data/README.md` and `data/data.csv` were checked and contain
  no pixel-spacing/slice-thickness metadata (stripped when the dataset was converted
  to `.tif`). Every size figure this feature reports is therefore in **pixels**
  (in-plane) and **slice-count** (depth) only - never a fabricated mm/cm number. The
  `CALIBRATION_NOTE` constant in `backend/app.py` is attached to both the `/volume`
  JSON response and the printable report so this limitation travels with the data.
- `load_patient_stack(pid)` loads one patient's slices in **correct numeric z-order**
  (filenames sort lexicographically - `_11` would sort before `_2` - so the slice
  index is parsed and sorted numerically instead).
- `predict_volume_batch()` is the batched, whole-stack version of `predict_mask()`:
  same 5-model ensemble, same flip TTA, same threshold/min-area rule (so any single
  slice's result matches `/predict` exactly), but processes all of a patient's slices
  in one batched pass instead of slice-by-slice.
- `volume_stats()` computes tumor voxel count, % of the scanned stack, and the 3D
  bounding-box extent (width/height in px, depth in slice-count) from `np.nonzero`.
- `tumor_pointcloud()` returns the tumor's (z,y,x) voxel coordinates, downsampled to
  at most 15,000 points, for the 3D viewer.
- The `/volume` page's 3D viewer is **plain Canvas2D with a hand-rolled perspective
  projection** (yaw/pitch rotation matrix applied per point, painter's-algorithm
  depth sort) - deliberately not Three.js/WebGL, so there's no external CDN script
  that could silently fail to load.
- **Verified so far:** all endpoints tested end-to-end via curl with real data
  (correct, internally-consistent numbers). **Not verified:** the actual in-browser
  3D rendering - no headless-browser tool is available in this environment, so the
  canvas drag/rotate/zoom behaviour needs a manual check in a real browser.
- **Performance:** ~60s per patient on this CPU-only laptop for `/dataset/patients/{pid}/volume`
  and `/report` (5 models x 3 TTA x ~20-90 slices), vs ~1s for single-slice `/predict`.

## 6. File map

| File | Role |
|---|---|
| `common.py` | Shared helpers: `collect_pairs`, `load_pair`, `dice_iou`, `DiceBCELoss`, `build_model`, `to_tensor`. Imported by every script/notebook and by `backend/app.py`. |
| `01_explore.py` | Dataset stats (patient/slice/tumor counts). |
| `02_prepare.py` | Patient-level train/val/test split (seed 42). |
| `03_train.py`, `04_evaluate.py` | Local (CPU) v1-design pipeline scripts; superseded by the Colab notebooks for actual training, kept for the smoke-test/reference path. |
| `predict.py` | CLI inference: `python predict.py scan.tif [out_mask.png]`. Loads the 5-fold ensemble + `postprocess_v4.json`, writes a binary mask PNG. |
| `colab_train_v4.ipynb` / `make_notebook_v4.py` | Notebook (and its generator script) that trained the shipped v4 5-fold ensemble on Colab. |
| `colab_train.ipynb`/`_v2`/`_v3` + matching `make_notebook*.py` | Earlier model versions, superseded by v4 (see `PROCESS_LOG.md` Stages 4-8). |
| `colab_train_v5.ipynb` / `make_notebook_v5.py` | Focal Tversky loss experiment, written but blocked by Colab GPU quota - not yet run (`PROCESS_LOG.md` Stage 9). |
| `colab_train_v6.ipynb` / `make_notebook_v6.py` | EfficientNet-B4 + Unet++/Focal-Tversky, ensembled with the v4 5 folds. Run - result 0.7823 tumor Dice, essentially flat vs v4 (`PROCESS_LOG.md` Stage 12). Confirmed the 2D-approach ceiling; v4 stays shipped. |
| `colab_train_3d.ipynb` / `make_notebook_3d.py` | 3D volumetric U-Net (MONAI) - stacks each patient's slices into one volume so the model sees depth context, unlike every 2D model above. Written, not yet run (`PROCESS_LOG.md` Stage 13). |
| `backend/app.py` | FastAPI service: `/health`, `/predict`, `/` demo page, `/docs`. |
| `outputs/` | `fold0_model.pth`..`fold4_model.pth`, `postprocess_v4.json`, `evaluation_report_v4.txt`, `cv_summary_v4.json`, `sample_predictions_v4.png`, `dice_hist_v4.png`. |
| `sample_scan.tif` / `.png` | One known-tumor training-set slice, kept only as a manual smoke-test image (not part of evaluation). |
| `data/` | Extracted dataset (`<patient_id>/<slice>.tif` + `<slice>_mask.tif`). |
| `PROCESS_LOG.md` | Stage-by-stage history: decisions, experiments, bugs hit and fixed, numbers at each stage. |
| `SUMMARY.md` | One-page results summary (guide deliverable). |
| `IMPLEMENTATION.md` | This document. |

## 7. Running it

**Setup (once):** `.venv` (Python 3.14) with PyTorch (CPU), `segmentation-models-pytorch`,
`albumentations`, `tifffile`, `scikit-learn`, `matplotlib`, `pandas`, `tqdm`, `fastapi`,
`uvicorn`, `python-multipart`, `Pillow`.

**CLI inference on a new scan:**
```
D:\oncology\.venv\Scripts\python predict.py path\to\scan.tif out_mask.png
```
Requires `outputs/fold0_model.pth`..`fold4_model.pth` and `outputs/postprocess_v4.json`.

**Backend API + demo page:**
```
cd D:\oncology
.venv\Scripts\python -m uvicorn backend.app:app --port 8000
```
(run from `D:\oncology` so `backend` is importable as a package and `backend/app.py`'s
`sys.path.insert` to the repo root resolves `common.py` correctly). Then:
- `http://127.0.0.1:8000` - demo page
- `http://127.0.0.1:8000/docs` - interactive API docs
- `http://127.0.0.1:8000/health` - readiness check

**Retraining / reproducing from scratch:** see `PROCESS_LOG.md` -> "Reproduction"
section for the full data-download-to-Colab-to-local-outputs path.

## 8. Known gaps (not implemented)

- No authentication, no HTTPS, no deployment beyond local (`127.0.0.1`) - CORS is
  wide open (`allow_origins=["*"]`), which is fine for a local demo but not for any
  networked deployment.
- No handling for scans that aren't 256x256 3-channel LGG-style slices beyond a naive
  resize (`read_scan()`); non-brain-MRI input will silently produce a meaningless
  prediction rather than being rejected.
- No batching - `/predict` handles one image per request, single-threaded model
  inference (CPU).
- `predict.py` and `backend/app.py` duplicate the TTA/ensemble logic rather than
  sharing a function from `common.py`; a change to one must be mirrored in the other.
