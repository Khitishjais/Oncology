# TumorVision - Brain MRI Tumor Segmentation: Process Log

Living document. Every stage is recorded in order: what we did, why we did it, and what came out.
Project guide: `TumorVision_TeamGuide_Brain_MRI_Segmentation.pdf` (owner: Soumya Kant Prusty).
Goal: train a model that outlines brain tumors on MRI scans (segmentation) using the Kaggle LGG MRI Segmentation dataset.

---

## Stage 0 - Understanding the guide
**What:** Read the 9-section team guide and listed everything required.
**Why:** So nothing is missed before writing code.
**Requirements extracted**
- Deliverables: trained model file, script that makes a mask for a new MRI, evaluation report (Dice, IoU on unseen test images), 10-20 side-by-side samples (Scan | True Mask | Predicted Mask), one-page summary.
- Pipeline: explore data -> load/normalise/binarise -> split 70/15/15 **by patient** (seed 42) -> augmentation -> U-Net (ResNet34, ImageNet weights) -> train (Dice/Dice+BCE loss, Adam 1e-4, batch 16, early stopping, save best weights) -> evaluate on test -> package.
- Mistakes to avoid: split by image, plain accuracy as metric, augmenting scan but not mask, no early stopping, no normalisation, deleting empty-mask slices.

## Stage 1 - Environment and data (local, VS Code, `D:\oncology`)
**What**
- Found the dataset at `Downloads\archive.zip` (748 MB) and extracted it to `D:\oncology\data`.
- The zip contains two identical copies of the dataset (`kaggle_3m` and `lgg-mri-segmentation`, 7,860 files each). Only one copy was used; nothing is lost.
- Created a Python virtual environment `.venv` (Python 3.14) with PyTorch (CPU), segmentation-models-pytorch, albumentations, tifffile, scikit-learn, matplotlib, pandas, tqdm.
**Why:** PyTorch instead of the guide's Keras `segmentation_models`, because that library is TensorFlow-based and often breaks on new Python/TF versions. Same architecture and settings, different framework.
**Finding:** this laptop has no NVIDIA GPU, so full training on CPU would take ~12 hours.

## Stage 2 - Explore, prepare, split (`01_explore.py`, `02_prepare.py`)
- Counted: **110 patients, 3,929 slices, 1,373 with tumor, 2,556 empty.** (Matches the guide.)
- Empty-mask slices were **kept** (the model must learn what healthy tissue looks like).
- Scans divided by 255 (range 0-1); masks binarised (non-zero -> 1).
- **Patient-level split, seed 42:**

| Set | Patients | Slices | Tumor slices |
|---|---|---|---|
| Train | 77 | 2,750 | 915 |
| Validation | 16 | 618 | 234 |
| Test | 17 | 561 | 224 |

- **Why by patient:** slices of one patient look nearly identical; putting them in both train and test would inflate the score.
- Test set is used only once per model family, at the end.

## Stage 3 - Code written locally (smoke-tested)
Files: `common.py`, `01_explore.py`, `02_prepare.py`, `03_train.py` (augmentation, U-Net ResNet34, Dice+BCE, Adam 1e-4, early stopping), `04_evaluate.py`, `predict.py`.
- Augmentation: flips, rotation up to 15 deg, brightness/contrast (scan only). Geometric transforms are applied to scan and mask together; masks use nearest-neighbour so they stay binary.
- 1-epoch smoke test on 160 slices ran end to end (CPU ~115 s for 160 train + 618 val slices). Its Dice (0.04) is meaningless; the goal was only to prove the pipeline works.
- **Decision:** train on Google Colab (free T4 GPU) instead of the laptop.

## Stage 4 - v1 on Colab (`colab_train.ipynb`): the guide's baseline
**Setup:** U-Net / ResNet34 / ImageNet weights, Dice+BCE loss, Adam lr 1e-4, batch 16, up to 40 epochs, best epoch chosen by validation loss.
**Run:** 40 epochs at ~41 s each (~27 min). Best epoch 35 (val loss 0.1887). Train loss kept falling (~0.13) while val loss plateaued (~0.2) -> mild overfitting.
**Test result (17 unseen patients):**

| | Dice | IoU |
|---|---|---|
| All slices | 0.8369 | 0.8068 |
| Tumor slices only | 0.6184 | 0.5430 |

**Key lesson:** the all-slice number is inflated because ~60% of test slices are empty and correct "no tumor" predictions score 1.0. The tumor-slice number (0.62) is the true segmentation quality and was below the guide's 0.80 target.
**Mistake made and fixed during the run:** the `pip install` cell was skipped, giving `ModuleNotFoundError: segmentation_models_pytorch`; fixed by running that cell first.

## Stage 5 - v2 on Colab (`colab_train_v2.ipynb`): improved single model
**Changes vs v1 and why**

| Change | Why |
|---|---|
| ResNet50 encoder | Stronger feature extractor |
| Cosine learning-rate decay (2e-4 -> 1e-6) | Settles better than a fixed rate |
| Mixed precision (fp16) | Faster on T4 |
| Tumor slices sampled ~2x as often (~50% of each epoch) | Tumor slices are only ~33% of the data |
| Stronger augmentation (small shift/zoom added) | More variety, less memorising |
| Best epoch chosen by validation **Dice** (mean of all-slice and tumor-slice), not loss | Dice is what we are judged on |
| Threshold and minimum blob size tuned on **validation** | Never tune on test |
| Flip test-time augmentation (original + H-flip + V-flip averaged) | Small free accuracy gain |

**Run:** early stopping at epoch 48 (~35 s/epoch); best epoch 33 (val Dice all 0.898, tumor 0.816). Train and val loss tracked each other closely -> no overfitting. Validation Dice plateaued around epoch 30.
**Tuned post-processing:** threshold 0.4, min_area 0 (blob removal gave no benefit; predictions were already confident).
**Test result (17 unseen patients):**

| | v1 | **v2** |
|---|---|---|
| Pixel accuracy | not measured | **99.447%** |
| Dice, all slices | 0.8369 | **0.8843** |
| IoU, all slices | 0.8068 | **0.8506** |
| Dice, tumor slices | 0.6184 | **0.7594** |
| IoU, tumor slices | 0.5430 | **0.6749** |

**Notes:** (1) the validation-vs-test gap (0.816 vs 0.759 tumor Dice) is expected: the best epoch was picked on validation and the test set has only 17 patients. (2) "Pixel accuracy" is high for almost any model because tumor is a tiny fraction of pixels; always report it together with Dice. (3) The report's last line says "no TTA tuning" but TTA is used there; label only, numbers unaffected.

## Stage 6 - Realistic targets (discussion)
- 95%+ **pixel accuracy**: achieved (99.4%).
- 95%+ **Dice on tumor slices**: not realistic. Published results on this dataset are roughly 0.82-0.90; values near 1.0 usually mean leakage or a bug.
- Practical target for further work: tumor-slice Dice ~0.80-0.82.

## Stage 7 - v3 (`colab_train_v3.ipynb`): ensemble of diverse models
**Plan:** keep the v2 model, train two more with different architectures/encoders and seeds (Unet++ / ResNet50, Unet / ResNeXt50), then average their predictions. Ensembles usually add ~1-3% Dice because different models make different mistakes.
**Safeguards:** same patient split; every model saved to Drive as soon as it finishes (re-running skips finished models); the single model or the ensemble is **chosen on validation only**, then evaluated once on test.
**Hiccup during the run:** the Colab session restarted mid-notebook, giving `NameError: name 'torch' is not defined` on the predict cell (later cells were run without re-running the earlier import cells). No work was lost: each model is saved to Drive as soon as it finishes, and the training cell skips any model whose weights file already exists. Fix: re-run the notebook top to bottom; training cells skipped instantly since the files already existed.

**Validation scores (mean of all-slice Dice and tumor-slice Dice), used to choose the final model:**

| Model | thr / min_area | Score |
|---|---|---|
| v2_unet_resnet50 (reused) | 0.4 / 0 | 0.8655 |
| m3_unetpp_resnet50 (new) | 0.7 / 60 | 0.8301 |
| m4_unet_resnext50 (new) | 0.3 / 60 | 0.8601 |
| **ensemble_all (all three averaged)** | 0.4 / 60 | **0.8693 - chosen** |

**Test result (17 unseen patients), chosen = ensemble_all, threshold 0.4, min_area 60, flip TTA on:**

| | v1 | v2 | **v3 (ensemble)** |
|---|---|---|---|
| Pixel accuracy | not measured | 99.447% | **99.509%** |
| Dice, all slices | 0.8369 | 0.8843 | 0.8707 |
| IoU, all slices | 0.8068 | 0.8506 | 0.8359 |
| Dice, tumor slices | 0.6184 | 0.7594 | **0.7922** |
| IoU, tumor slices | 0.5430 | 0.6749 | **0.7052** |

**Reading the result:** the ensemble's all-slice Dice is slightly below v2's, but the number that matters (tumor-slice Dice) is the best so far: 0.62 -> 0.76 -> 0.79 across v1 -> v2 -> v3. Sample predictions still show the hardest cases are small, irregular tumor shapes, where the predicted blob is close in position but not a pixel-perfect match (e.g. one sample scored Dice 0.63 despite a visually reasonable overlap).
**Files saved to Drive:** `best_model_v2.pth` (reused), `best_model_m3.pth`, `best_model_m4.pth`, `postprocess_v3.json` (records the chosen model, threshold, min_area), `evaluation_report_v3.txt`, `sample_predictions_v3.png`, `dice_hist_v3.png`.
Superseded by v4 below (statistically tied score, but v4's cross-validation confirms it's a reliable number rather than a lucky split).

## Stage 8 - v4 (`colab_train_v4.ipynb`): 5-fold cross-validation ensemble
**Plan:** the 93 patients that were train+val in v1/v2/v3 were split into 5 folds (KFold, shuffle, seed 42); one Unet/ResNet50 model was trained per fold (same recipe as v2: Dice+BCE, cosine LR, tumor-oversampling, mixed precision). The held-out 17-patient test set (identical patients as v1/v2/v3) was **not used in any fold's training**.
**Why:** the previous validation set (16 patients) and test set (17 patients) are both small, so a single split's score can be somewhat lucky/unlucky. Cross-validation gives 5 independent estimates plus a much larger (93-patient) out-of-fold set for tuning the detection threshold, instead of only 16 patients.
**Bugs hit and fixed during the run (both mine, in the generated notebook, not the user's fault):**
1. The "Mount Drive and unzip" markdown heading was included but its code cell was accidentally left out of the notebook -> `FileNotFoundError` on the data-loading cell. Fixed by correcting the cell range in `make_notebook_v4.py` and re-uploading the notebook.
2. The out-of-fold probability array was created with the wrong shape (`np.zeros(len(Xp))`, 1-D) when each prediction is a full 256x256 mask -> `TypeError` on assignment. Fixed to `np.zeros(Yp.shape)`. No retraining was needed; the fix only affected the post-training prediction step, and all 5 fold models were already saved to Drive.
**Training:** each fold ran up to 35 epochs (early stopping, patience 8), ~35s/epoch; all 5 folds completed without loss.

**Out-of-fold cross-validation scores (each patient scored only by the fold model that never trained on them):**

| Fold | Patients | Dice (all) | Dice (tumor) |
|---|---|---|---|
| 0 | 19 | 0.7787 | 0.8344 |
| 1 | 19 | 0.9091 | 0.8023 |
| 2 | 19 | 0.8830 | 0.7194 |
| 3 | 18 | 0.8756 | 0.8347 |
| 4 | 18 | 0.8004 | 0.7575 |
| **Mean (93 patients)** | | **0.8493** | **0.7896 +/- 0.0451** |

The +/-0.045 spread across folds shows the natural patient-to-patient variability of the result.
Post-processing (threshold=0.5, min_area=100) was tuned on these out-of-fold predictions only.

**Test result (same 17 unseen patients as v1/v2/v3, ensemble of the 5 fold models, flip TTA):**

| | v1 | v2 | v3 | **v4 (5-fold ensemble)** |
|---|---|---|---|---|
| Pixel accuracy | not measured | 99.447% | 99.509% | **99.547%** |
| Dice, all slices | 0.8369 | 0.8843 | 0.8707 | 0.8881 |
| IoU, all slices | 0.8068 | 0.8506 | 0.8359 | 0.8542 |
| Dice, tumor slices | 0.6184 | 0.7594 | **0.7922** | 0.7867 |
| IoU, tumor slices | 0.5430 | 0.6749 | **0.7052** | 0.7018 |

**Reading the result:** v4's test tumor-Dice (0.7867) is statistically indistinguishable from v3's (0.7922) - both sit inside v4's own cross-validation spread (0.79 +/- 0.045). The valuable outcome of v4 is not a higher number but **confidence**: the out-of-fold estimate (0.7896) and the held-out test estimate (0.7867) agree closely, which shows the true, reliable segmentation quality of this approach is **~0.78-0.79 tumor-slice Dice**, not a lucky single split.
**Conclusion: v1->v2->v3 gave large gains (0.62 -> 0.76 -> 0.79); v4 confirms that level rather than exceeding it. Further architecture/ensemble tweaks are expected to give only marginal (low single-digit percent) gains from here.**
**Files saved to Drive:** `fold0_model.pth` .. `fold4_model.pth`, `cv_summary_v4.json`, `postprocess_v4.json`, `evaluation_report_v4.txt`, `sample_predictions_v4.png`, `dice_hist_v4.png`.

## Stage 9 - v5 attempted (Focal Tversky loss), blocked by GPU quota
**Plan:** train one more model with Focal Tversky loss (alpha=0.3, beta=0.7, gamma=0.75 - weights missed tumor more than over-drawn tumor, aimed at the small/hard tumors that pull the average down), same train/val/test split as v2 for a direct comparison; bonus cell would add it to the v4 5-fold models for a 6-model ensemble.
**Outcome:** Colab's free GPU quota ran out mid-notebook ("Cannot connect to GPU backend"). No training happened. Decision: stop chasing further score gains here (v1->v2->v3->v4 already moved tumor Dice from 0.62 to ~0.79 and v4's cross-validation confirmed that level is reliable; further gains are expected to be marginal) and finalize the deliverables with the v4 5-fold ensemble as the shipped model. `colab_train_v5.ipynb` / `make_notebook_v5.py` are kept for later if GPU access frees up (Colab quota reset, or Kaggle Notebooks as an alternative free-GPU source).

## Stage 10 - Deliverables finalised
- Downloaded from Drive into `D:\oncology\outputs\`: `fold0_model.pth` .. `fold4_model.pth`, `postprocess_v4.json`, `evaluation_report_v4.txt`, `sample_predictions_v4.png`, `cv_summary_v4.json`, `dice_hist_v4.png`.
- `predict.py` rewritten to load all 5 fold models, average their flip-TTA predictions, and apply the tuned threshold/min_area from `postprocess_v4.json`.
- **Verified locally:** ran `predict.py` on `data/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_11.tif` (a training-set scan, used only to confirm the script works end to end) -> predicted mask vs true mask Dice = 0.931. This number is expected to be high because the model saw this patient during training; it is not a substitute for the 0.79 test-set score, only a smoke test that the inference code is correct.
- Wrote `SUMMARY.md`: the guide's Step 8 one-page summary (final numbers, what worked, what did not, limitations, list of deliverables).
- **Final shipped model: v4, the 5-fold cross-validation ensemble.** Test-set Dice(tumor) = 0.7867, IoU(tumor) = 0.7018, pixel accuracy = 99.547%.

## Stage 11 - Backend + demo UI (Guide section 8: "Deploying the model as an API")
**What:** built `backend/app.py`, a FastAPI service wrapping the v4 5-fold ensemble.
- `GET /health` - model/ensemble status.
- `POST /predict` - accepts a scan (`.tif`, or PNG/JPG), runs the 5-model ensemble with flip TTA, returns JSON: tumor detected (bool), tumor pixel count, tumor area %, and three base64 PNGs (original scan re-encoded, predicted mask, red overlay on the scan).
- `GET /` - a self-contained HTML demo page (no external dependencies, works offline): drag-and-drop upload, loading state, a 3-panel result view (Original scan | Predicted mask | Overlay), stat cards (result pill, pixel count, area %), a clinical-looking header/badges, dark-mode support, and a "research POC, not for clinical use" disclaimer.
- `GET /docs` - automatic interactive API docs (FastAPI/Swagger).
**Verified:** started the server, confirmed all 5 fold models load; called `/predict` on `sample_scan.tif` (a known tumor-containing slice) via curl - got 1,459 tumor pixels, matching `predict.py`'s local result exactly (same ensemble + TTA logic in both). Saved and visually checked the overlay image; the red highlight lines up with the true tumor location. Confirmed in-browser after redesigning the page (see screenshot review) - the 3-panel view, stat cards, and original-scan preview all render correctly.
**Sample test file:** `sample_scan.tif` / `sample_scan.png` at the project root (a training-set slice with a visible tumor, 1,426 true tumor pixels) - kept only as a quick manual smoke-test image, not part of the evaluation.
**How to run:** from `D:\oncology`, `.venv\Scripts\python -m uvicorn backend.app:app --port 8000`, then open `http://127.0.0.1:8000`.
**Not done / out of scope so far:** authentication, HTTPS, deployment to a real server (currently local-only), and handling of scans that are not 256x256 3-channel LGG-style slices (the guide's dataset format is assumed).

## Stage 12 - Manager asked for >=95% accuracy; v6 planned (2026-09-24)
**Context:** manager asked for a minimum of 95% accuracy. Two different things can be meant by "accuracy" here, and the distinction matters:
- **Pixel accuracy** (fraction of all pixels classified correctly): already **99.55%** on the test set - exceeds 95%. This number is included for completeness but is not a meaningful target on its own, because tumor pixels are a small fraction of each image (~2-5%); a model that predicts "no tumor" everywhere would still score ~97-98% pixel accuracy without detecting anything.
- **Dice/IoU on tumor slices** (how well the predicted tumor outline actually overlaps the true one - the honest segmentation-quality number): currently **0.7867** (~79%). 95% is not a realistic target here: published results on this exact dataset (LGG MRI Segmentation) sit roughly in the 0.82-0.90 range, and Dice values near 0.95+ on medical segmentation tasks like this one are usually a sign of data leakage or a bug, not genuine quality.
**Decision (with the user):** treat "95% accuracy" as referring to Dice/IoU segmentation quality (the harder, more meaningful reading), and make one more real attempt to push it higher rather than stopping at v4's ~0.79 - while being upfront that 95% itself is very unlikely to be reached.
**Plan - v6 (`colab_train_v6.ipynb` / `make_notebook_v6.py`):** train 2 new models, diverse from everything trained so far, on the same 77/16/17 patient split (seed 42) as v1/v2/v3/v5:
1. `m6_effnet` - Unet with an **EfficientNet-B4** encoder (stronger/deeper than the ResNet50 used in every model so far), Dice+BCE loss.
2. `m6_unetpp_ft` - **Unet++** architecture (used in v3) combined with **Focal Tversky loss** (used alone in v5, never run) - a combination not tried before.
Then build the largest ensemble yet: these 2 new models + the v4 5-fold models (7 models total, loaded not retrained), flip TTA, threshold/min_area tuned on the 16-patient validation set only, evaluated once on the 17-patient test set.
**Expectation set in advance, so the result isn't a surprise either way:** v4's own conclusion (Stage 8) was that further architecture/ensemble tweaks give only marginal (low single-digit percent) gains from ~0.79. Realistic target for v6 is **~0.80-0.83 tumor-slice Dice**, not 0.95.

**Run completed (2026-09-25).** Validation scores (mean of all-slice + tumor-slice Dice, used to pick the model before touching test):

| Model | thr / min_area | Score |
|---|---|---|
| m6_effnet (EfficientNet-B4, alone) | 0.5 / 60 | 0.8793 |
| m6_unetpp_ft (Unet++ + Focal Tversky, alone) | 0.3 / 30 | 0.7656 (worst - this combination did not work well on this dataset) |
| ensemble_balanced (2 new + v4-mean, 3-way) | 0.3 / 60 | 0.8868 |
| **ensemble_all (2 new + all 5 v4 folds, 7-way)** | 0.5 / 60 | **0.8952 - chosen** |

**Test result (17 unseen patients, 561 slices, 224 with tumor), chosen = ensemble_all, threshold 0.5, min_area 60, flip TTA:**

| | v3 | v4 (shipped) | **v6 (ensemble_all)** |
|---|---|---|---|
| Pixel accuracy | 99.509% | 99.547% | 99.545% |
| Dice, all slices | 0.8707 | 0.8881 | 0.8917 |
| IoU, all slices | 0.8359 | 0.8542 | 0.8583 |
| **Dice, tumor slices** | 0.7922 | **0.7867** | **0.7823** |
| IoU, tumor slices | 0.7052 | 0.7018 | 0.6988 |

**Honest reading: v6 did NOT improve on v4.** Tumor-slice Dice went from 0.7867 (v4) to 0.7823 (v6) - a 0.4-point drop, well inside v4's own cross-validation spread (±0.045 / ~4.5 points), so this is noise, not a real regression or a real gain. Notably, `ensemble_all` scored much higher on validation (0.8952) than v4's out-of-fold estimate (0.8493), but that advantage disappeared on the held-out test set - a sign of mild overfitting to the small (16-patient) validation set used to choose the ensemble, not a genuine improvement.
**Conclusion: v6 confirms, rather than breaks, the plateau identified at v4.** A stronger encoder (EfficientNet-B4), a new architecture+loss combination (Unet++ + Focal Tversky), and a bigger 7-model ensemble all failed to move the number past ~0.79. **~0.78-0.79 tumor-slice Dice is the practical ceiling for this approach** (U-Net-family models, this dataset's 110 patients, 2D-slice-based training) - not the 95% the manager asked for. Reaching materially higher would need a different order of resource: substantially more labeled data, 3D volumetric context instead of independent 2D slices, or a medical-imaging foundation model - out of scope for this project as currently defined.
**Decision: v4 remains the shipped model** (simpler - 5 models vs v6's 7 - and marginally better on the actual test set). `colab_train_v6.ipynb` / `make_notebook_v6.py` and this result are kept as the record that the higher-accuracy attempt was made and what it found, for reporting to the manager.

## Stage 13 - 3D volumetric U-Net attempted (the one lever not yet tried)
**Why:** every model so far (v1-v6) predicted each 256x256 slice **independently** - the model never saw the slices above/below it, even though a tumor is a real 3D blob spanning many slices. v6 confirmed that pushing harder on 2D-only techniques (stronger encoder, new loss, bigger ensemble) does not move the number past ~0.79. Real 3D context - letting the model use neighbouring slices - is the one substantively different approach left to try before concluding 95% (or any further gain) is out of reach for this project.
**Plan (`colab_train_3d.ipynb` / `make_notebook_3d.py`):**
- Stack each patient's slices into one `(3 channels, D slices, 256, 256)` volume, in the **correct numeric z-order** (filenames sort lexicographically, which is the wrong order for stacking - "_11" would sort before "_2" - so slices are ordered by their numeric index instead).
- **Same 77/16/17 patient split (seed 42)** as v1-v6, built with identical code, so results are directly comparable.
- **MONAI** (medical-imaging deep learning library) 3D U-Net, trained on randomly-sampled 3D patches (32 slices x 128 x 128) rather than full volumes, because a full 256x256xD volume does not fit in a free-tier T4 GPU's memory - the standard trade-off for 3D medical segmentation on limited hardware.
- Patches are sampled with `RandCropByPosNegLabeld` so training sees tumor-containing regions often enough (the 3D analogue of v1-v6's "oversample tumor slices" trick); patients with fewer than 32 slices (47 of 110 patients - dataset ranges from 20 to 88 slices per patient) are padded first so a patch can still be cropped.
- Evaluation uses **sliding-window inference** to stitch patches back into a full-size prediction per patient, so despite training on patches, the reported result covers the whole image - then the same per-slice metrics (all-slice/tumor-slice Dice, IoU, pixel accuracy) used for v1-v6 are computed, for a direct comparison, plus whole-volume 3D Dice (the metric most 3D segmentation papers report).
**Honest expectation set in advance:** this is not expected to reach 95% either - published 3D brain-tumor benchmarks (BraTS) that reach 0.85-0.92 Dice use 1000+ patients and 4 MRI modalities; this dataset has 110 patients and 3 channels. A meaningful result would be **anything clearly above 0.79** (real evidence 3D context helps); "about the same as v4" is also a genuine, useful answer - it would mean 110 patients is too little to learn reliable 3D shape, closing off this direction with confidence rather than a guess.
**Status:** written, not yet run. This is the project's first 3D/MONAI code, so - matching the pattern of every earlier notebook's first real run (v1's missing pip cell, v4's two bugs) - expect it may need a debugging pass or two once actually run on a Colab GPU; that is normal, not a sign the plan is wrong.

## Stage 14 - 3D view, real-size extent, and a radiologist report added to the backend
**Ask:** implement a 3D view, report the tumor's actual length/breadth/area, and generate a report for a radiologist to verify before it goes to a doctor.
**Critical check done first:** whether "actual" (mm/cm) length/breadth is even possible with this data. Checked `data/README.md` and `data/data.csv` - **neither contains pixel-spacing or slice-thickness metadata** (the original DICOM geometry was stripped when the dataset was converted to `.tif`). Decision (confirmed with the user): report size **only in pixels (in-plane) and slice-count (depth)**, never a fabricated mm/cm figure - every place a size number appears also carries an explicit "uncalibrated" note explaining why, so nothing looks more precise/real than it is.
**What was built, all in `backend/app.py`, reusing the already-shipped v4 5-fold ensemble (no new model needed for this):**
- `GET /dataset/patients` - lists the 110 patient IDs available locally (`data/<patient_id>/`).
- `GET /dataset/patients/{pid}/volume` - runs the 5-fold ensemble + flip TTA across **every slice** of one patient (batched, not one API call per slice), in the correct numeric z-order (same lexicographic-order pitfall as Stage 13's `_11`-before-`_2` issue, handled the same way), and returns: total slices, slices with tumor, tumor voxel count, tumor volume as % of the scanned stack, the tumor's 3D extent (width/height in pixels, depth in slice-count), a downsampled (z,y,x) point cloud of the tumor's voxels for 3D rendering, and 3 representative labelled slice overlays (largest tumor slice, first/last tumor-containing slice).
- `GET /dataset/patients/{pid}/report` - a print-ready HTML report: patient ID, model version, the volume summary table above (with the calibration note repeated), the representative slice images, a mandatory disclaimer ("research prototype, not a diagnosis, must be reviewed by a qualified radiologist"), and a **Radiologist verification** section (reviewed-by/date fields, confirmed/amended/rejected checkboxes, notes lines, signature line) plus a Print/Save-as-PDF button. Per the user's direction, the system only *generates* this report - actually sending it to a doctor is left to the radiologist, not automated.
- `GET /volume` - a new page (linked from the main demo page's nav) with a patient picker, the stats/representative-slices panel, a "Generate radiologist report" link, and a **dependency-free 3D point-cloud viewer** built with plain Canvas2D (no Three.js/WebGL library) - drag to rotate, scroll to zoom, auto-rotate toggle, reset view, a bounding wireframe box for orientation. Built without an external 3D library deliberately: no CDN script to verify/trust, one less thing that can silently fail to load.
**Verified (server-side, via curl):** `/dataset/patients` returns all 110 IDs; `/dataset/patients/TCGA_CS_4941_19960909/volume` returns correct, internally-consistent numbers (23 total slices, 8 with tumor, extent width 81px/height 124px/depth 8 slices, 14,508 tumor voxels, point-cloud length matches voxel count exactly since under the 15,000-point downsampling cap) - and the tumor-containing slice range (10-17) correctly includes slice 11, the same slice used as the Stage-10 smoke test; `/dataset/patients/.../report` renders correctly with all fields filled in and no template-formatting errors; `/volume` page HTML/JS is structurally sound (balanced braces/parens, loads with HTTP 200).
**Not verified:** the actual in-browser 3D rendering (canvas drag-to-rotate, point cloud draw) - no headless-browser tool (Playwright/chromium-cli) is available in this environment, so this could only be checked by logic review and a server-side HTML/JS structure check, not a real screenshot. **Needs a manual check in an actual browser** before being trusted.
**Performance note:** running the ensemble across a whole patient (5 models x 3 TTA passes x ~20-90 slices) takes **~60 seconds on this CPU-only laptop** for the volume/report endpoints - much slower than the ~1s single-slice `/predict`. Expected given no GPU is available locally; flagged to the user rather than silently shipped as "slow for no reason."

## Reproduction
1. `Downloads\archive.zip` -> `D:\oncology\data` (one copy).
2. Local: `.venv\Scripts\python 01_explore.py`, then `02_prepare.py`.
3. Colab: upload `archive.zip` to `MyDrive/oncology/`, upload a notebook, set T4 GPU, run all cells in order (do not skip the pip cell).
4. Files saved to `MyDrive/oncology/`; copy them into `D:\oncology\outputs\`.

## File map
| File | Purpose |
|---|---|
| `common.py` | Shared helpers (data loading, Dice/IoU, model builder) |
| `01_explore.py` .. `04_evaluate.py`, `predict.py` | Local pipeline scripts (v1 design; `predict.py` updated for the v2 model) |
| `colab_train.ipynb` / `make_notebook.py` | v1 Colab notebook and its generator |
| `colab_train_v2.ipynb` / `make_notebook_v2.py` | v2 Colab notebook and its generator |
| `colab_train_v3.ipynb` / `make_notebook_v3.py` | v3 (ensemble) Colab notebook and its generator |
| `colab_train_v5.ipynb` / `make_notebook_v5.py` | v5 (Focal Tversky) Colab notebook and its generator - written, blocked by GPU quota, not yet run |
| `colab_train_v6.ipynb` / `make_notebook_v6.py` | v6 (EfficientNet-B4 + Unet++/Focal Tversky, mega-ensemble with v4) Colab notebook and its generator - run, confirmed the ~0.79 plateau (Stage 12) |
| `colab_train_3d.ipynb` / `make_notebook_3d.py` | 3D volumetric U-Net (MONAI) Colab notebook and its generator - written, not yet run (Stage 13) |
| `outputs/` | Local results (v1 files present; v2/v3 files to be downloaded from Drive) |
| `PROCESS_LOG.md` | This document |
| `IMPLEMENTATION.md` | Technical reference: architecture, pipeline, model, backend API, file map |
