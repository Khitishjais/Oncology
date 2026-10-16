# TumorVision - Brain MRI Tumor Segmentation: One-Page Summary

**Dataset:** LGG MRI Segmentation (Kaggle, Mateusz Buda) - 110 patients, 3,929 slices, 256x256, 3-channel scans (pre-contrast/FLAIR/post-contrast), binary masks.
**Model:** U-Net, ResNet50 encoder (ImageNet pretrained), sigmoid output. Final deliverable is a **5-fold cross-validation ensemble** (5 U-Net models, one per fold, predictions averaged with flip test-time augmentation).
**Split:** patient-level, seed 42 - 77 train / 16 validation / 17 held-out test patients, no patient in more than one set. The 17-patient test set was never used for training or tuning and was evaluated only at the end.

## Final results (17 unseen test patients, 561 slices)

| Metric | Score |
|---|---|
| Pixel accuracy | 99.55% |
| Dice, all slices (empty-vs-empty scored 1.0) | 0.888 |
| IoU, all slices | 0.854 |
| **Dice, tumor-containing slices (224 slices) - the honest segmentation quality** | **0.787** |
| IoU, tumor-containing slices | 0.702 |
| 5-fold cross-validation estimate (93 patients, out-of-fold) | Dice(tumor) = 0.790 +/- 0.045 |

**Why two Dice numbers:** about 60% of test slices contain no tumor; scoring an empty prediction against an empty true mask as 1.0 inflates the "all slices" figure. The tumor-slice Dice (0.79) is the number that reflects actual outline quality and is the one we report as the headline result. Pixel accuracy is included for completeness but is not meaningful alone - tumor is a small fraction of each image, so even a poor model scores 98%+.

## What worked
- **U-Net + a pretrained (ImageNet) encoder** gave a strong, fast-converging baseline versus training from scratch.
- **Splitting by patient**, not by image, was essential - an image-level split would have let the model see near-duplicate slices from the same patient in both train and test, inflating the score artificially.
- **Oversampling tumor-containing slices** (drawn ~2x as often during training) mattered because only ~35% of slices contain a tumor; without it the model saw too few positive examples per epoch.
- **Dice+BCE loss** instead of plain accuracy - accuracy is misleading here since tumor pixels are a small minority.
- **Ensembling** (multiple models' predictions averaged) reliably improved the score over any single model: tumor Dice rose from 0.62 (single ResNet34 U-Net) to 0.76 (one tuned ResNet50 U-Net) to ~0.79-0.79 (ensembles of 3 and then 5 models).
- **5-fold cross-validation** confirmed the ~0.79 tumor Dice is a stable, reliable number rather than a lucky split: the out-of-fold estimate (93 patients) and the held-out test estimate (17 patients) agreed closely.

## What did not work / limitations
- **Post-processing** (removing small predicted blobs, tuning the detection threshold) gave little to no improvement - the model's predictions were already confident, so there was not much noise to clean up.
- **Small, irregularly shaped tumors remain the hardest case.** Sample predictions show near-perfect matches on clearly defined tumors (Dice ~0.9+) but weaker overlap on small or diffuse ones (Dice ~0.6), which pulls the average down.
- **Progress has plateaued around Dice 0.79 on tumor slices.** Four rounds of improvement (better encoder, learning-rate schedule, tumor oversampling, test-time augmentation, multi-architecture ensemble, 5-fold cross-validation) moved the score from 0.62 to ~0.79; a recall-weighted loss (Focal Tversky) aimed at the small-tumor case was planned as a fifth round but could not be run because Colab's free GPU quota was exhausted mid-session.
- **The test set is small (17 patients).** The 5-fold cross-validation's +/-0.045 spread across folds is a realistic estimate of how much the score can vary with a different set of patients; a single-number score without that context can be misleading.

## Deliverables produced
- Trained model: 5 fold weights (`fold0_model.pth` .. `fold4_model.pth`) + `postprocess_v4.json` (encoder, ensemble file list, tuned threshold=0.5, min_area=100).
- Inference script: `predict.py <scan.tif> [out_mask.png]` - loads all 5 models, applies flip test-time augmentation, and writes a binary tumor mask for any new scan.
- Evaluation report: `outputs/evaluation_report_v4.txt`, `outputs/cv_summary_v4.json`.
- Sample predictions: `outputs/sample_predictions_v4.png` (20 examples: Scan | True Mask | Predicted Mask) and `outputs/dice_hist_v4.png` (distribution of per-slice Dice on tumor slices).
- Full process record of every stage, decision and result: `PROCESS_LOG.md`.
