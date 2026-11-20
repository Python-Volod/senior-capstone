# Training and model comparison

The small 3D U-Net now has a CPU training workflow, saved checkpoints, whole-scan
evaluation, and a common evaluator for another model's exported predictions.
Real held-out performance is **not yet established**. A small six-patient
tumor-only pilot is now prepared locally (four train, one validation, one
reserved test); it exercises the pipeline and is not an adequate benchmark.
See [pilot run notes](pilot-training.md).

## 1. Prepare and review a local manifest

Confirm access terms, HU scaling, matching CT/mask grids, and the annotation
label definitions. Use expert lesion masks, not model-generated pseudo-labels
unless a separate experiment explicitly documents that choice.

Create `data/experiment/manifest.csv`, keeping actual patient identities local.
The following is a schema illustration, not an inventory of available data:

```csv
case_id,patient_id,ct,mask,split
case_a,patient_a,../cases/a_ct.nii.gz,../cases/a_mask.nii.gz,train
case_b,patient_b,../cases/b_ct.nii.gz,../cases/b_mask.nii.gz,val
case_c,patient_c,../cases/c_ct.nii.gz,../cases/c_mask.nii.gz,test
```

Paths resolve relative to the manifest. There must be at least one distinct
patient in each split; three patients only exercise the workflow and do not
make an adequate scientific benchmark. Choose cohort size and split fractions
with the team before experimentation. A patient may have multiple scans, but
all must stay in one split. Every case needs a unique case ID and separate
CT/mask paths. The validator rejects repeated paths and patient overlap.
It cannot detect a patient assigned multiple false IDs or duplicated files at
different paths: verify grouping and duplicates from source metadata first.

Do not fabricate additional patients by copying the sample, splitting its
slices, or calling patches different patients. Do not infer patient identity
from annotation IDs without checking the provider's mapping.

## 2. Train and save a baseline

PowerShell, using the existing environment:

```powershell
Set-Location C:\UTampa\CSC401.402\senior-capstone
$py = 'C:\UTampa\CSC401.402\lung-ct-capstone\.venv\Scripts\python.exe'
& $py -m model.experiment train --manifest data/experiment/manifest.csv --output outputs/experiments/unet3d-run1 --labels 1 --epochs 5 --patches-per-case 8 --patch-size 32
```

`--labels 1` is an example: replace it with confirmed lesion IDs. Multiple IDs
can be supplied, such as `--labels 1 2`. Selected IDs become binary foreground.
Unselected nonzero IDs cause an error rather than silently becoming negatives.
Label 0 is background. Datasets with ignore labels need an explicit policy
before this workflow can be used.

Defaults: four base channels, CPU, four threads, Adam at 0.001, BCEWithLogitsLoss,
seed 0, CT clipping from -1000 to 400 and fixed-range normalization. Training
samples half uniform and half foreground-centered patches when foreground is
present. Padding uses normalized air/background. This is an initial sampler,
not a guarantee of balanced patches or an optimal loss. No augmentation or
physical resampling is applied. Different voxel spacing remains a limitation.

Validation predicts **every voxel** using nonoverlapping 3D tiles, without using
the mask to locate lesions. Patch size also controls inference tiles; edge
padding is removed before scoring. Tile seams are a baseline limitation;
overlapping/blended sliding windows are a future refinement. Whole-scan CPU
validation may take considerably longer than the three-step patch demo.
The loader still materializes a whole case in memory; patching bounds model
activation size, not total file-loading memory.

Saved locally:

- `best.pt`: weights and optimizer state from the best validation macro-case Dice.
- `last.pt`: final epoch weights and optimizer state.
- `config.json`: hyperparameters and a digest of the resolved manifest.
- `history.json`: training-patch BCE and complete validation metrics each epoch.

The command refuses to overwrite a nonempty run directory. Training
loads train/validation volumes only. Merely checking that test paths exist
does not evaluate their contents.

### Continue a saved run

New checkpoints also save the optimizer, random-number-generator states, epoch,
and best validation score. Keep `last.pt`, `best.pt`, and `history.json` together.
Resume into a **new** directory, preserving the original run:

```powershell
& $py -m model.experiment resume --manifest data/experiment/manifest.csv --checkpoint outputs/experiments/unet3d-run1/last.pt --output outputs/experiments/unet3d-run2 --epochs 5 --threads 4
```

Here `--epochs 5` means five **additional** epochs. Resume reuses the checkpoint's
architecture, target labels, preprocessing, patch settings, seed and learning
rate; it restores Adam state and sampling RNG rather than restarting weights.
Use the original thread count and software/hardware for reproducibility.
Earlier checkpoints without RNG state are rejected. The new run retains the
earlier best model if validation never improves, and carries forward history.
If training was interrupted during a checkpoint/history write and their epochs
do not agree, this initial implementation rejects the inconsistent state.

## 3. Evaluate the frozen checkpoint

### Foreground-aware follow-up experiments

The initial real pilot converged to background-only validation output. To test
an alternative objective while preserving its weights and original artifacts:

```powershell
& $py -m model.experiment train --manifest data/nlstseg/pilot6/manifest.csv --output outputs/experiments/pilot6-foreground-run1 --initialize outputs/experiments/pilot6-run1/last.pt --labels 1 --loss dice_bce --positive-weight 20 --dice-weight 0.5 --foreground-probability 0.75 --inference-batch-size 8 --epochs 8 --patches-per-case 32 --patch-size 32 --threads 4
```

`--initialize` loads model weights but starts a new optimizer, epoch numbering
and validation selection. It records the source checkpoint hash and epoch.
This differs from exact `resume`, which retains the optimizer and objective.
Initialization requires the same patient manifest, labels, channel count and
intensity settings. Run directories may not be overwritten.

`dice_bce` mixes positive-weighted BCE and per-patch soft Dice with smoothing 1.
`--dice-weight .5` gives each term weight .5. `--positive-weight 20` weights
foreground voxels within BCE. These are explicit experimental settings, not
validated optimal values. BCE still penalizes empty patches. A foreground
sampling probability of .75 means positive-centered crops are chosen with that
probability when foreground exists; remaining crops are uniform and can also
contain foreground. Validation never uses annotation-guided crops.

Inference can batch independent tiles for CPU efficiency. Batch size is saved
in the configuration; the grid, threshold and coverage do not change. Timing
comparisons must state the batch size. Full-volume arrays still consume memory.
History records the actual objective as `train_patch_loss`, ordinary unweighted
BCE separately as `train_patch_bce`, and sampled foreground fraction. Losses
under different objectives should not be compared as identical quantities.

Keep the reserved test case unscored during these experiments. With only one
validation patient, repeated model selection can overfit that patient's anatomy;
a larger patient-level cohort is required before claiming generalization.

### Frozen test evaluation

Select architecture, hyperparameters, threshold, and checkpoint using training
and validation only. Then evaluate the held-out test set:

```powershell
& $py -m model.experiment evaluate --manifest data/experiment/manifest.csv --checkpoint outputs/experiments/unet3d-run1/best.pt --report outputs/experiments/unet3d-run1/test.json
```

The manifest must match the training digest; changing patients or assignments
requires a separate experiment. The digest covers the resolved manifest, not
the contents of scan files. Freeze the dataset files as well; changing files
in place would invalidate the experiment. Do not tune on test results or
repeatedly evaluate variations to select a model.

Reports include per-case metrics, pooled-voxel (micro) metrics, case-averaged
(macro) metrics and the number of cases contributing to each macro metric.
They also record an all-background baseline, parameter count, checkpoint hash,
environment, thread count, and CPU inference time. Timing covers network
tiling and output assembly, excludes loading/preprocessing, and includes no
separate warmup. It is not an end-to-end UI latency or peak-memory measurement.

Metric definitions, **at the voxel level**:

| Metric | Definition |
| --- | --- |
| Dice / binary F1 | 2TP / (2TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall / sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Accuracy | (TP + TN) / all voxels |

Both-empty prediction/target has Dice and IoU 1. Undefined precision, recall,
or specificity is JSON `null`, excluded from that macro mean, with its defined
case count reported. Missed foreground has Dice 0 and recall 0. Always show
foreground metrics alongside accuracy: many background voxels can inflate it.
These are not lesion-level sensitivity or false positives per scan.

## 4. Compare the teammate's 2D model

### Export a model prediction for review or a demo

```powershell
& $py -m model.predict --checkpoint outputs/experiments/pilot6-foreground-run1/best.pt --ct data/nlstseg/pilot6/pilot_05/ct.nii.gz --output outputs/experiments/pilot6-foreground-run1/validation_prediction.nii.gz --threads 4
```

This example uses the validation patient, not the reserved test patient. The
command takes no expert mask. It uses checkpoint intensity settings, threshold,
tile size and batch size, exports a binary mask in the canonical CT grid and
preserves source spatial units. Its JSON sidecar identifies it as a **model
prediction**, records the checkpoint hash, and does not claim accuracy. Output
files are not overwritten. Keep predictions local and clearly distinguish them
from expert annotations in any display. Values of 1 indicate predicted
foreground at the configured threshold, not a calibrated cancer probability.

### Shared scoring of exported predictions

Obtain the code, environment instructions, trained checkpoint, and training
provenance. Use the same patient splits, target labels, HU bounds, and scoring
grid. Keep all slices from one patient together. Tune each model only on
validation and document differences in sampling, training budget, augmentation,
and preprocessing. Report those differences rather than asserting a controlled
architecture-only experiment. A 2D model must reconstruct the full 3D output
for each test case, including negative slices, without annotation-guided crops.

Export binary 0/1 predictions as NIfTI with the original canonical grid/affine.
An example local prediction CSV, `data/experiment/2d_predictions.csv`:

```csv
case_id,prediction
case_c,../../outputs/experiments/unet2d/case_c_prediction.nii.gz
```

Include exactly one row per test case, no training or validation cases. Then:

```powershell
& $py -m evaluation.compare --manifest data/experiment/manifest.csv --reference-report outputs/experiments/unet3d-run1/test.json --predictions data/experiment/2d_predictions.csv --name teammate_2d_unet --output outputs/experiments/comparison1
```

The evaluator verifies the case set, grid, and binary predictions, then writes
`report.json` and `comparison.csv`. It cannot verify training/test contamination
inside another model: team review of provenance is mandatory for a fair claim.
Exported masks supply no trustworthy comparable timing; benchmark both models
on the same hardware and workload separately. Do not mix literature scores into
the measured comparison table. Add patient-level uncertainty estimates when a
sufficient cohort exists; none are implemented in this initial workflow.

## Software verification

```powershell
& $py -m pytest -q -p no:cacheprovider
```

Tests create separate synthetic patients and verify arithmetic, leakage checks,
whole-volume edge coverage, checkpoint reload, split immutability checks, and
comparison output. Synthetic test scores are never project accuracy results.
