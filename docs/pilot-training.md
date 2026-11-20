# First saved real-data pilot — November 15

This is a small training experiment, not a reliable test of clinical accuracy.

## Completed first run

The first run completed five epochs and 320 optimizer updates. Training-patch
BCE fell from 0.372275 to 0.110875, but validation Dice remained 0 at every
epoch. The last model predicted only background on the validation scan. This
does not establish useful tumor segmentation. All validation scores tied, so
`best.pt` retains epoch 1, while `last.pt` contains epoch 5 and is the continuation
checkpoint. Reload and a finite forward pass were verified. The reserved test
patient was not scored. Before a long continuation, investigate class imbalance,
foreground-aware loss/sampling and a larger validation cohort using validation
only. Local run details and plots are in
`outputs/experiments/pilot6-run1/training_report.md`.

## Foreground-aware follow-up

After the original run, a new experiment initialized from its epoch-5 weights
and completed eight additional epochs / 1,024 optimizer updates. It uses
positive-weighted BCE (weight 20) mixed equally with soft Dice, 75% foreground
sampling, and 32 patches per training case. The optimizer was reset explicitly
for the new objective. The original checkpoint/run was preserved. Inference
uses batches of eight independent tiles; scoring still covers the whole scan.

The best checkpoint is epoch 8 of this follow-up. On the single validation
patient: Dice **0.003449**, IoU **0.001727**, precision **0.001733**, recall
**0.350892**. Counts are 806 TP, 464,287 FP and 1,491 FN voxels. This is no longer
a background-only output, but false positives are extensive and segmentation
quality remains poor. No independent test accuracy is claimed; the reserved
patient remains unscored. Do not report the foreground recall without also
showing precision or Dice.

Local artifacts: `outputs/experiments/pilot6-foreground-run1/`. Use `best.pt` for
the selected prediction model and `last.pt` to continue the same objective with
the saved optimizer. The new `model.predict` command exports a full-volume
binary prediction using the CT and checkpoint only. See the
[training/evaluation guide](training-evaluation.md) for commands and limitations.

## Data and split details

Six distinct patient IDs are selected using seed 42 from tumor-only cases
available in the second NLSTseg image archive. Eligibility is checked against
the local Patient and Label metadata tables. Selection is random among eligible
cases, rather than selecting the smallest scans or choosing by performance.
The first four selected patients train, the fifth validates, and the sixth is
reserved for later testing. Splits are frozen before training.

[The dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12375103/) distinguishes
voxel annotation IDs (`Mark_labels`) from lesion types (`labels_type`: tumor 1,
nodule 2). This pilot selects patients whose recorded lesions are all tumor
type. Each mask's observed nonzero IDs must exactly match its table annotation
IDs. All selected tumor IDs become binary foreground during training; original
NIfTI masks are preserved. This does not mean label 1 is the only possible
tumor annotation ID.

Files are retrieved selectively from the
[Zenodo dataset](https://zenodo.org/records/14838349) with ZIP member CRC checking.
The whole archive MD5 is not checked with selective retrieval. The earlier
large local ZIP is incomplete and is not used for extraction. Downloads,
source identifiers, manifest, provenance, images, reports, and weights remain
under ignored local data/output directories.

Local manifest: `data/nlstseg/pilot6/manifest.csv`. Provenance including selected
members and source IDs: `data/nlstseg/pilot6/provenance.json`. Initial checks
compare CT/mask shapes and affines, inspect labels and intensity ranges, and
save annotation overlays in `outputs/experiments/pilot6-qc/`. Dataset authors
describe HU preprocessing; range checks do not independently certify original
scanner calibration or prove annotation correctness. No spacing resampling
or anatomical registration is applied.

## Training protocol

Initial run directory: `outputs/experiments/pilot6-run1/`.
Five epochs, 16 patches per training case per epoch, 32-cubed patches,
four base channels, CPU with four threads, seed 0, Adam at 0.001, BCE loss,
HU bounds -1000 to 400, foreground threshold 0.5. This is 320 optimizer steps.
Whole-volume validation runs after every epoch using the predefined tiling
procedure. The best checkpoint is chosen by validation macro-case Dice.
The reserved test patient is checked for data integrity during preparation
but is not used for training, hyperparameter selection, or test scoring.

## Continuing later

After the initial run completes, continue into a new directory:

```powershell
Set-Location C:\UTampa\CSC401.402\senior-capstone
$py = 'C:\UTampa\CSC401.402\lung-ct-capstone\.venv\Scripts\python.exe'
& $py -m model.experiment resume --manifest data/nlstseg/pilot6/manifest.csv --checkpoint outputs/experiments/pilot6-run1/last.pt --output outputs/experiments/pilot6-run2 --epochs 5 --threads 4
```

Resume restores weights, Adam state, sampling RNG, and epoch counters, keeps
prior history, and retains the earlier best model if validation does not improve.
Do not repeatedly inspect the reserved test set to decide how long to train.

## Limits

One validation patient makes checkpoint selection unstable. All six patients
come from one archive and have annotated tumors; there is no cancer-negative
patient cohort here. Background patches are not equivalent to negative-patient
coverage. Raw voxel spacing varies, patch context is limited, and nonoverlapping
inference tiles may create seams. A 3D implementation with a working optimizer
may still converge to background or perform poorly on unseen data. Use the
recorded validation Dice/IoU and error counts honestly, alongside the background
baseline. More patients and a common 2D comparison protocol remain necessary.
