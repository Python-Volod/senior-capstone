# Training-scan error mining and varied patch coverage

## November 17 result

Eight epochs / 1,024 new optimizer updates completed. The run made 394 mined
hard-negative draws with replacement. Epoch 3 was selected by whole-scan
validation Dice; the final epoch was worse and was not promoted.

| Same validation patient | Prior foreground-aware model | Hard-negative follow-up |
| --- | --- | --- |
| Dice | 0.003449 | 0.008429 |
| Precision | 0.001733 | 0.004254 |
| Recall | 0.350892 | 0.452764 |
| True-positive voxels | 806 | 1,040 |
| False-positive voxels | 464,287 | 243,419 |
| False-negative voxels | 1,491 | 1,257 |

This is an improvement on one repeatedly consulted validation patient, not a
reliable independent accuracy estimate. Absolute segmentation quality is still
poor, and several intermediate epochs had zero validation overlap. More data
and broader validation are necessary. The reserved test patient was not scored.
Local run artifacts are in `outputs/experiments/pilot6-hardneg-run1/`; use
`best.pt` (epoch 3) for the selected candidate and `last.pt` (epoch 8) to resume.

## Method

The fixed-patch diagnostic established that the network can fit two selected
tumor patches from one training patient. It did not establish whole-scan
generalization. This follow-up addresses false positives using **training-only**
error mining and a broader distribution of crop positions.

## Mining protocol

Freeze the selected foreground-aware checkpoint and original patient splits.
Predict each complete training CT with its saved threshold, intensity settings,
tile size and inference batch size. No validation/test CT or mask is loaded by
the mining command. Rank full 32-cubed tiles with **zero selected target voxels**
by predicted false-positive count, retaining up to 32 per training case.

The initial audit found 507,968 / 653,073 / 601,352 / 628,784 FP voxels across
the four training scans and selected 128 tiles total. This is training error,
not held-out performance. "Negative" means no selected expert annotation in
the tile; it does not guarantee that a region is clinically normal.

The fixed index records source checkpoint hash, manifest digest, preprocessing,
threshold, size and D/H/W coordinates. Training rejects held-out case entries,
different settings, out-of-bounds tiles, and any mined crop containing a target.
Resuming verifies that the index file's SHA256 is unchanged. The pool is mined
once; this experiment does not dynamically refresh it as the model changes.

```powershell
Set-Location C:\UTampa\CSC401.402\senior-capstone
$py = 'C:\UTampa\CSC401.402\lung-ct-capstone\.venv\Scripts\python.exe'
& $py -m model.hard_negatives --manifest data/nlstseg/pilot6/manifest.csv --checkpoint outputs/experiments/pilot6-foreground-run1/best.pt --output outputs/experiments/hard-negatives1 --limit-per-case 32 --threads 4
```

## Follow-up training protocol

Start from the same selected checkpoint in a new run, with a fresh Adam
optimizer. Keep the previous objective (20-weighted BCE mixed equally with soft
Dice), learning rate 0.001, seed 0, four channels, 32-cubed patches, threshold
0.5, eight-tile inference batches, four CPU threads, and the same patient splits.

For each training crop, the planned mixture is 50% foreground-centered, 37.5%
mined hard negative, and 12.5% uniform. These are sampling probabilities; actual
counts and foreground fractions are logged. Uniform crops may also include
foreground. If a training case has no mined candidates, that draw falls back
to uniform sampling. Positive crop origins are jittered by up to eight voxels
per axis while retaining the selected target voxel. This reduces reliance on
lesions always appearing at crop centers. It adds spatial variety, not new
patients. No intensity augmentation, physical resampling or inference blending
is introduced in this run.

```powershell
& $py -m model.experiment train --manifest data/nlstseg/pilot6/manifest.csv --output outputs/experiments/pilot6-hardneg-run1 --initialize outputs/experiments/pilot6-foreground-run1/best.pt --labels 1 --loss dice_bce --positive-weight 20 --dice-weight 0.5 --foreground-probability 0.5 --hard-negatives outputs/experiments/hard-negatives1/index.json --hard-negative-probability 0.375 --foreground-jitter 8 --inference-batch-size 8 --epochs 8 --patches-per-case 32 --patch-size 32 --threads 4
```

This is a combined sampling experiment; it does not isolate the individual
effects of mining and jitter. It runs eight epochs / 1,024 new updates. Checkpoint
selection uses complete validation scans and macro-case Dice. Review recall
alongside precision/FP counts: a lower FP count from predicting only background
would not constitute success. Preserve the previous model if this experiment
does not improve validation. The reserved test patient stays unscored.

## Remaining limitations

Four training patients and one repeatedly consulted validation patient are still
too few to establish generalization. The fixed pool covers only one checkpoint's
errors. Missing annotations, tile boundary/context effects, different voxel
spacing and limited lesion diversity remain possible issues. More patients and
a larger independent validation cohort are still required; this run expands
patch coverage within the existing cohort, not dataset size.
