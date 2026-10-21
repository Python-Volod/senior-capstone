# Senior Capstone Project UTampa

## Overview

This project investigates computer vision for segmenting lung lesions in CT
scans. Segmentation means marking the individual 3D image points, called voxels,
that belong to a lesion, rather than assigning one label to an entire scan.

The initial goal is to load one real CT scan and its matching expert annotation,
verify that they align, display the annotation over a scan slice, and prepare
the data for a future segmentation model.

This is an educational/research prototype and is not intended for clinical
diagnosis or treatment decisions.

## Target architecture

```text
CT volume + expert segmentation mask
                  |
                  v
Load files and verify dimensions and spatial alignment
                  |
                  v
Preprocessing: intensity clipping, normalization, tensor conversion
                  |
                  v
3D U-Net: predict a segmentation for the CT volume
                  |
                  v
Compare predicted mask with expert mask
                  |
                  v
Evaluate segmentation quality and inspect overlays
```

The expert mask supplies the reference during training and evaluation. At
inference time, the intended model takes only the CT volume as input.

### Data and preprocessing

Python utilities will load CT scans and their corresponding expert masks,
starting with NIfTI files. The exact dataset and annotation format still need
to be verified against a real sample. Datasets will be obtained separately and
will not be committed to this repository.

The pipeline will check image geometry, normalize CT intensities, preserve
discrete mask labels, and convert each case into PyTorch tensors. Any future
resampling must account for voxel spacing and use nearest-neighbor
interpolation for masks.

### Segmentation model

The planned starting architecture is a simple 3D U-Net:

- An encoder learns image features at progressively smaller spatial scales.
- A bottleneck combines features at the deepest level.
- A decoder restores the spatial resolution to produce a voxel-level output.
- Skip connections carry detail from the encoder into the decoder.

The initial binary segmentation target is an input shaped `[B,1,D,H,W]` and
an output of voxel logits with the same shape. Here, `B` is batch size, `1`
is the grayscale channel, and `D,H,W` describe the 3D volume. Lesion label
definitions must be confirmed before preparing binary targets. Model size,
patch size, and training settings will depend on available hardware.

### Evaluation

Planned metrics include Dice coefficient, IoU, precision, recall, and voxel
false-positive and false-negative counts. Dice will be a primary segmentation
metric. All scans from the same patient must remain in one train, validation,
or test partition to prevent data leakage.

No trained model results are available. Performance metrics are **N/A**.

### Future application layer

A minimal Python API, potentially using FastAPI, may eventually accept a CT,
run preprocessing and inference, and return segmentation results. A simple
web interface may display those results. Framework choices are provisional;
frontend development is a lower priority than verifying the data pipeline.

Temporal comparison between scans is a stretch goal, dependent on suitable
repeated scans and reliable lesion correspondence.

## Current status and workflow

This README describes the intended architecture, not a completed system.
Setup documentation is being introduced on `feature/project-setup`. Changes
will be added in small steps, reviewed through `develop`, and brought into
`main` when ready.

The next milestone is verifying one real CT volume and its matching expert
lesion mask end to end before implementing full model training.
