# Inspect a CT and expert mask

Run from the repository folder with its Python environment active. Supply
paths to a matching NIfTI CT and expert mask; no identifiers are hard-coded.

```bash
python -m preprocessing.inspect_pair --ct data/sample/ct.nii.gz --mask data/sample/mask.nii.gz
```

The command validates array sizes and physical grids, normalizes CT values,
preserves mask label IDs, and reports `[1,D,H,W]` image and mask tensors.
It saves `outputs/pair_overlay.png` and `outputs/pair_summary.json` locally.
The overlay is an expert annotation, not a predicted segmentation.

By default, the slice with the most positive-label voxels is shown. Use
`--slice 10` to inspect a particular axial slice, including neighboring slices.
Indices start at zero. Verify positive labels represent the intended lesions
from the dataset documentation. A single overlay cannot validate a whole case.

The configurable `--min-hu` and `--max-hu` bounds default to -1000 and 400.
Confirm source HU scaling and spatial units first. No resizing, training, or
inference takes place. Shape/grid checks cannot prove correct case pairing.

Automated tests use synthetic temporary files. If an existing Windows pytest
temporary directory has permission problems, use a fresh directory and disable
the cache in PowerShell:

```powershell
$testTemp = Join-Path $env:TEMP ("capstone-tests-" + [guid]::NewGuid().ToString())
python -m pytest -q -p no:cacheprovider --basetemp "$testTemp"
```

Never commit downloads, source metadata, provenance, or generated overlays.
Keep outputs under the ignored `outputs/` directory. Existing output filenames
are replaced on another run; use `--output-dir outputs/another-check` to retain
separate checks. This utility does not download data or include a real sample.
