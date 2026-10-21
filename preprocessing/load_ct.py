"""Load scalar 3D NIfTI images in canonical RAS+ orientation.

Arrays use [X, Y, Z]; axis 2 indexes axial slices. Metadata contains
the canonical affine (voxel indices to world millimeters) and voxel spacing.
DICOM series/SEG conversion requires a separate, dataset-specific workflow.
"""
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def _load(path: str | Path, *, dtype: type = np.float32) -> tuple[NDArray, dict[str, Any]]:
    """Shared file checks and reading used by both CT and mask loaders."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("NIfTI input file does not exist; check the supplied path.")
    if not source.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Expected .nii or .nii.gz; DICOM is not supported yet.")
    try:
        # Put scans into a consistent axis order without resizing their voxels.
        image = nib.as_closest_canonical(nib.load(source))
        # get_fdata also applies any scaling recorded in the NIfTI header.
        array = image.get_fdata(dtype=dtype)
    except Exception:
        raise ValueError("Could not read the NIfTI input; verify its format and integrity.") from None
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError("Expected a nonempty scalar 3D volume.")
    if not np.isfinite(array).all():
        raise ValueError("Volume contains nonfinite values.")
    # The affine is a map from array positions to positions in physical space.
    affine = image.affine.copy()
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Volume has invalid spatial geometry.")
    return array, {"affine": affine, "spacing": tuple(image.header.get_zooms()[:3]),
                   "shape": array.shape, "orientation": "RAS+"}


def load_volume(path: str | Path, *, return_metadata: bool = False) -> NDArray | tuple[NDArray, dict[str, Any]]:
    """Load CT values, applying NIfTI scaling; confirm HU units with the provider."""
    array, metadata = _load(path)
    return (array, metadata) if return_metadata else array


def load_mask(path: str | Path, *, return_metadata: bool = False) -> NDArray | tuple[NDArray, dict[str, Any]]:
    """Load nonnegative integer labels without assuming their lesion meaning."""
    # float64 preserves every int32 label exactly while checking scaled values.
    # float32 could round a large label before we check whether it is valid.
    array, metadata = _load(path, dtype=np.float64)
    if (array < 0).any() or not np.equal(array, np.floor(array)).all():
        raise ValueError("Mask must contain nonnegative integer labels.")
    if array.max() > np.iinfo(np.int32).max:
        raise ValueError("Mask labels exceed the supported int32 range.")
    array = array.astype(np.int32)
    return (array, metadata) if return_metadata else array


def validate_pair(volume: NDArray, mask: NDArray,
                  volume_metadata: dict | None = None,
                  mask_metadata: dict | None = None) -> None:
    """Check dimensions and, when supplied, physical grids (1e-4 tolerance).

    Shape alone cannot establish physical alignment. Supply both metadata
    dictionaries for real medical data; visual verification is still needed.
    """
    if volume.ndim != 3 or volume.shape != mask.shape or not all(volume.shape):
        raise ValueError("CT and mask must have identical nonempty 3D dimensions.")
    if (volume_metadata is None) != (mask_metadata is None):
        raise ValueError("Supply metadata for both CT and mask, or neither.")
    # Two equally sized arrays can still refer to different physical locations.
    if volume_metadata is not None and not np.allclose(
        volume_metadata["affine"], mask_metadata["affine"], rtol=0, atol=1e-4
    ):
        raise ValueError("CT and mask physical grids differ; registration/resampling needs review.")
