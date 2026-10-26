"""Prepare aligned CT/mask arrays without resizing or changing label meaning."""
import numpy as np
import torch
from numpy.typing import NDArray

from .load_ct import validate_pair


def clip_hu(volume: NDArray, min_hu: float = -1000, max_hu: float = 400) -> NDArray:
    """Clip verified HU values to configurable experimental bounds as float32."""
    if not np.isfinite([min_hu, max_hu]).all() or min_hu >= max_hu:
        raise ValueError("HU bounds must be finite and increasing.")
    if volume.ndim != 3 or not all(volume.shape) or not np.isfinite(volume).all():
        raise ValueError("CT must be a nonempty finite 3D array.")
    # Clipping limits extreme values; it does not change image dimensions.
    with np.errstate(over="ignore"):
        clipped = np.clip(volume, min_hu, max_hu).astype(np.float32)
    if not np.isfinite(clipped).all():
        raise ValueError("Clipped CT values exceed the float32 range.")
    return clipped


def normalize_ct(volume: NDArray, min_hu: float = -1000, max_hu: float = 400) -> NDArray:
    """Clip and map fixed HU bounds to [0,1]; constant scans are supported."""
    clipped = clip_hu(volume, min_hu, max_hu)
    # Fixed bounds give the same scale across cases, unlike per-scan min/max.
    # Calculate in float64 to avoid overflow with unusually wide bounds.
    lower, upper = np.float64(min_hu), np.float64(max_hu)
    with np.errstate(over="ignore", invalid="ignore"):
        width = upper - lower
    if not np.isfinite(width):
        raise ValueError("HU range is too wide for normalization.")
    normalized = (clipped.astype(np.float64) - lower) / width
    return np.clip(normalized, 0, 1).astype(np.float32)


def validate_mask(mask: NDArray) -> NDArray:
    """Return int64 labels; reject invalid values without binarizing them.

    Label range matches the loader's int32 range. The int64 output is convenient
    for PyTorch label tensors, not a decision about the future training loss.
    """
    if mask.ndim != 3 or not all(mask.shape) or not np.isfinite(mask).all():
        raise ValueError("Mask must be a nonempty finite 3D array.")
    if (mask < 0).any() or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Mask labels must be nonnegative whole numbers.")
    if np.any(mask.astype(np.float64) > np.iinfo(np.int32).max):
        raise ValueError("Mask labels exceed the supported int32 range.")
    return mask.astype(np.int64)


def to_tensors(volume: NDArray, mask: NDArray, *,
               min_hu: float = -1000, max_hu: float = 400) -> dict[str, torch.Tensor]:
    """Normalize CT and preserve labels as [1,D,H,W] CPU tensors.

    Inputs use the loader's [X,Y,Z] order. Validate physical grids using both
    metadata dictionaries BEFORE calling this array-only function. No resampling
    occurs. Binary training targets must be selected later from confirmed labels.
    """
    validate_pair(volume, mask)
    image = normalize_ct(volume, min_hu, max_hu)
    labels = validate_mask(mask)
    # Reorder [X,Y,Z] to [Z,Y,X], then add one grayscale channel.
    # Copies handle negative strides from flipped arrays and isolate the tensors.
    image_array = image.transpose(2, 1, 0)[None].copy(order="C")
    label_array = labels.transpose(2, 1, 0)[None].copy(order="C")
    return {"image": torch.from_numpy(image_array), "mask": torch.from_numpy(label_array)}
