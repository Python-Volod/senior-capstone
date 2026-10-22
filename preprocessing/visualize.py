"""Scientific axial views; positive labels are highlighted, not diagnosed."""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from .load_ct import validate_pair


def find_slice_with_lesion(mask: NDArray) -> int:
    """Choose the slice with most positive-label voxels, or middle if empty.

    Assumes positive labels are lesions; verify the dataset label definitions.
    """
    if mask.ndim != 3 or not all(mask.shape):
        raise ValueError("Expected a nonempty 3D mask.")
    # Count marked voxels across X and Y, leaving one total for each Z slice.
    counts = np.count_nonzero(mask > 0, axis=(0, 1))
    return int(np.argmax(counts)) if counts.any() else mask.shape[2] // 2


def _slice(array: NDArray, index: int) -> NDArray:
    if array.ndim != 3 or not 0 <= index < array.shape[2]:
        raise ValueError("Axial slice index is outside the 3D volume.")
    # Matplotlib expects [row,column] = [Y,X], hence the transpose (.T).
    return array[:, :, index].T


def show_slice(volume: NDArray, slice_index: int) -> tuple[Figure, Axes]:
    """Return a figure/axes; caller chooses savefig or plt.show()."""
    fig, ax = plt.subplots()
    ax.imshow(_slice(volume, slice_index), cmap="gray", origin="lower")
    ax.set_title(f"Axial slice {slice_index}")
    ax.set_axis_off()
    return fig, ax


def show_mask(mask: NDArray, slice_index: int) -> tuple[Figure, Axes]:
    """Display discrete expert labels."""
    fig, ax = plt.subplots()
    ax.imshow(_slice(mask, slice_index), cmap="tab20", interpolation="nearest", origin="lower")
    ax.set_title("Expert mask labels")
    ax.set_axis_off()
    return fig, ax


def show_mask_overlay(volume: NDArray, mask: NDArray, slice_index: int | None = None) -> tuple[Figure, Axes]:
    """Overlay positive expert labels with transparency; no prediction involved."""
    validate_pair(volume, mask)
    index = find_slice_with_lesion(mask) if slice_index is None else slice_index
    fig, ax = show_slice(volume, index)
    labels = _slice(mask, index)
    # Hide background label 0 so the CT remains visible beneath the markings.
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap="autumn", alpha=0.45,
              interpolation="nearest", origin="lower")
    return fig, ax
