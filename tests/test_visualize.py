"""Synthetic visualization checks; no medical results or real scans."""
import matplotlib

# Draw without opening windows so tests can run on different computers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from preprocessing.visualize import (
    find_slice_with_lesion,
    show_mask,
    show_mask_overlay,
    show_slice,
)


@pytest.fixture
def pair():
    """A tiny volume with label 2 on one slice makes selection predictable."""
    ct = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    mask = np.zeros(ct.shape, dtype=np.int32)
    mask[1:3, 1:4, 2] = 2
    return ct, mask


@pytest.fixture(autouse=True)
def close_figures():
    """Release figures after every test, even if an assertion fails."""
    yield
    plt.close("all")


def test_choose_marked_or_middle_slice(pair):
    _, mask = pair
    assert find_slice_with_lesion(mask) == 2
    assert find_slice_with_lesion(np.zeros_like(mask)) == 3


def test_slice_axis_order(pair):
    """The displayed slice must preserve values while switching X/Y to rows/columns."""
    ct, _ = pair
    _, ax = show_slice(ct, 2)
    np.testing.assert_array_equal(ax.images[0].get_array(), ct[:, :, 2].T)


def test_mask_labels_are_not_blended(pair):
    _, mask = pair
    _, ax = show_mask(mask, 2)
    np.testing.assert_array_equal(ax.images[0].get_array(), mask[:, :, 2].T)
    assert ax.images[0].get_interpolation() == "nearest"


def test_overlay_hides_background_and_saves(pair, tmp_path):
    """The mask covers marked voxels only; background stays transparent."""
    ct, mask = pair
    fig, ax = show_mask_overlay(ct, mask)
    assert len(ax.images) == 2
    overlay = ax.images[1].get_array()
    np.testing.assert_array_equal(np.ma.getmaskarray(overlay), mask[:, :, 2].T == 0)
    assert ax.images[1].get_alpha() == .45
    destination = tmp_path / "overlay.png"
    fig.savefig(destination)
    assert destination.stat().st_size > 0


def test_explicit_slice(pair):
    ct, mask = pair
    _, ax = show_mask_overlay(ct, mask, slice_index=1)
    np.testing.assert_array_equal(ax.images[0].get_array(), ct[:, :, 1].T)


def test_invalid_inputs(pair):
    ct, mask = pair
    with pytest.raises(ValueError, match="dimensions"):
        show_mask_overlay(ct, mask[:, :, :-1])
    with pytest.raises(ValueError, match="slice index"):
        show_slice(ct, 6)
    with pytest.raises(ValueError, match="slice index"):
        show_slice(ct, -1)
    with pytest.raises(ValueError, match="nonempty 3D"):
        find_slice_with_lesion(np.empty((0, 2, 3)))
