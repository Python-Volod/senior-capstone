"""Synthetic preprocessing checks; these do not measure medical performance."""
import numpy as np
import pytest
import torch
import nibabel as nib

from preprocessing.preprocess import clip_hu, normalize_ct, to_tensors, validate_mask
from preprocessing.load_ct import load_volume, load_mask, validate_pair


def test_clipping_and_normalization():
    """Bounds are configurable; clipping and normalization leave input unchanged."""
    ct = np.array([-1200, -1000, -300, 400, 800, 0], dtype=np.float64).reshape(1, 2, 3)
    original = ct.copy()
    clipped = clip_hu(ct)
    assert clipped.dtype == np.float32
    np.testing.assert_array_equal(clipped.ravel(), [-1000, -1000, -300, 400, 400, 0])
    result = normalize_ct(ct)
    np.testing.assert_allclose(result.ravel()[:5], [0, 0, .5, 1, 1])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(ct, original)
    np.testing.assert_allclose(normalize_ct(np.zeros((2, 2, 2)), -10, 10), .5)


@pytest.mark.parametrize("bounds", [(10, 10), (10, -10), (np.nan, 400), (-1000, np.inf)])
def test_invalid_hu_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        clip_hu(np.zeros((2, 2, 2)), *bounds)


@pytest.mark.parametrize("value", [-1., .5, np.nan, np.inf, 2_147_483_648.])
def test_invalid_mask_labels(value):
    with pytest.raises(ValueError):
        validate_mask(np.full((2, 2, 2), value))


def test_tensor_layout_and_label_preservation():
    """Distinct axes expose ordering errors; label 2 must remain label 2."""
    ct = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros(ct.shape, dtype=np.int32)
    mask[1, 2, 3] = 2
    # Flipped views have negative strides, which torch.from_numpy cannot accept directly.
    result = to_tensors(ct[::-1], mask[::-1])
    assert result["image"].shape == result["mask"].shape == (1, 4, 3, 2)
    assert result["image"].dtype == torch.float32
    assert result["mask"].dtype == torch.int64
    assert set(result["mask"].unique().tolist()) == {0, 2}
    assert result["mask"][0, 3, 2, 0] == 2
    np.testing.assert_allclose(result["image"][0].numpy(), normalize_ct(ct[::-1]).transpose(2, 1, 0))
    np.testing.assert_array_equal(mask[1, 2, 3], 2)


def test_invalid_arrays():
    with pytest.raises(ValueError, match="dimensions"):
        to_tensors(np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))
    for array in (np.empty((0, 2, 3)), np.zeros((2, 3)), np.full((2, 3, 4), np.nan)):
        with pytest.raises(ValueError):
            normalize_ct(array)
        with pytest.raises(ValueError):
            validate_mask(array)


def test_nifti_pair_to_tensors(tmp_path):
    """Exercise the file-to-tensor path without needing a medical dataset."""
    ct = np.linspace(-1200, 800, 24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros(ct.shape, dtype=np.int16)
    mask[1, 2, 3] = 3
    paths = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    for path, array in zip(paths, (ct, mask)):
        nib.save(nib.Nifti1Image(array, np.diag([1., 2., 3., 1.])), path)
    loaded_ct, ct_info = load_volume(paths[0], return_metadata=True)
    loaded_mask, mask_info = load_mask(paths[1], return_metadata=True)
    validate_pair(loaded_ct, loaded_mask, ct_info, mask_info)
    sample = to_tensors(loaded_ct, loaded_mask)
    assert sample["image"].shape == sample["mask"].shape == (1, 4, 3, 2)
    assert sample["image"].min() == 0 and sample["image"].max() == 1
    assert sample["mask"][0, 3, 2, 1] == 3
