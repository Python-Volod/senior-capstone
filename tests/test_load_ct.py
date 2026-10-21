"""Loader checks using tiny artificial images, not real medical results."""
import nibabel as nib
import numpy as np
import pytest

from preprocessing.load_ct import load_mask, load_volume, validate_pair


@pytest.fixture
def pair(tmp_path):
    """Write a fake CT and expert mask; pytest removes them after testing."""
    ct = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    mask = np.zeros(ct.shape, dtype=np.int16)
    mask[1:3, 1:4, 2] = 2
    paths = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    # Each diagonal value sets the voxel spacing along one spatial axis.
    affine = np.diag([1., 2., 3., 1.])
    for path, array in zip(paths, (ct, mask)):
        nib.save(nib.Nifti1Image(array, affine), path)
    return paths, ct, mask


def test_load_pair(pair):
    """Both image values and geometry should survive loading."""
    paths, expected_ct, expected_mask = pair
    ct, ct_info = load_volume(paths[0], return_metadata=True)
    mask, mask_info = load_mask(paths[1], return_metadata=True)
    np.testing.assert_array_equal(ct, expected_ct)
    np.testing.assert_array_equal(mask, expected_mask)
    assert ct.dtype == np.float32
    assert mask.dtype == np.int32
    assert ct_info["spacing"] == (1, 2, 3)
    validate_pair(ct, mask, ct_info, mask_info)


def test_misaligned_pair(pair):
    """Equal sizes do not guarantee equal physical positions."""
    paths, _, _ = pair
    ct, ct_info = load_volume(paths[0], return_metadata=True)
    mask, mask_info = load_mask(paths[1], return_metadata=True)
    # Move the mask by five units while leaving its dimensions unchanged.
    mask_info["affine"][0, 3] += 5
    with pytest.raises(ValueError, match="physical grids"):
        validate_pair(ct, mask, ct_info, mask_info)
    with pytest.raises(ValueError, match="dimensions"):
        validate_pair(ct, mask[:, :, :-1])
    with pytest.raises(ValueError, match="both"):
        validate_pair(ct, mask, ct_info)


def test_missing_and_unsupported_files(tmp_path):
    """Bad paths and unsupported formats should produce readable errors."""
    with pytest.raises(FileNotFoundError):
        load_volume(tmp_path / "missing.nii")
    unsupported = tmp_path / "ct.dcm"
    unsupported.touch()
    with pytest.raises(ValueError, match="DICOM"):
        load_volume(unsupported)


@pytest.mark.parametrize("value", [-1., .5, 2_147_483_648.])
def test_invalid_mask_labels(tmp_path, value):
    """Labels must be nonnegative whole numbers within the int32 range."""
    path = tmp_path / "mask.nii"
    nib.save(nib.Nifti1Image(np.full((2, 2, 2), value), np.eye(4)), path)
    with pytest.raises(ValueError):
        load_mask(path)


def test_large_valid_label(tmp_path):
    """Reading a mask must not round large integer label IDs."""
    labels = np.full((2, 2, 2), 16_777_217, dtype=np.int32)
    path = tmp_path / "mask.nii"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
    np.testing.assert_array_equal(load_mask(path), labels)


def test_reorientation(tmp_path):
    """A reversed axis must be corrected consistently in both images."""
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([-1., 2., 3., 1.])
    paths = tmp_path / "ct.nii", tmp_path / "mask.nii"
    for path in paths:
        nib.save(nib.Nifti1Image(array, affine), path)
    ct, ct_info = load_volume(paths[0], return_metadata=True)
    mask, mask_info = load_mask(paths[1], return_metadata=True)
    validate_pair(ct, mask, ct_info, mask_info)
    np.testing.assert_array_equal(ct, array[::-1])
    assert nib.aff2axcodes(ct_info["affine"]) == ("R", "A", "S")


def test_empty_pair():
    """An empty volume cannot be used for inspection or modeling."""
    with pytest.raises(ValueError, match="nonempty"):
        validate_pair(np.empty((0, 2, 3)), np.empty((0, 2, 3)))
