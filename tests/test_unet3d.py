"""Model mechanics on synthetic patches; no medical performance claims."""
import numpy as np
import pytest
import torch

from model.unet3d import UNet3D
from model.smoke_test import extract_patch, main


@pytest.mark.parametrize("shape", [(1, 1, 8, 12, 16), (1, 1, 9, 11, 13)])
def test_forward_backward_and_update(shape):
    """The output grid is unchanged and an optimizer can update parameters."""
    torch.manual_seed(0)
    network = UNet3D(base_channels=2)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    before = network.output.weight.detach().clone()
    logits = network(torch.rand(shape))
    assert logits.shape == shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters())
    optimizer.step()
    assert not torch.equal(before, network.output.weight)


def test_patch_contains_selected_label():
    ct = np.zeros((12, 13, 14), dtype=np.float32)
    mask = np.zeros(ct.shape, dtype=np.int32)
    mask[0, 0, 13] = 2
    image_patch, mask_patch = extract_patch(ct, mask, 2, 8)
    assert image_patch.shape == mask_patch.shape == (8, 8, 8)
    assert 2 in mask_patch
    with pytest.raises(ValueError, match="absent"):
        extract_patch(ct, mask, 3, 8)
    with pytest.raises(ValueError, match="patch size"):
        extract_patch(ct, mask, 2, 20)


def test_model_input_validation():
    with pytest.raises(ValueError, match="Expected"):
        UNet3D()(torch.zeros(1, 1, 3, 4, 4))


def test_smoke_command(tmp_path, monkeypatch, capsys):
    """Run one complete synthetic file-to-model training update."""
    import nibabel as nib
    ct = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(ct)
    mask[3:5, 3:5, 3:5] = 2
    paths = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    for path, array in zip(paths, (ct, mask)):
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    monkeypatch.setattr("sys.argv", ["smoke_test", "--ct", str(paths[0]), "--mask", str(paths[1]),
                                    "--lesion-label", "2", "--patch-size", "8", "--steps", "1"])
    previous_threads = torch.get_num_threads()
    try:
        main()
    finally:
        torch.set_num_threads(previous_threads)
    assert "optimizer updates completed" in capsys.readouterr().out
