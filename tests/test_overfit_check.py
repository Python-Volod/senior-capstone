"""Training-only diagnosis: sampling, measurable learning and split isolation."""
import numpy as np
import pytest
import torch

from model.overfit_check import fixed_patches, fit_fixed, main


def test_fixed_patches_are_repeatable_and_include_negative():
    image = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
    target = np.zeros(image.shape, bool)
    target[1:4, 1:4, 1:4] = True
    x, y = fixed_patches(image, target, 8, 17)
    other_x, other_y = fixed_patches(image, target, 8, 17)
    assert torch.equal(x, other_x) and torch.equal(y, other_y)
    assert y[0].sum() > 0 and y[1].sum() > 0 and y[2].sum() == 0
    assert x.shape == y.shape == (3, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="foreground"):
        fixed_patches(image, np.zeros_like(target), 8, 17)


def test_diagnostic_can_measure_learning_on_fixed_inputs():
    # A small independently understandable task checks the diagnostic loop.
    x = torch.zeros(3, 1, 4, 4, 4)
    x[:2, :, 1:3, 1:3, 1:3] = 1
    y = x.clone()
    network = torch.nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        network.weight.fill_(0)
        network.bias.fill_(0)
    before, after, step, history = fit_fixed(network, x, y, steps=100, lr=.2,
                                           positive_weight=1., dice_weight=.5, report_every=10)
    assert after["min_positive_patch_dice"] > before["min_positive_patch_dice"]
    assert after["min_positive_patch_dice"] >= .9 and after["negative_patch_fp"] == 0
    assert step > 0 and history[-1]["step"] <= 100


def test_diagnostic_rejects_validation_before_checkpoint_loading(tmp_path):
    import csv
    rows = []
    for i, split in enumerate(["train", "val", "test"]):
        for kind in ("ct", "mask"):
            (tmp_path / f"{split}_{kind}.nii.gz").touch()
        rows.append(dict(case_id=f"c{i}", patient_id=f"p{i}", split=split,
                         ct=f"{split}_ct.nii.gz", mask=f"{split}_mask.nii.gz"))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest), "--case-id", "c1", "--initialize", "missing.pt",
              "--output", str(tmp_path / "should_not_exist")])
    assert not (tmp_path / "should_not_exist").exists()
