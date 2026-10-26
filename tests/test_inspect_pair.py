"""Exercise the inspection command with synthetic files, never real scans."""
import json

import nibabel as nib
import numpy as np
import pytest

from preprocessing.inspect_pair import main


@pytest.fixture
def arguments(tmp_path):
    """Create a tiny matching pair and a temporary destination for generated files."""
    ct = np.linspace(-1200, 800, 120, dtype=np.float32).reshape(4, 5, 6)
    mask = np.zeros(ct.shape, dtype=np.int16)
    mask[1:3, 1:4, 2] = 2
    paths = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    for path, array in zip(paths, (ct, mask)):
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    output = tmp_path / "outputs"
    return ["inspect_pair", "--ct", str(paths[0]), "--mask", str(paths[1]),
            "--output-dir", str(output)], output


def test_inspection_outputs(arguments, monkeypatch, capsys):
    """Check the complete file-to-overlay path and the reported tensor layout."""
    argv, output = arguments
    monkeypatch.setattr("sys.argv", argv)
    main()
    summary = json.loads((output / "pair_summary.json").read_text(encoding="utf-8"))
    assert (output / "pair_overlay.png").stat().st_size > 0
    assert summary["selected_axial_slice"] == 2
    assert summary["image_tensor_shape"] == summary["mask_tensor_shape"] == [1, 6, 5, 4]
    assert set(summary["mask_label_counts"]) == {"0", "2"}
    assert summary["normalized_range"] == [0, 1]
    assert str(output.parent) not in capsys.readouterr().out


def test_explicit_slice(arguments, monkeypatch):
    argv, output = arguments
    monkeypatch.setattr("sys.argv", argv + ["--slice", "1"])
    main()
    summary = json.loads((output / "pair_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_axial_slice"] == 1


def test_invalid_slice(arguments, monkeypatch, capsys):
    argv, output = arguments
    monkeypatch.setattr("sys.argv", argv + ["--slice", "6"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "outside the volume" in capsys.readouterr().err
    assert not output.exists()
