"""Integration tests use separate synthetic patients, never medical evidence."""
import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from evaluation.metrics import confusion, scores, summarize
from evaluation.compare import main as compare
from model.experiment import read_manifest, infer_volume, main


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        image = np.full((7, 8, 9), -800, np.float32)
        mask = np.zeros(image.shape, np.int16)
        mask[2:5, 2:5, 2:5] = 2
        image[mask == 2] = 50
        for kind, array in (("ct", image), ("mask", mask)):
            nib.save(nib.Nifti1Image(array, np.eye(4)), tmp_path / f"{split}_{kind}.nii.gz")
        rows.append(dict(case_id=f"case{i}", patient_id=f"patient{i}", split=split,
                         ct=f"{split}_ct.nii.gz", mask=f"{split}_mask.nii.gz"))
    path = tmp_path / "manifest.csv"
    write_rows(path, rows)
    return path


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_metrics_hand_calculated():
    result = scores(confusion([1, 1, 0, 0, 0], [1, 0, 1, 1, 0]))
    assert result["tp"] == result["fp"] == result["tn"] == 1
    assert result["fn"] == 2
    assert result["dice"] == pytest.approx(.4)
    assert result["iou"] == pytest.approx(.25)
    assert result["precision"] == .5
    assert result["recall"] == pytest.approx(1/3)
    assert result["accuracy"] == .4


def test_empty_mask_and_macro_conventions():
    empty = scores(confusion([0, 0], [0, 0]))
    missed = scores(confusion([0, 0], [1, 0]))
    assert empty["dice"] == empty["iou"] == 1
    assert empty["precision"] is empty["recall"] is None
    assert missed["dice"] == 0 and missed["recall"] == 0
    summary = summarize([empty, missed])
    assert summary["micro"]["dice"] == 0
    assert summary["macro_case"]["dice"] == .5
    assert summary["macro_defined_cases"]["recall"] == 1


@pytest.mark.parametrize("pred,target", [([.2], [0]), ([1], [1, 0]), ([], [])])
def test_invalid_metrics_rejected(pred, target):
    with pytest.raises(ValueError):
        confusion(pred, target)


def test_patient_leakage_and_duplicate_files(manifest):
    rows = read_manifest(manifest)
    rows[1]["patient_id"] = rows[0]["patient_id"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Patient leakage"):
        read_manifest(manifest)
    rows[1]["patient_id"] = "different"
    rows[1]["ct"] = rows[0]["ct"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_single_patient_cannot_be_evaluation(manifest):
    rows = read_manifest(manifest)
    write_rows(manifest, rows[:1])
    with pytest.raises(ValueError, match="one sample"):
        read_manifest(manifest)


def test_whole_volume_inference_edges_and_no_annotation_input():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return (x - .5) * 10
    image = np.zeros((3, 9, 11), np.float32)
    image[-1, -1, -1] = 1
    pred = infer_volume(Identity(), image, 4, .5)
    assert np.array_equal(pred, image.astype(bool))


def test_training_reload_evaluation_and_comparison(manifest, tmp_path, monkeypatch):
    import model.experiment as experiment
    original_load = experiment.load_case
    calls = []
    def checked_load(row, *args):
        calls.append(row["split"])
        return original_load(row, *args)
    monkeypatch.setattr(experiment, "load_case", checked_load)
    output = tmp_path / "run"
    main(["train", "--manifest", str(manifest), "--output", str(output),
          "--labels", "2", "--epochs", "1", "--patches-per-case", "1",
          "--patch-size", "8", "--base-channels", "1", "--threads", "1"])
    assert set(calls) == {"train", "val"}
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()
    report = output / "test.json"
    main(["evaluate", "--manifest", str(manifest), "--checkpoint", str(output / "best.pt"),
          "--report", str(report), "--threads", "1"])
    assert calls[-1] == "test"
    result = json.loads(report.read_text())
    assert result["aggregation"]["case_count"] == 1
    assert result["cases"][0]["tp"] + result["cases"][0]["fn"] == 27
    # Different thread count must not change CPU evaluation predictions.
    report2 = output / "test2.json"
    main(["evaluate", "--manifest", str(manifest), "--checkpoint", str(output / "best.pt"),
          "--report", str(report2), "--threads", "1"])
    assert json.loads(report2.read_text())["aggregation"] == result["aggregation"]
    # A hypothetical second model exports a full-volume binary prediction.
    target = nib.load(tmp_path / "test_mask.nii.gz")
    prediction = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image((target.get_fdata() == 2).astype(np.uint8), target.affine), prediction)
    predictions = tmp_path / "predictions.csv"
    write_rows(predictions, [{"case_id": "case2", "prediction": prediction.name}])
    comparison = tmp_path / "comparison"
    compare(["--manifest", str(manifest), "--reference-report", str(report),
             "--predictions", str(predictions), "--name", "synthetic_oracle",
             "--output", str(comparison)])
    assert json.loads((comparison / "report.json").read_text())["aggregation"]["micro"]["dice"] == 1
    assert (comparison / "comparison.csv").is_file()
    # Explicitly reject the wrong case coverage.
    write_rows(predictions, [{"case_id": "case0", "prediction": prediction.name}])
    with pytest.raises(SystemExit):
        compare(["--manifest", str(manifest), "--reference-report", str(report),
                 "--predictions", str(predictions), "--name", "other", "--output", str(tmp_path / "bad")])
    # Prevent silently reassigning test cases after training.
    rows = read_manifest(manifest)
    rows[1]["split"], rows[2]["split"] = rows[2]["split"], rows[1]["split"]
    write_rows(manifest, rows)
    with pytest.raises(SystemExit):
        main(["evaluate", "--manifest", str(manifest), "--checkpoint", str(output / "best.pt"),
              "--report", str(tmp_path / "bad.json")])


def test_unselected_labels_rejected(manifest):
    from model.experiment import load_case
    with pytest.raises(ValueError, match="Unselected"):
        load_case(read_manifest(manifest)[0], [1], -1000, 400)


def test_resume_matches_uninterrupted_training(manifest, tmp_path):
    """Restore optimizer and RNGs, not just weights, and retain the earlier best."""
    common = ["--manifest", str(manifest), "--labels", "2", "--patches-per-case", "2",
              "--patch-size", "8", "--base-channels", "1", "--threads", "1"]
    uninterrupted, first, resumed = (tmp_path / name for name in ("full", "first", "resumed"))
    main(["train", *common, "--output", str(uninterrupted), "--epochs", "2"])
    main(["train", *common, "--output", str(first), "--epochs", "1"])
    main(["resume", "--manifest", str(manifest), "--checkpoint", str(first / "last.pt"),
          "--output", str(resumed), "--epochs", "1", "--threads", "1"])
    full = torch.load(uninterrupted / "last.pt", weights_only=True)
    continued = torch.load(resumed / "last.pt", weights_only=True)
    assert full["epoch"] == continued["epoch"] == 2
    assert all(torch.equal(v, continued["model_state"][k]) for k, v in full["model_state"].items())
    assert full["numpy_rng_state"] == continued["numpy_rng_state"]
    assert full["best_validation_macro_dice"] == continued["best_validation_macro_dice"]
    assert (resumed / "best.pt").is_file()
    history = json.loads((resumed / "history.json").read_text())
    assert [r["epoch"] for r in history] == [1, 2]
    assert history[-1]["train_patch_bce"] == json.loads((uninterrupted / "history.json").read_text())[-1]["train_patch_bce"]
    # A changed patient split cannot be used for continuation.
    rows = read_manifest(manifest)
    rows[1]["split"], rows[2]["split"] = rows[2]["split"], rows[1]["split"]
    write_rows(manifest, rows)
    with pytest.raises(SystemExit):
        main(["resume", "--manifest", str(manifest), "--checkpoint", str(first / "last.pt"),
              "--output", str(tmp_path / "leak"), "--epochs", "1", "--threads", "1"])


def test_foreground_loss_and_sampler():
    from model.losses import segmentation_loss
    from model.experiment import sample_patch
    target = torch.zeros(1, 1, 8, 8, 8)
    target[:, :, 3:5, 3:5, 3:5] = 1
    bad = torch.full_like(target, -8., requires_grad=True)
    good = torch.where(target.bool(), 8., -8.)
    loss = segmentation_loss(bad, target, mode="dice_bce", positive_weight=20)
    assert loss > segmentation_loss(good, target, mode="dice_bce", positive_weight=20)
    loss.backward()
    assert torch.isfinite(bad.grad).all()
    assert (bad.grad[target.bool()] < 0).all()
    empty = segmentation_loss(torch.full_like(target, -8.), torch.zeros_like(target), mode="dice_bce")
    assert torch.isfinite(empty)
    image = np.zeros((20, 20, 20), np.float32)
    labels = np.zeros_like(image, dtype=bool)
    labels[0, 0, -1] = True
    for seed in range(5):
        _, sampled = sample_patch(image, labels, 8, np.random.default_rng(seed), 1.)
        assert sampled.sum() == 1


def test_batched_inference_matches_single_tile():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return (x - .5) * 10
    image = np.random.default_rng(0).random((7, 9, 11)).astype(np.float32)
    expected = infer_volume(Identity(), image, 4, .5, 1)
    assert np.array_equal(expected, infer_volume(Identity(), image, 4, .5, 5))


def test_new_loss_checkpoint_can_resume(manifest, tmp_path):
    first, second = tmp_path / "weighted", tmp_path / "continued"
    main(["train", "--manifest", str(manifest), "--output", str(first), "--labels", "2",
          "--loss", "dice_bce", "--positive-weight", "20", "--foreground-probability", ".75",
          "--inference-batch-size", "3", "--epochs", "1", "--patches-per-case", "2",
          "--patch-size", "8", "--base-channels", "1", "--threads", "1"])
    main(["resume", "--manifest", str(manifest), "--checkpoint", str(first / "last.pt"),
          "--output", str(second), "--epochs", "1", "--threads", "1"])
    state = torch.load(second / "last.pt", weights_only=True)
    assert state["config"]["loss"] == "dice_bce" and state["epoch"] == 2
    assert state["config"]["foreground_probability"] == .75
    assert state["config"]["positive_weight"] == 20
    assert state["config"]["inference_batch_size"] == 3


def test_weight_initialization_is_a_new_experiment(manifest, tmp_path):
    base, fine = tmp_path / "base", tmp_path / "fine"
    common = ["--manifest", str(manifest), "--labels", "2", "--epochs", "1",
              "--patches-per-case", "1", "--patch-size", "8", "--base-channels", "1", "--threads", "1"]
    main(["train", *common, "--output", str(base)])
    main(["train", *common, "--output", str(fine), "--initialize", str(base / "last.pt"),
          "--loss", "dice_bce", "--positive-weight", "20"])
    state = torch.load(fine / "last.pt", weights_only=True)
    assert state["epoch"] == 1 and state["config"]["initialized_from_epoch"] == 1
    assert "initialized_from_sha256" in state["config"]
    assert state["config"]["loss"] == "dice_bce"
    assert len(json.loads((fine / "history.json").read_text())) == 1
    # A different target cannot silently reuse the binary checkpoint.
    with pytest.raises(SystemExit):
        main(["train", "--manifest", str(manifest), "--labels", "1", "--output", str(tmp_path / "bad_init"),
              "--initialize", str(base / "last.pt"), "--base-channels", "1", "--threads", "1"])
    # Export uses CT only, and preserves the original physical grid and units.
    from model.predict import main as predict
    ct_path = tmp_path / "val_ct.nii.gz"
    original = nib.load(ct_path)
    original.header.set_xyzt_units("mm")
    nib.save(original, ct_path)
    prediction = tmp_path / "exported.nii.gz"
    predict(["--checkpoint", str(fine / "best.pt"), "--ct", str(ct_path),
             "--output", str(prediction), "--threads", "1"])
    exported = nib.load(prediction)
    assert exported.shape == original.shape
    assert np.allclose(exported.affine, original.affine)
    assert exported.header.get_xyzt_units()[0] == "mm"
    assert set(np.unique(exported.get_fdata())) <= {0, 1}
    assert json.loads((tmp_path / "exported.nii.gz.json").read_text())["kind"] == "model_prediction"
    with pytest.raises(SystemExit):
        predict(["--checkpoint", str(fine / "best.pt"), "--ct", str(ct_path),
                 "--output", str(prediction), "--threads", "1"])


def test_hard_negative_training_and_resume(manifest, tmp_path):
    from model.experiment import manifest_digest
    rows = read_manifest(manifest)
    settings = dict(labels=[2], patch_size=4, min_hu=-1000, max_hu=400, threshold=.5)
    index = {"manifest_digest": manifest_digest(rows), "config": settings,
             "source_checkpoint_sha256": "synthetic-test-pool",
             "cases": {"case0": [{"start_dhw": [5, 0, 0], "false_positive_voxels": 16}]}}
    path = tmp_path / "hard.json"
    path.write_text(json.dumps(index))
    first, second = tmp_path / "hard_run", tmp_path / "hard_resume"
    main(["train", "--manifest", str(manifest), "--output", str(first), "--labels", "2",
          "--patch-size", "4", "--base-channels", "1", "--epochs", "1", "--patches-per-case", "8",
          "--foreground-probability", "0.25", "--hard-negative-probability", "0.75",
          "--foreground-jitter", "1", "--hard-negatives", str(path), "--threads", "1"])
    history = json.loads((first / "history.json").read_text())
    assert history[0]["hard_negative_samples"] > 0
    main(["resume", "--manifest", str(manifest), "--checkpoint", str(first / "last.pt"),
          "--output", str(second), "--epochs", "1", "--threads", "1"])
    state = torch.load(second / "last.pt", weights_only=True)
    assert state["epoch"] == 2 and state["config"]["hard_negative_probability"] == .75
    index["cases"]["case0"][0]["false_positive_voxels"] = 15
    path.write_text(json.dumps(index))
    with pytest.raises(SystemExit):
        main(["resume", "--manifest", str(manifest), "--checkpoint", str(first / "last.pt"),
              "--output", str(tmp_path / "tampered"), "--epochs", "1", "--threads", "1"])
