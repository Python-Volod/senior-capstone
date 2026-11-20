"""Hard negatives must never include selected targets or held-out case IDs."""
import json

import numpy as np
import pytest
import torch

from model.hard_negatives import candidates, read_index, extract_negative
from model.experiment import manifest_digest, sample_patch


def test_candidate_ranking_excludes_foreground_and_partial_edges():
    pred = np.ones((9, 8, 8), bool)
    target = np.zeros_like(pred)
    target[1, 1, 1] = True
    found = candidates(pred, target, 4, 3)
    assert len(found) == 3
    assert all(entry["start_dhw"] != [0, 0, 0] for entry in found)
    assert all(entry["false_positive_voxels"] == 64 for entry in found)
    assert all(entry["start_dhw"][0] < 8 for entry in found)


def test_negative_extraction_refuses_foreground_and_out_of_bounds():
    image = np.zeros((8, 8, 8), np.float32)
    target = np.zeros_like(image, dtype=bool)
    target[0, 0, 0] = True
    with pytest.raises(ValueError, match="foreground"):
        extract_negative(image, target, 4, {"start_dhw": [0, 0, 0]})
    with pytest.raises(ValueError, match="bounds"):
        extract_negative(image, target, 4, {"start_dhw": [6, 0, 0]})
    x, y = extract_negative(image, target, 4, {"start_dhw": [4, 4, 4]})
    assert x.shape == y.shape == (1, 1, 4, 4, 4) and y.sum() == 0


def test_index_rejects_held_out_entries_and_config_changes(tmp_path):
    rows = [{"case_id": "a", "patient_id": "p1", "split": "train"},
            {"case_id": "b", "patient_id": "p2", "split": "val"}]
    config = dict(labels=[1], patch_size=4, min_hu=-1000, max_hu=400, threshold=.5)
    index = {"manifest_digest": manifest_digest(rows), "config": config.copy(), "cases": {"a": []}}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    assert read_index(path, rows, config)["cases"] == {"a": []}
    index["cases"]["b"] = []
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="exactly the training"):
        read_index(path, rows, config)
    index["cases"] = {"a": []}
    index["config"]["threshold"] = .2
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="settings"):
        read_index(path, rows, config)


def test_foreground_jitter_retains_selected_voxel():
    image = np.zeros((20, 20, 20), np.float32)
    for point in [(0, 0, 0), (19, 19, 19), (10, 10, 10)]:
        target = np.zeros_like(image, dtype=bool)
        target[point] = True
        for seed in range(10):
            _, y = sample_patch(image, target, 8, np.random.default_rng(seed), 1., 3)
            assert y.sum() == 1
