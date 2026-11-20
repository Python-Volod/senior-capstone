"""Mine annotation-free false-positive tiles from TRAINING patients only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


def candidates(prediction, target, size, limit):
    """Rank full tiles by FP count; exclude any tile containing target voxels."""
    if prediction.shape != target.shape or prediction.ndim != 3 or size < 4 or limit < 1:
        raise ValueError("Invalid hard-negative arrays, size or limit.")
    found = []
    for start in itertools.product(*(range(0, n - size + 1, size) for n in target.shape)):
        region = tuple(slice(s, s + size) for s in start)
        if not target[region].any():
            count = int(np.count_nonzero(prediction[region]))
            if count:
                found.append({"start_dhw": list(start), "false_positive_voxels": count})
    return sorted(found, key=lambda r: (-r["false_positive_voxels"], r["start_dhw"]))[:limit]


def read_index(path, rows, config):
    """Require the original cohort/settings and forbid validation/test entries."""
    from .experiment import manifest_digest
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if index["manifest_digest"] != manifest_digest(rows):
        raise ValueError("Hard-negative index has different patient splits.")
    for key in ("labels", "patch_size", "min_hu", "max_hu", "threshold"):
        if index["config"][key] != config[key]:
            raise ValueError("Hard-negative index settings differ from training.")
    training_ids = {r["case_id"] for r in rows if r["split"] == "train"}
    if set(index["cases"]) != training_ids:
        raise ValueError("Hard-negative index must contain exactly the training cases.")
    for entries in index["cases"].values():
        for entry in entries:
            start = entry["start_dhw"]
            if len(start) != 3 or any(type(v) is not int or v < 0 for v in start):
                raise ValueError("Invalid hard-negative start coordinates.")
    return index


def extract_negative(image, target, size, entry):
    start = entry["start_dhw"]
    region = tuple(slice(s, s + size) for s in start)
    if any(s + size > n for s, n in zip(start, image.shape)) or target[region].any():
        raise ValueError("Hard-negative tile is out of bounds or contains foreground.")
    return (torch.from_numpy(image[region].copy())[None, None],
            torch.from_numpy(target[region].astype(np.float32))[None, None])


def main(argv=None):
    from .experiment import read_manifest, manifest_digest, load_case, infer_volume, write_json
    from .unet3d import UNet3D
    from evaluation.metrics import confusion, scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-case", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1 or args.limit_per_case < 1:
            raise ValueError("Thread and tile limits must be positive.")
        rows = read_manifest(args.manifest)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = state["config"]
        if config["manifest_digest"] != manifest_digest(rows):
            raise ValueError("Checkpoint and manifest splits differ.")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose a new mining output directory.")
        args.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(args.threads)
        network = UNet3D(config["base_channels"])
        network.load_state_dict(state["model_state"])
        index = {"purpose": "TRAINING-ONLY fixed hard-negative pool", "config": config,
                 "manifest_digest": manifest_digest(rows), "limit_per_case": args.limit_per_case,
                 "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "cases": {}}
        audit = []
        for row in rows:
            if row["split"] != "train":
                continue
            image, target = load_case(row, config["labels"], config["min_hu"], config["max_hu"])
            pred = infer_volume(network, image, config["patch_size"], config["threshold"],
                                config.get("inference_batch_size", 1))
            entries = candidates(pred, target, config["patch_size"], args.limit_per_case)
            index["cases"][row["case_id"]] = entries
            audit.append({"case_id": row["case_id"], "split": "train", **scores(confusion(pred, target)),
                          "mined_tiles": len(entries)})
            print(f"Training scan {len(audit)}: {int(np.count_nonzero(pred & ~target))} FP voxels; {len(entries)} hard-negative tiles", flush=True)
        write_json(args.output / "index.json", index)
        write_json(args.output / "training_audit.json", audit)
        print("Saved training-only index.json and training_audit.json. No validation/test scans loaded.")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
