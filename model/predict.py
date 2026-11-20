"""Export a trained checkpoint's whole-CT binary mask without expert labels."""
import argparse
import hashlib
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from preprocessing.load_ct import load_volume
from preprocessing.preprocess import normalize_ct
from .experiment import infer_volume, write_json
from .unet3d import UNet3D


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("Thread count must be positive.")
        if not args.output.name.endswith((".nii", ".nii.gz")):
            raise ValueError("Output must be .nii or .nii.gz.")
        summary_path = Path(str(args.output) + ".json")
        if args.output.exists() or summary_path.exists():
            raise ValueError("Prediction output already exists; choose a new filename.")
        torch.set_num_threads(args.threads)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = state["config"]
        ct, metadata = load_volume(args.ct, return_metadata=True)
        image = normalize_ct(ct, config["min_hu"], config["max_hu"]).transpose(2, 1, 0).copy()
        network = UNet3D(config["base_channels"])
        network.load_state_dict(state["model_state"])
        start = time.perf_counter()
        prediction = infer_volume(network, image, config["patch_size"], config["threshold"],
                                  config.get("inference_batch_size", 1))
        seconds = time.perf_counter() - start
        # Convert back to canonical XYZ and preserve source spatial units. A new
        # binary image must not inherit CT intensity scaling/calibration values.
        header = nib.load(args.ct).header.copy()
        header.set_data_dtype(np.uint8)
        header["cal_min"], header["cal_max"] = 0, 1
        exported = nib.Nifti1Image(prediction.transpose(2, 1, 0).astype(np.uint8), metadata["affine"], header)
        exported.header.set_slope_inter(1., 0.)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(exported, args.output)
        write_json(summary_path, {"kind": "model_prediction", "output_orientation": "canonical RAS+",
                                  "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                  "checkpoint_epoch": state["epoch"], "threshold": config["threshold"],
                                  "foreground_voxels": int(prediction.sum()), "shape_xyz": list(ct.shape),
                                  "network_inference_seconds": seconds, "threads": args.threads,
                                  "inference_batch_size": config.get("inference_batch_size", 1),
                                  "performance": "Not measured by prediction command; no expert mask supplied.",
                                  "clinical_validation": False})
        print("Saved model prediction and JSON summary. This command does not calculate accuracy.")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
