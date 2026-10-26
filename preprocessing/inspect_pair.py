"""Inspect one expert CT/mask pair; no model or diagnosis is involved."""
import argparse
import json
from pathlib import Path

import matplotlib

# Save figures without depending on a desktop window or editor integration.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .load_ct import load_volume, load_mask, validate_pair
from .preprocess import to_tensors
from .visualize import find_slice_with_lesion, show_mask_overlay


def main() -> None:
    """Read user-supplied paths and save a local overlay and numerical summary."""
    parser = argparse.ArgumentParser(description="Inspect an aligned CT and expert mask, without inference.")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--slice", type=int, dest="slice_index", help="Axial index; default selects most marked voxels.")
    parser.add_argument("--min-hu", type=float, default=-1000)
    parser.add_argument("--max-hu", type=float, default=400)
    args = parser.parse_args()

    # Check physical alignment before preprocessing; matching sizes alone are insufficient.
    try:
        ct, ct_info = load_volume(args.ct, return_metadata=True)
        mask, mask_info = load_mask(args.mask, return_metadata=True)
        validate_pair(ct, mask, ct_info, mask_info)
        index = find_slice_with_lesion(mask) if args.slice_index is None else args.slice_index
        if not 0 <= index < ct.shape[2]:
            raise ValueError("Axial slice index is outside the volume.")
        tensors = to_tensors(ct, mask, min_hu=args.min_hu, max_hu=args.max_hu)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # Summaries deliberately omit input paths, headers, and source case identifiers.
    labels, counts = np.unique(mask, return_counts=True)
    summary = {
        "shape_xyz": list(ct.shape),
        "spacing_header_units": [float(value) for value in ct_info["spacing"]],
        "ct_range": [float(ct.min()), float(ct.max())],
        "mask_label_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "selected_axial_slice": index,
        "image_tensor_shape": list(tensors["image"].shape),
        "mask_tensor_shape": list(tensors["mask"].shape),
        "image_dtype": str(tensors["image"].dtype),
        "mask_dtype": str(tensors["mask"].dtype),
        "normalized_range": [float(tensors["image"].min()), float(tensors["image"].max())],
        "grid_validation": "passed",
        "model_performance": "N/A; no inference",
        "intensity_note": "Verify source HU scaling before interpreting normalized values.",
    }

    # Generated medical images stay local; the default outputs/ folder is ignored.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = show_mask_overlay(ct, mask, index)
    try:
        fig.savefig(args.output_dir / "pair_overlay.png", dpi=150)
    finally:
        plt.close(fig)
    (args.output_dir / "pair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved pair_overlay.png and pair_summary.json in the output directory.")


if __name__ == "__main__":
    main()
