"""Run a small patch experiment, not a training/evaluation pipeline."""
import argparse
from pathlib import Path

import numpy as np
import torch

from preprocessing.load_ct import load_volume, load_mask, validate_pair
from preprocessing.preprocess import to_tensors
from .unet3d import UNet3D


def extract_patch(ct: np.ndarray, mask: np.ndarray, label: int,
                  size: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop around a labeled voxel without resizing or using the whole scan as a batch.

    This uses the expert mask to find a patch for a smoke test. It is NOT a
    deployable lesion detector, an unbiased sampler, or a validation protocol.
    """
    validate_pair(ct, mask)
    if label <= 0 or size < 4 or size > min(ct.shape):
        raise ValueError("Use a positive lesion label and patch size between 4 and the smallest scan dimension.")
    # Find one marked voxel without allocating coordinates for every background voxel.
    slices = np.flatnonzero(np.any(mask == label, axis=(0, 1)))
    if not len(slices):
        raise ValueError("Selected lesion label is absent from the mask.")
    z = int(slices[len(slices) // 2])
    xy = np.argwhere(mask[:, :, z] == label)
    center = (*xy[len(xy) // 2], z)
    starts = [max(0, min(int(point) - size // 2, limit - size))
              for point, limit in zip(center, ct.shape)]
    region = tuple(slice(start, start + size) for start in starts)
    return ct[region], mask[region]


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU 3D U-Net patch smoke test; not accuracy evaluation.")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--lesion-label", type=int, required=True,
                        help="Explicit label ID to use for this binary experiment; confirm its meaning first.")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("Use 1 through 20 steps; this command is only a small smoke test.")

    torch.manual_seed(0)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    ct, ct_info = load_volume(args.ct, return_metadata=True)
    mask, mask_info = load_mask(args.mask, return_metadata=True)
    validate_pair(ct, mask, ct_info, mask_info)
    ct_patch, mask_patch = extract_patch(ct, mask, args.lesion_label, args.patch_size)
    sample = to_tensors(ct_patch, mask_patch)
    image = sample["image"].unsqueeze(0)
    # Original mask labels are preserved; only this explicit experiment makes a binary target.
    target = (sample["mask"] == args.lesion_label).float().unsqueeze(0)
    network = UNet3D()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_function = torch.nn.BCEWithLogitsLoss()
    print(f"CPU patch input: {tuple(image.shape)}; positive target voxels: {int(target.sum())}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in network.parameters())}")
    for step in range(args.steps):
        optimizer.zero_grad()
        logits = network(image)
        loss = loss_function(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; stop and inspect the inputs/model.")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in network.parameters()):
            raise RuntimeError("Nonfinite gradients; stop this experiment.")
        optimizer.step()
        print(f"Step {step + 1}: training-patch BCE loss = {loss.item():.6f}; output {tuple(logits.shape)}")
    print("Forward pass, loss, backward pass, and optimizer updates completed.")
    print("Held-out accuracy: N/A. This one-patch experiment is not trained-model validation.")


if __name__ == "__main__":
    main()
