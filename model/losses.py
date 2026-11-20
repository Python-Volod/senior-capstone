"""Foreground-aware binary segmentation loss; hyperparameters need validation."""
import math
from torch.nn import functional as F


def segmentation_loss(logits, target, *, mode="bce", positive_weight=1., dice_weight=.5):
    """BCE uses raw logits; soft Dice gives foreground overlap an explicit signal.

    Empty patches still contribute BCE. Dice is averaged per patch, with a
    smoothing constant to keep empty-target gradients finite. Weights are fixed
    experimental choices, not estimates derived from the validation/test set.
    """
    if logits.shape != target.shape or mode not in {"bce", "dice_bce"}:
        raise ValueError("Matching logits/targets and a supported loss are required.")
    if not math.isfinite(positive_weight) or not positive_weight > 0 or not 0 <= dice_weight <= 1:
        raise ValueError("Invalid positive or Dice weight.")
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor(positive_weight))
    if mode == "bce":
        return bce
    probability = logits.sigmoid().flatten(1)
    truth = target.flatten(1)
    dice = (2 * (probability * truth).sum(1) + 1) / (probability.sum(1) + truth.sum(1) + 1)
    return (1 - dice_weight) * bce + dice_weight * (1 - dice.mean())
