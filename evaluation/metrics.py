"""Explicit binary counts and empty-mask conventions for model comparisons."""
import numpy as np


def confusion(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError("Prediction and target must have identical nonempty shapes.")
    if not np.isin(prediction, [0, 1]).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("Metrics require binary masks, not logits or label IDs.")
    p, t = prediction.astype(bool), target.astype(bool)
    return dict(tp=int(np.count_nonzero(p & t)), fp=int(np.count_nonzero(p & ~t)),
                fn=int(np.count_nonzero(~p & t)), tn=int(np.count_nonzero(~p & ~t)))


def scores(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    def divide(a, b):
        return a / b if b else None
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return {**counts, "dice": dice, "f1": dice,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "sensitivity": divide(tp, tp + fn), "specificity": divide(tn, tn + fp),
            "accuracy": divide(tp + tn, tp + fp + fn + tn)}


def summarize(cases):
    """Micro pools voxels; macro averages cases with defined metric values."""
    if not cases:
        raise ValueError("No cases to summarize.")
    micro = scores({k: sum(c[k] for c in cases) for k in ("tp", "fp", "fn", "tn")})
    names = [k for k in micro if k not in ("tp", "fp", "fn", "tn")]
    macro, defined = {}, {}
    for name in names:
        values = [c[name] for c in cases if c[name] is not None]
        macro[name] = sum(values) / len(values) if values else None
        defined[name] = len(values)
    return {"micro": micro, "macro_case": macro, "macro_defined_cases": defined,
            "case_count": len(cases)}
