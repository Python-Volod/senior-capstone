"""Memorize fixed patches from ONE TRAINING patient; not a validation score."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import confusion, scores
from .experiment import read_manifest, manifest_digest, load_case, sample_patch, write_json
from .losses import segmentation_loss
from .unet3d import UNet3D


def fixed_patches(image, target, size, seed):
    """Two positive crops plus a reproducible, relatively dense negative crop."""
    if not target.any():
        raise ValueError("Choose a training case with annotated foreground.")
    rng = np.random.default_rng(seed)
    positive = [sample_patch(image, target, size, rng, 1.) for _ in range(2)]
    negatives = []
    for _ in range(64):
        pair = sample_patch(image, target, size, rng, 0.)
        if pair[1].sum() == 0:
            negatives.append(pair)
    if not negatives:
        raise ValueError("Could not find a background-only training patch.")
    # Prefer real tissue over an all-air negative; selection uses training only.
    negative = max(negatives, key=lambda pair: pair[0].mean().item())
    pairs = positive + [negative]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


@torch.inference_mode()
def measure(network, image, target):
    network.eval()
    probability = network(image).sigmoid()
    predicted = probability >= .5
    cases = [scores(confusion(p[0].numpy(), t[0].numpy()))
             for p, t in zip(predicted, target)]
    positive = [c["dice"] for c, t in zip(cases, target) if t.any()]
    negative_fp = sum(c["fp"] for c, t in zip(cases, target) if not t.any())
    return {"patches": cases, "mean_patch_dice": sum(c["dice"] for c in cases) / len(cases),
            "min_positive_patch_dice": min(positive), "negative_patch_fp": negative_fp}, probability


def fit_fixed(network, image, target, *, steps, lr, positive_weight, dice_weight,
              report_every=25, callback=None):
    """Keep every input fixed, measuring the SAME examples after optimization."""
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    before, _ = measure(network, image, target)
    history = [{"step": 0, **before}]
    best, best_weights, best_step = before, {k: v.detach().clone() for k, v in network.state_dict().items()}, 0
    for step in range(1, steps + 1):
        network.train()
        optimizer.zero_grad()
        loss = segmentation_loss(network(image), target, mode="dice_bce",
                                 positive_weight=positive_weight, dice_weight=dice_weight)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic loss.")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise ValueError("Nonfinite diagnostic gradients.")
        optimizer.step()
        if step % report_every == 0 or step == steps:
            measured, _ = measure(network, image, target)
            record = {"step": step, "objective": loss.item(), **measured}
            history.append(record)
            passed = measured["min_positive_patch_dice"] >= .9 and measured["negative_patch_fp"] == 0
            if measured["mean_patch_dice"] > best["mean_patch_dice"] or passed:
                best, best_step = measured, step
                best_weights = {k: v.detach().clone() for k, v in network.state_dict().items()}
            if callback:
                callback(record)
            if passed:
                break
    network.load_state_dict(best_weights)
    return before, best, best_step, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.steps < 1 or args.threads < 1:
            raise ValueError("Steps and thread count must be positive.")
        rows = read_manifest(args.manifest)
        row = next((r for r in rows if r["case_id"] == args.case_id), None)
        if row is None or row["split"] != "train":
            raise ValueError("Overfit check may only use a case assigned to train.")
        state = torch.load(args.initialize, map_location="cpu", weights_only=True)
        config = state["config"].copy()
        if config["manifest_digest"] != manifest_digest(rows):
            raise ValueError("Initialization checkpoint has different patient splits.")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose a new diagnostic output directory.")
        args.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        image, target = load_case(row, config["labels"], config["min_hu"], config["max_hu"])
        x, y = fixed_patches(image, target, config["patch_size"], args.seed)
        del image, target
        network = UNet3D(config["base_channels"])
        network.load_state_dict(state["model_state"])
        print("TRAINING-ONLY DIAGNOSTIC: memorizing two positive patches and one negative patch.", flush=True)
        print("Positive voxels per fixed patch:", [int(t.sum()) for t in y], flush=True)
        progress_records = []
        def progress(record):
            progress_records.append(record)
            write_json(args.output / "progress.json", progress_records)
            print(f"Step {record['step']}: min positive Dice={record['min_positive_patch_dice']:.4f}; negative FP={record['negative_patch_fp']}", flush=True)
        before, after, selected_step, history = fit_fixed(
            network, x, y, steps=args.steps, lr=.001,
            positive_weight=config.get("positive_weight", 20.), dice_weight=config.get("dice_weight", .5),
            callback=progress)
        _, probabilities = measure(network, x, y)
        purpose = "TRAINING PATCH MEMORIZATION ONLY; not whole-volume or held-out performance"
        config["purpose"] = purpose
        torch.save({"model_state": network.state_dict(), "config": config, "selected_step": selected_step,
                    "purpose": purpose}, args.output / "diagnostic.pt")
        # Verify serialization without promoting this checkpoint as a production model.
        loaded = torch.load(args.output / "diagnostic.pt", map_location="cpu", weights_only=True)
        reloaded = UNet3D(config["base_channels"])
        reloaded.load_state_dict(loaded["model_state"])
        repeat, _ = measure(reloaded, x, y)
        assert repeat == after
        np.savez_compressed(args.output / "patches.npz", image=x.numpy(), target=y.numpy(), probability=probabilities.numpy())
        summary = {"purpose": purpose, "training_case": args.case_id, "seed": args.seed,
                   "maximum_steps": args.steps, "completed_steps": history[-1]["step"], "selected_step": selected_step,
                   "initial_checkpoint_sha256": hashlib.sha256(args.initialize.read_bytes()).hexdigest(),
                   "criterion": "each positive patch Dice >=0.9 and negative patch FP=0, threshold 0.5",
                   "passed": after["min_positive_patch_dice"] >= .9 and after["negative_patch_fp"] == 0,
                   "before": before, "after": after, "reload_verification": "passed",
                   "untouched": "No validation/test scans loaded by this command; original checkpoint preserved."}
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "history.json", history)
        print(json.dumps(summary, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
