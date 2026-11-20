"""CPU baseline training and whole-volume evaluation with patient-disjoint splits.

Run `python -m model.experiment --help`. All manifests, checkpoints and reports
are local research artifacts. This baseline does not establish clinical utility.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import shutil
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.metrics import confusion, scores, summarize
from preprocessing.load_ct import load_volume, load_mask, validate_pair
from preprocessing.preprocess import normalize_ct
from .unet3d import UNet3D
from .losses import segmentation_loss


def read_manifest(path):
    """Resolve paths relative to the CSV and reject overlapping patient groups."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"case_id", "patient_id", "ct", "mask", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs case_id,patient_id,ct,mask,split columns.")
        rows = list(reader)
    patients, cases, files = {}, set(), set()
    result = []
    for row in rows:
        row = {k: (row.get(k) or "").strip() for k in
               ("case_id", "patient_id", "ct", "mask", "split")}
        if not all(row.values()) or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Every manifest field is required; split must be train, val or test.")
        patient, split = row["patient_id"], row["split"]
        if patient in patients and patients[patient] != split:
            raise ValueError("Patient leakage: repeated scans must remain in one split.")
        patients[patient] = split
        if row["case_id"] in cases:
            raise ValueError("case_id must be unique.")
        cases.add(row["case_id"])
        for key in ("ct", "mask"):
            source = (path.parent / row[key]).resolve()
            # Reject repeated paths even if they are given different case/patient IDs.
            identity = str(source).casefold()
            if identity in files:
                raise ValueError("Duplicate CT/mask path in manifest.")
            files.add(identity)
            if not source.is_file():
                raise ValueError("A manifest CT/mask file does not exist.")
            row[key] = str(source)
        result.append(row)
    if {r["split"] for r in result} != {"train", "val", "test"}:
        raise ValueError("Provide distinct patients in train, val and test; one sample is insufficient.")
    return result


def manifest_digest(rows):
    ordered = sorted(rows, key=lambda row: row["case_id"])
    return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()


def load_case(row, labels, min_hu, max_hu):
    ct, ci = load_volume(row["ct"], return_metadata=True)
    mask, mi = load_mask(row["mask"], return_metadata=True)
    validate_pair(ct, mask, ci, mi)
    if not set(np.unique(mask)) <= {0, *labels}:
        raise ValueError("Unselected nonzero mask labels: review the binary target policy.")
    image = normalize_ct(ct, min_hu, max_hu).transpose(2, 1, 0).copy()
    target = np.isin(mask, labels).transpose(2, 1, 0).copy()
    return image, target


def sample_patch(image, target, size, rng, foreground_probability=.5, foreground_jitter=0):
    """Training only: mix uniform and foreground-centered crops when present."""
    pads = [(0, max(0, size - n)) for n in image.shape]
    if any(p[1] for p in pads):
        image, target = np.pad(image, pads), np.pad(target, pads)
    starts = [int(rng.integers(n - size + 1)) for n in image.shape]
    # Slice-first sampling avoids allocating all foreground coordinates in 3D.
    if rng.random() < foreground_probability and target.any():
        zs = np.flatnonzero(target.any(axis=(1, 2)))
        z = int(rng.choice(zs))
        xy = np.argwhere(target[z])
        y, x = xy[int(rng.integers(len(xy)))]
        offsets = rng.integers(-foreground_jitter, foreground_jitter + 1, size=3) if foreground_jitter else (0, 0, 0)
        starts = [max(0, min(int(c) - size // 2 + int(offset), n - size))
                  for c, n, offset in zip((z, y, x), image.shape, offsets)]
    region = tuple(slice(v, v + size) for v in starts)
    return (torch.from_numpy(image[region].copy())[None, None],
            torch.from_numpy(target[region].astype(np.float32))[None, None])


@torch.inference_mode()
def infer_volume(network, image, size, threshold, batch_size=1):
    """Tile the entire input independently of annotations, including edge voxels.

    Nonoverlapping tiles keep this baseline simple; seams are a known limitation.
    Padding is normalized air (0). Padded voxels are cropped out of the result.
    """
    network.eval()
    result = np.empty(image.shape, dtype=bool)
    ranges = [range(0, n, size) for n in image.shape]
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive.")
    positions = iter(itertools.product(*ranges))
    while starts := list(itertools.islice(positions, batch_size)):
        tiles, regions, shapes = [], [], []
        for start in starts:
            region = tuple(slice(s, min(s + size, n)) for s, n in zip(start, image.shape))
            tile = image[region]
            tiles.append(np.pad(tile, [(0, size - n) for n in tile.shape]))
            regions.append(region)
            shapes.append(tile.shape)
        probability = network(torch.from_numpy(np.stack(tiles)[:, None])).sigmoid().numpy()[:, 0]
        for pred, region, shape in zip(probability, regions, shapes):
            result[region] = pred[tuple(slice(0, n) for n in shape)] >= threshold
    return result


def evaluate_rows(network, rows, config):
    cases, baseline, times = [], [], []
    for row in rows:
        image, target = load_case(row, config["labels"], config["min_hu"], config["max_hu"])
        start = time.perf_counter()
        pred = infer_volume(network, image, config["patch_size"], config["threshold"],
                            config.get("inference_batch_size", 1))
        elapsed = time.perf_counter() - start
        cases.append({"case_id": row["case_id"], **scores(confusion(pred, target)),
                      "inference_seconds": elapsed, "shape_dhw": list(image.shape)})
        baseline.append(scores(confusion(np.zeros_like(target), target)))
        times.append(elapsed)
    return {"unit": "voxel", "aggregation": summarize(cases), "cases": cases,
            "all_background_baseline": summarize(baseline),
            "timing": {"mean_inference_seconds": sum(times) / len(times),
                       "scope": "CPU network tiling + mask assembly; excludes file loading/preprocessing; no warmup"},
            "clinical_validation": False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def train(args, rows):
    resume = getattr(args, "resume", None)
    previous = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if previous is not None:
        if previous["config"]["manifest_digest"] != manifest_digest(rows):
            raise ValueError("Resume manifest differs from the original training splits.")
        if "numpy_rng_state" not in previous or "torch_rng_state" not in previous:
            raise ValueError("This older checkpoint has no RNG state; exact training resume is unsupported.")
        parent = resume.parent
        if resume.name != "last.pt" or not (parent / "best.pt").is_file() or not (parent / "history.json").is_file():
            raise ValueError("Resume from last.pt alongside its best.pt and history.json.")
        history = json.loads((parent / "history.json").read_text(encoding="utf-8"))
        if not history or history[-1]["epoch"] != previous["epoch"]:
            raise ValueError("Checkpoint and history epochs differ.")
        best_state = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
        if (best_state["config"]["manifest_digest"] != previous["config"]["manifest_digest"]
                or best_state["epoch"] > previous["epoch"]
                or best_state["validation_macro_dice"] != previous["best_validation_macro_dice"]):
            raise ValueError("Best checkpoint does not match this run.")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in
              ("labels", "patch_size", "base_channels", "epochs", "patches_per_case",
               "seed", "lr", "min_hu", "max_hu", "threshold", "threads")}
    config["manifest_digest"] = manifest_digest(rows)
    for key, default in (("loss", "bce"), ("positive_weight", 1.), ("dice_weight", .5),
                         ("foreground_probability", .5), ("inference_batch_size", 1),
                         ("hard_negative_probability", 0.), ("foreground_jitter", 0)):
        config[key] = getattr(args, key, default)
    config["schema_version"] = 4
    hard_negative_path = getattr(args, "hard_negatives", None)
    hard_index = None
    if hard_negative_path is not None:
        from .hard_negatives import read_index
        hard_index = read_index(hard_negative_path, rows, config)
        config["hard_negatives"] = str(Path(hard_negative_path).resolve())
        digest = hashlib.sha256(Path(hard_negative_path).read_bytes()).hexdigest()
        if previous is not None and previous["config"].get("hard_negative_index_sha256") != digest:
            raise ValueError("Hard-negative index changed since the checkpoint.")
        config["hard_negative_index_sha256"] = digest
        config["hard_negative_source_checkpoint_sha256"] = hard_index["source_checkpoint_sha256"]
    elif config["hard_negative_probability"] > 0:
        raise ValueError("Hard-negative sampling requires --hard-negatives index.json.")
    if previous is not None:
        config["resumed_from_sha256"] = hashlib.sha256(resume.read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    network = UNet3D(args.base_channels)
    initialize = getattr(args, "initialize", None)
    if initialize is not None:
        if previous is not None:
            raise ValueError("Weight initialization and exact resume are separate operations.")
        initial = torch.load(initialize, map_location="cpu", weights_only=True)
        if initial["config"]["manifest_digest"] != config["manifest_digest"]:
            raise ValueError("Initialization checkpoint must use the same patient splits.")
        for key in ("labels", "base_channels", "min_hu", "max_hu"):
            if initial["config"][key] != config[key]:
                raise ValueError("Initialization must preserve label, architecture and intensity conventions.")
        network.load_state_dict(initial["model_state"])
        config["initialized_from_sha256"] = hashlib.sha256(initialize.read_bytes()).hexdigest()
        config["initialized_from_epoch"] = initial["epoch"]
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    best, first_epoch = -1., 1
    if previous is None:
        history = []
    else:
        network.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        rng.bit_generator.state = previous["numpy_rng_state"]
        torch.set_rng_state(previous["torch_rng_state"])
        best = previous["best_validation_macro_dice"]
        first_epoch = previous["epoch"] + 1
        # Preserve the earlier best even if subsequent epochs do not improve.
        shutil.copyfile(resume.parent / "best.pt", output / "best.pt")
    write_json(output / "config.json", config)
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        network.train()
        losses, bces, foreground_fractions = [], [], []
        hard_samples = 0
        for case_number, index in enumerate(rng.permutation(len(train_rows)), 1):
            image, target = load_case(train_rows[index], args.labels, args.min_hu, args.max_hu)
            for _ in range(args.patches_per_case):
                hard_probability = config["hard_negative_probability"]
                if hard_probability > 0 and rng.random() < hard_probability:
                    entries = hard_index["cases"][train_rows[index]["case_id"]]
                    if entries:
                        from .hard_negatives import extract_negative
                        x, y = extract_negative(image, target, args.patch_size, entries[int(rng.integers(len(entries)))])
                        hard_samples += 1
                    else:
                        x, y = sample_patch(image, target, args.patch_size, rng, 0.)
                else:
                    conditional_foreground = config["foreground_probability"] / (1 - hard_probability)
                    x, y = sample_patch(image, target, args.patch_size, rng, conditional_foreground,
                                        config["foreground_jitter"])
                optimizer.zero_grad()
                logits = network(x)
                loss = segmentation_loss(logits, y, mode=config["loss"],
                                         positive_weight=config["positive_weight"],
                                         dice_weight=config["dice_weight"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss.")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                    raise ValueError("Nonfinite gradients.")
                optimizer.step()
                losses.append(loss.item())
                bces.append(F.binary_cross_entropy_with_logits(logits.detach(), y).item())
                foreground_fractions.append(y.mean().item())
            print(f"Epoch {epoch}: training case {case_number}/{len(train_rows)} completed", flush=True)
        train_seconds = time.perf_counter() - epoch_start
        print(f"Epoch {epoch}: evaluating complete validation scans", flush=True)
        validation = evaluate_rows(network, val_rows, config)
        # Macro case Dice selects the checkpoint; test data is never loaded here.
        dice = validation["aggregation"]["macro_case"]["dice"]
        history.append({"epoch": epoch, "train_patch_loss": sum(losses) / len(losses),
                        "train_patch_bce": sum(bces) / len(bces),
                        "train_patch_foreground_fraction": sum(foreground_fractions) / len(foreground_fractions),
                        "hard_negative_samples": hard_samples,
                        "train_seconds": train_seconds,
                        "validation_seconds": time.perf_counter() - epoch_start - train_seconds,
                        "validation": validation})
        improved = dice > best
        best = max(best, dice)
        state = {"config": config, "epoch": epoch, "model_state": network.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "validation_macro_dice": dice,
                 "best_validation_macro_dice": best, "numpy_rng_state": rng.bit_generator.state,
                 "torch_rng_state": torch.get_rng_state()}
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}: patch loss={history[-1]['train_patch_loss']:.6f}; val macro Dice={dice:.6f}", flush=True)
    print("Saved best.pt and last.pt. Test performance is not measured by training.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Train patches; select checkpoint on whole validation scans.")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--labels", type=int, nargs="+", required=True)
    training.add_argument("--patch-size", type=int, default=32)
    training.add_argument("--base-channels", type=int, default=4)
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--patches-per-case", type=int, default=8)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--lr", type=float, default=.001)
    training.add_argument("--min-hu", type=float, default=-1000)
    training.add_argument("--max-hu", type=float, default=400)
    training.add_argument("--threshold", type=float, default=.5)
    training.add_argument("--loss", choices=("bce", "dice_bce"), default="bce")
    training.add_argument("--positive-weight", type=float, default=1.)
    training.add_argument("--dice-weight", type=float, default=.5)
    training.add_argument("--foreground-probability", type=float, default=.5)
    training.add_argument("--inference-batch-size", type=int, default=1)
    training.add_argument("--initialize", type=Path, help="Start a new experiment from weights; reset optimizer and epoch count.")
    training.add_argument("--hard-negatives", type=Path, help="Training-only index.json from model.hard_negatives.")
    training.add_argument("--hard-negative-probability", type=float, default=0.)
    training.add_argument("--foreground-jitter", type=int, default=0)
    testing = sub.add_parser("evaluate", help="Reload a checkpoint and evaluate untouched test scans.")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--report", type=Path, required=True)
    resuming = sub.add_parser("resume", help="Continue a saved last.pt into a fresh run directory.")
    resuming.add_argument("--checkpoint", type=Path, required=True)
    resuming.add_argument("--output", type=Path, required=True)
    resuming.add_argument("--epochs", type=int, required=True, help="Additional epochs.")
    for command in (training, testing, resuming):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive.")
        torch.set_num_threads(args.threads)
        rows = read_manifest(args.manifest)
        if args.command == "resume":
            if args.epochs < 1:
                raise ValueError("Additional epochs must be positive.")
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            config = state["config"]
            if args.threads != config["threads"]:
                raise ValueError("Use the original thread count for reproducible resume.")
            resumed = argparse.Namespace(**config)
            resumed.output, resumed.resume = args.output, args.checkpoint
            resumed.epochs = state["epoch"] + args.epochs
            train(resumed, rows)
        elif args.command == "train":
            if (not 0 <= args.hard_negative_probability < 1
                    or args.foreground_probability + args.hard_negative_probability > 1
                    or not 0 <= args.foreground_jitter < args.patch_size // 2):
                raise ValueError("Sampling probabilities must sum to <=1; jitter must be less than half the patch size.")
            if (not math.isfinite(args.positive_weight) or args.positive_weight <= 0
                    or not 0 <= args.dice_weight <= 1 or not 0 <= args.foreground_probability <= 1
                    or args.inference_batch_size < 1):
                raise ValueError("Invalid loss weight, sampling probability or inference batch size.")
            if (args.patch_size < 4 or args.base_channels < 1 or args.epochs < 1
                    or args.patches_per_case < 1 or not args.labels or min(args.labels) < 1
                    or len(set(args.labels)) != len(args.labels)):
                raise ValueError("Use positive unique label IDs, patch size >=4 and positive training sizes.")
            if (not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.threshold < 1
                    or not math.isfinite(args.min_hu) or not math.isfinite(args.max_hu)
                    or args.min_hu >= args.max_hu):
                raise ValueError("Invalid learning rate, HU bounds or threshold.")
            train(args, rows)
        else:
            if args.report.exists():
                raise ValueError("Report already exists; choose a new report path.")
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            config = state["config"]
            if config["manifest_digest"] != manifest_digest(rows):
                raise ValueError("Manifest differs from the training manifest; preserve the original splits.")
            network = UNet3D(config["base_channels"])
            network.load_state_dict(state["model_state"])
            report = evaluate_rows(network, [r for r in rows if r["split"] == "test"], config)
            report.update(model="small_3d_unet", checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          config=config, epoch=state["epoch"], evaluation_threads=args.threads,
                          torch_version=str(torch.__version__),
                          environment={"python": platform.python_version(), "platform": platform.platform(),
                                       "processor": platform.processor(), "device": "cpu"},
                          parameters=sum(p.numel() for p in network.parameters()))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report)
            print(json.dumps(report["aggregation"], indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
