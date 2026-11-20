"""Score another model's binary NIfTI outputs on the identical held-out cases.

Predictions must cover whole scans and must be produced without looking at test
annotations. This utility cannot verify another model's training provenance.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from preprocessing.load_ct import load_mask, validate_pair
from .metrics import confusion, scores, summarize


def main(argv=None):
    # Import here to avoid a module cycle with the experiment's metric imports.
    from model.experiment import read_manifest, manifest_digest, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="CSV columns: case_id,prediction; paths relative to this CSV.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = read_manifest(args.manifest)
        reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
        config = reference["config"]
        if config["manifest_digest"] != manifest_digest(rows):
            raise ValueError("Comparison manifest must match the reference evaluation exactly.")
        if not args.name.strip() or args.name == reference["model"]:
            raise ValueError("Use a distinct nonempty model name.")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Comparison output directory must be empty.")
        tests = {r["case_id"]: r for r in rows if r["split"] == "test"}
        with args.predictions.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"case_id", "prediction"} <= set(reader.fieldnames or []):
                raise ValueError("Prediction CSV needs case_id,prediction columns.")
            predictions = list(reader)
        ids = [r["case_id"] for r in predictions]
        if len(ids) != len(set(ids)) or set(ids) != set(tests):
            raise ValueError("Predictions must cover each held-out case exactly once, with no extras.")
        if {c["case_id"] for c in reference["cases"]} != set(tests):
            raise ValueError("Reference report must cover the same test cases.")
        cases = []
        for prediction in predictions:
            row = tests[prediction["case_id"]]
            predicted, pi = load_mask(args.predictions.resolve().parent / prediction["prediction"], return_metadata=True)
            target, ti = load_mask(row["mask"], return_metadata=True)
            validate_pair(predicted, target, pi, ti)
            if not np.isin(predicted, [0, 1]).all():
                raise ValueError("Predictions must be binary 0/1 masks, not probability maps.")
            if not set(np.unique(target)) <= {0, *config["labels"]}:
                raise ValueError("Unselected target labels in test mask.")
            cases.append({"case_id": row["case_id"],
                          **scores(confusion(predicted, np.isin(target, config["labels"])))})
        report = {"model": args.name, "config": config, "unit": "voxel",
                  "aggregation": summarize(cases), "cases": cases,
                  "training_provenance": "User must verify no test patients were used in training or tuning.",
                  "clinical_validation": False}
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "report.json", report)
        columns = ["model", "aggregation", "dice", "iou", "precision", "recall", "f1", "accuracy"]
        with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in (reference, report):
                for aggregation in ("micro", "macro_case"):
                    writer.writerow({"model": item["model"], "aggregation": aggregation,
                                     **{k: item["aggregation"][aggregation][k] for k in columns[2:]}})
        print("Saved report.json and comparison.csv; no runtime ranking is inferred from exported masks.")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
