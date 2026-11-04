from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def evaluate_batches(model, batches, *, threshold=0.5, loss="bce"):

    from .metrics import BinaryMetrics
    accumulator = BinaryMetrics(threshold, loss)
    for index in range(len(batches)):
        images, labels = batches[index]
        probabilities = model(images, training=False)
        if hasattr(probabilities, "numpy"):
            probabilities = probabilities.numpy()
        accumulator.update(labels, probabilities)
    return accumulator.result()


def run_evaluation(options):

    from .arguments import validate_evaluation_arguments

    config = validate_evaluation_arguments(options)
    threshold = config["threshold"]
    output = Path(config["output"])
    if output.exists():
        raise ValueError("Evaluation output already exists; choose a new output path: " + str(output))
    weights = Path(config["weights"])
    if not weights.is_file():
        raise ValueError("Missing checkpoint: " + str(weights))
    from .data import PatchBatches, load_manifest, validate_records
    records = load_manifest(config["manifest"])
    validate_records(records, config["input_shape"], config["preprocessing"])

    from .models import build_model, load_weights_checked
    model = build_model(config["model"], input_shape=tuple(config["input_shape"]))
    load_weights_checked(config["model"], model, weights)
    batches = PatchBatches(records, config["input_shape"], config["batch_size"], config["preprocessing"])
    metrics = evaluate_batches(model, batches, threshold=threshold, loss=config["loss"])
    report = {"model": config["model"], "threshold": threshold, "loss_name": config["loss"],
              "preprocessing": config["preprocessing"], "input_shape": config["input_shape"],
              "record_count": len(records), "aggregation": "global_voxel_counts_and_log_sums",
              "zero_division": 0, "parameters": config, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def build_parser():
    from .arguments import add_common_arguments
    parser = argparse.ArgumentParser(description="Evaluate one checked checkpoint on an explicitly selected manifest.")
    add_common_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=Path("data/test.json"), help="Evaluation manifest, separate from model-selection validation data (default: data/test.json)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON metrics file; existing files are protected")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_evaluation(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print("Evaluation error: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
