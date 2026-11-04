from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys


def _set_seed(seed):

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def run_training(options):

    from .arguments import validate_training_arguments

    config = validate_training_arguments(options)
    output = Path(config["output_dir"])
    if output.exists():
        raise ValueError("Run output already exists; choose a new output_dir (resume is not implemented): " + str(output))
    from .data import PatchBatches, keras_sequence, load_dataset_splits

    train_records, validation_records = load_dataset_splits(config, validate_volumes=True)
    tf = _set_seed(config["seed"])
    from .evaluation import evaluate_batches
    from .losses import make_loss
    from .models import build_model, save_weights_checked

    model = build_model(config["model"], input_shape=tuple(config["input_shape"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss=make_loss(config["loss"]))
    train = keras_sequence(train_records, config["input_shape"], config["batch_size"],
                           config["preprocessing"], config["shuffle"], config["seed"])
    validation = PatchBatches(validation_records, config["input_shape"], config["batch_size"],
                              config["preprocessing"], shuffle=False)
    train_evaluation = PatchBatches(train_records, config["input_shape"], config["batch_size"],
                                    config["preprocessing"], shuffle=False)
    output.mkdir(parents=True, exist_ok=False)

    with (output / "run_config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, allow_nan=False)
        handle.write("\n")
    best_loss = math.inf
    best_epoch = None
    checkpoint = output / "best.weights.h5"
    metric_names = ("loss", "accuracy", "precision", "recall", "dice", "iou", "miou", "tp", "fp", "fn", "tn", "voxel_count")
    fields = (["epoch", "train_optimization_loss"]
              + [split + "_" + name for split in ("train", "val") for name in metric_names]
              + ["best_checkpoint_updated"])
    with (output / "history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, config["epochs"] + 1):
            weighted_loss, voxel_count = 0.0, 0


            for index in range(len(train)):
                images, labels = train[index]
                batch_loss = float(model.train_on_batch(images, labels, reset_metrics=True))
                if not math.isfinite(batch_loss):
                    raise RuntimeError("Nonfinite training loss at epoch " + str(epoch))
                weighted_loss += batch_loss * int(labels.size)
                voxel_count += int(labels.size)
            train_metrics = evaluate_batches(model, train_evaluation, threshold=config["threshold"], loss=config["loss"])
            metrics = evaluate_batches(model, validation, threshold=config["threshold"], loss=config["loss"])
            improved = metrics["loss"] < best_loss
            if improved:

                save_weights_checked(config["model"], model, checkpoint)
                best_loss, best_epoch = metrics["loss"], epoch
            row = {"epoch": epoch, "train_optimization_loss": weighted_loss / voxel_count,
                   "best_checkpoint_updated": int(improved)}
            row.update({"train_" + key: train_metrics[key] for key in metric_names})
            row.update({"val_" + key: metrics[key] for key in metric_names})
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            train.on_epoch_end()
    summary = {"model": config["model"], "epochs_completed": config["epochs"],
               "best_epoch": best_epoch, "selection_metric": "validation_loss",
               "best_validation_loss": best_loss, "weights": checkpoint.name,
               "train_records": len(train_records), "validation_records": len(validation_records),
               "test_data_used": False}
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def build_parser():
    from .arguments import add_common_arguments
    parser = argparse.ArgumentParser(description="Train a selected registry model using separate train/validation manifests.")
    add_common_arguments(parser)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/train.json"), help="Training manifest (default: data/train.json)")
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/validation.json"), help="Validation manifest (default: data/validation.json)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; existing directories are protected")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Adam learning rate (default: 0.0001)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", default=True,
                        help="Disable seeded epoch-wise training shuffle")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run_training(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print("Training error: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
