from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from numbers import Integral
import os
from pathlib import Path
import sys
import uuid

import numpy as np


def _triple(values, name):
    values = tuple(values)
    if len(values) != 3 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in values):
        raise ValueError(f"{name} must contain three positive integers")
    return tuple(int(x) for x in values)


def sliding_starts(length, patch, stride):
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in (length, patch, stride)):
        raise ValueError("length, patch and stride must be positive integers")
    if stride > patch:
        raise ValueError("stride must not exceed patch size")
    end = max(0, int(length) - int(patch))
    starts = list(range(0, end + 1, int(stride)))
    if starts[-1] != end:
        starts.append(end)
    return starts


def infer_volume(volume, predictor, *, patch_size=(128, 128, 128), stride=(64, 64, 64), progress=None):

    patch_size = _triple(patch_size, "patch_size")
    stride = _triple(stride, "stride")
    if any(s > p for s, p in zip(stride, patch_size)):
        raise ValueError("stride must not exceed patch size on any axis")
    raw = np.asarray(volume)
    if raw.ndim != 3 or any(n == 0 for n in raw.shape) or raw.dtype.kind not in "fiu":
        raise ValueError("volume must be a non-empty real numeric 3-D array")
    array = raw.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values")
    original_shape = array.shape
    padding = tuple((0, max(0, p-n)) for n, p in zip(original_shape, patch_size))
    padded = np.pad(array, padding, mode="constant") if any(b for _, b in padding) else array
    sums = np.zeros(padded.shape, dtype=np.float64)
    counts = np.zeros(padded.shape, dtype=np.uint32)
    starts = [sliding_starts(n, p, s) for n, p, s in zip(padded.shape, patch_size, stride)]
    total = int(np.prod([len(x) for x in starts]))
    expected = (1,) + patch_size + (1,)
    for index, position in enumerate(itertools.product(*starts), 1):
        region = tuple(slice(start, start+size) for start, size in zip(position, patch_size))
        batch = padded[region][None, ..., None]
        predicted = np.asarray(predictor(batch))
        if predicted.shape != expected or predicted.dtype.kind not in "fiu":
            raise ValueError(f"predictor must return real numeric probabilities of shape {expected}")
        if not np.isfinite(predicted).all() or np.any(predicted < 0) or np.any(predicted > 1):
            raise ValueError("predictor output must be finite probabilities in [0, 1]")
        sums[region] += predicted[0, ..., 0]
        counts[region] += 1
        if progress is not None:
            progress(index, total)
    if np.any(counts == 0):
        raise RuntimeError("Internal error: an output voxel was not covered")
    sums /= counts
    crop = tuple(slice(0, n) for n in original_shape)
    return sums[crop].astype(np.float32)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_prediction(output, probabilities, metadata, *, mask_output=None, threshold=0.5):

    output = Path(output)
    sidecar = output.with_suffix(".json")
    if output.suffix.lower() != ".npy":
        raise ValueError("Probability output must end with .npy")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or probabilities.dtype.kind not in "fiu" or not np.isfinite(probabilities).all():
        raise ValueError("Probability array must be a finite real 3-D array")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be in [0, 1]")
    destinations = [output]
    arrays = [probabilities.astype(np.float32)]
    if mask_output is not None:
        mask_output = Path(mask_output)
        if mask_output.suffix.lower() != ".npy":
            raise ValueError("Mask output must end with .npy")
        destinations.append(mask_output)
        arrays.append((probabilities >= threshold).astype(np.uint8))
    destinations.append(sidecar)
    if len({str(x.resolve()).casefold() for x in destinations}) != len(destinations):
        raise ValueError("Probability, mask and metadata paths must be different")
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path.name}")
    reserved, temporaries = [], []
    try:
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb"):
                pass
            reserved.append(path)
        for path, array in zip(destinations[:-1], arrays):
            temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
            temporaries.append(temp)
            with temp.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temp, path)
        temp = sidecar.with_name(sidecar.name + ".tmp-" + uuid.uuid4().hex)
        temporaries.append(temp)
        content = dict(metadata)
        content.update(status="complete", probability_file=output.name,
                       probability_sha256=_sha256(output), shape=list(probabilities.shape),
                       dtype="float32", axis_order="unchanged from input array",
                       mask_file=mask_output.name if mask_output is not None else None,
                       threshold=threshold if mask_output is not None else None)
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, allow_nan=False)
        os.replace(temp, sidecar)
    except BaseException:

        for path in temporaries + reserved:
            if path.exists():
                path.unlink()
        raise
    return sidecar


def main(argv=None):
    from .models import list_models, normalize_model_name
    parser = argparse.ArgumentParser(description="Apply a checked model to a 3-D .npy or explicitly shaped raw volume.")
    parser.add_argument("--model", type=normalize_model_name, choices=list_models(), required=True,
                        help="Model ID (case-insensitive; hyphens and underscores are accepted)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, help="Required array shape for raw .dat/.bin input")
    parser.add_argument("--dtype", default="<f4", help="Raw dtype and byte order; default little-endian float32")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128,128,128))
    parser.add_argument("--stride", type=int, nargs=3, default=(64,64,64))
    parser.add_argument("--normalization", choices=("zscore","none"), default="zscore")
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        patch = _triple(args.patch_size, "patch_size")
        stride = _triple(args.stride, "stride")
        if any(p % 4 for p in patch):
            raise ValueError("Model patch dimensions must be divisible by four")
        if any(s>p for s,p in zip(stride,patch)):
            raise ValueError("stride must not exceed patch_size")
        if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if args.output.suffix.lower() != ".npy" or (args.mask_output and args.mask_output.suffix.lower() != ".npy"):
            raise ValueError("Output arrays must end with .npy")
        outputs = [args.output, args.output.with_suffix(".json")]
        if args.mask_output:
            outputs.append(args.mask_output)
        if len({str(p.resolve()).casefold() for p in outputs}) != len(outputs):
            raise ValueError("Output paths must be distinct")
        for path in outputs:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Output already exists: {path.name}")
        if not args.weights.is_file():
            raise ValueError("A model-specific checked checkpoint file is required")
        from .data import load_volume, normalize_volume
        from .models import build_model, load_weights_checked
        input_sha = _sha256(args.input)
        raw_volume = load_volume(args.input, shape=args.shape, dtype=args.dtype)
        input_dtype = raw_volume.dtype.str
        if _sha256(args.input) != input_sha:
            raise ValueError("Input volume changed while being read")
        volume = normalize_volume(raw_volume, mode=args.normalization)
        del raw_volume
        checkpoint_sha = _sha256(args.weights)
        model = build_model(args.model, input_shape=patch+(1,))
        load_weights_checked(args.model, model, args.weights)
        if _sha256(args.weights) != checkpoint_sha:
            raise ValueError("Checkpoint changed while being loaded")
        def report(done, total):
            if done == 1 or done == total or done % 100 == 0:
                print(f"Tiles {done}/{total}", file=sys.stderr)
        probability = infer_volume(volume, lambda x: model(x, training=False),
                                   patch_size=patch, stride=stride, progress=report)
        if _sha256(args.input) != input_sha:
            raise ValueError("Input volume changed during prediction; no results were published")
        sidecar = write_prediction(args.output, probability,
            {"model": args.model, "model_id": model.name, "checkpoint_sha256": checkpoint_sha,
             "input_sha256": input_sha, "input_format": args.input.suffix.lower(),
             "input_storage_dtype": input_dtype,
             "raw_dtype": args.dtype if args.input.suffix.lower() in (".dat", ".bin") else None,
             "patch_size": list(patch), "stride": list(stride),
             "normalization": args.normalization, "blending": "uniform probability mean",
             "padding": "right-side constant zero on undersized axes, cropped after prediction"},
            mask_output=args.mask_output, threshold=args.threshold)
        print(json.dumps({"status":"complete", "shape":list(probability.shape), "metadata":str(sidecar)}))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
