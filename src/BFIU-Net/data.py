from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _shape3(shape) -> Tuple[int, int, int]:
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError("shape must contain exactly three positive integer dimensions")
    return tuple(shape)


def _numeric_dtype(dtype):
    try:
        value = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric dtype") from exc
    if value.kind not in "fiu":
        raise ValueError("dtype must describe real floating-point or integer values")
    return value


def load_volume(path, shape=None, dtype="<f4") -> np.ndarray:

    path = Path(path)
    wanted_shape = _shape3(shape) if shape is not None else None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(str(path), allow_pickle=False)
        if not isinstance(array, np.ndarray):
            if hasattr(array, "close"):
                array.close()
            raise ValueError("Expected a single .npy array, not an archive: " + str(path))
    elif suffix in (".dat", ".bin"):
        if wanted_shape is None:
            raise ValueError("Raw .dat/.bin volume requires shape: " + str(path))
        raw_dtype = _numeric_dtype(dtype)
        expected_bytes = int(np.prod(wanted_shape, dtype=object)) * raw_dtype.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError("Raw file size does not match shape and dtype: " + str(path))
        array = np.fromfile(str(path), dtype=raw_dtype).reshape(wanted_shape, order="C")
    else:
        raise ValueError("Supported volume formats are .npy, .dat and .bin: " + str(path))
    if array.ndim != 3 or any(v == 0 for v in array.shape):
        raise ValueError("Volume must be a nonempty 3-D array: " + str(path))
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Volume must contain finite real numeric values: " + str(path))
    if wanted_shape is not None and array.shape != wanted_shape:
        raise ValueError("Volume shape disagrees with manifest: " + str(path))
    return np.ascontiguousarray(array)


def normalize_volume(volume, mode="zscore") -> np.ndarray:

    array = np.asarray(volume)
    if array.ndim != 3 or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Normalization requires a finite, real 3-D volume")
    if mode == "zscore":
        array = array.astype(np.float64)

        scale = float(np.max(np.abs(array)))
        if scale == 0:
            return np.zeros(array.shape, dtype=np.float32)
        array = array / scale
        mean = array.mean(dtype=np.float64)
        std = array.std(dtype=np.float64)
        array = np.zeros_like(array) if std == 0 else (array - mean) / std
    elif mode != "none":
        raise ValueError("preprocessing must be zscore or none")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Volume values are outside the finite float32 range")
    return result


@dataclass(frozen=True)
class VolumeRecord:
    id: str
    image: Path
    label: Path
    shape: Optional[Tuple[int, int, int]] = None
    dtype: str = "<f4"


def load_manifest(path) -> tuple:

    try:
        path = Path(path).expanduser().resolve()
        entries = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise ValueError("Read a valid dataset manifest: " + str(path)) from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("Dataset manifest must be a nonempty list of records")
    records, ids, files = [], set(), set()
    for entry in entries:
        if (not isinstance(entry, dict) or not {"id", "image", "label"}.issubset(entry)
                or set(entry) - {"id", "image", "label", "shape", "dtype"}):
            raise ValueError("Each manifest record requires id,image,label; optional shape,dtype")
        for key in ("id", "image", "label"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError("Manifest " + key + " must be a nonempty string")
        if entry["id"] in ids:
            raise ValueError("Duplicate manifest id: " + entry["id"])
        ids.add(entry["id"])
        paths = []
        for key in ("image", "label"):
            try:
                value = Path(entry[key]).expanduser()
                value = (value if value.is_absolute() else path.parent / value).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError("Invalid manifest volume path: " + entry[key]) from exc
            if value in files:
                raise ValueError("Duplicate volume path within manifest: " + str(value))
            if not value.is_file():
                raise ValueError("Missing volume file: " + str(value))
            files.add(value)
            paths.append(value)
        shape = _shape3(entry["shape"]) if "shape" in entry else None
        raw_dtype = entry.get("dtype", "<f4")
        if not isinstance(raw_dtype, str) or not raw_dtype:
            raise ValueError("Manifest dtype must be a nonempty numeric dtype string")
        dtype = _numeric_dtype(raw_dtype).str
        records.append(VolumeRecord(entry["id"], paths[0], paths[1], shape, dtype))
    return tuple(records)


def load_pair(record: VolumeRecord, input_shape, preprocessing="zscore"):
    image = load_volume(record.image, record.shape, record.dtype)
    label = load_volume(record.label, record.shape, record.dtype)
    if image.shape != label.shape:
        raise ValueError("Image/label shape mismatch for " + record.id)
    if image.shape != tuple(input_shape[:3]):
        raise ValueError("Patch shape must equal input_shape; pre-extract patches explicitly: " + record.id)
    if not np.logical_or(label == 0, label == 1).all():
        raise ValueError("Labels must contain only binary values 0 and 1: " + record.id)
    return normalize_volume(image, preprocessing)[..., None], np.asarray(label, dtype=np.float32)[..., None]


def validate_records(records, input_shape, preprocessing="zscore") -> None:
    for record in records:
        load_pair(record, input_shape, preprocessing)


def ensure_disjoint(first, second) -> None:

    paths = {p for record in first for p in (record.image, record.label)}
    overlap = paths & {p for record in second for p in (record.image, record.label)}
    if overlap:
        raise ValueError("Dataset splits overlap at resolved volume path: " + str(sorted(overlap)[0]))


def load_dataset_splits(config, *, validate_volumes=True):
    for key in ("train_manifest", "validation_manifest"):
        if config[key] is None:
            raise ValueError(key + " is unset; select separate training and validation manifests")
    train = load_manifest(config["train_manifest"])
    validation = load_manifest(config["validation_manifest"])
    ensure_disjoint(train, validation)
    if validate_volumes:
        for records in (train, validation):
            validate_records(records, config["input_shape"], config["preprocessing"])
    return train, validation


class PatchBatches:


    def __init__(self, records, input_shape, batch_size=1, preprocessing="zscore", shuffle=False, seed=42):
        if type(batch_size) is not int or batch_size < 1 or not records:
            raise ValueError("Positive batch_size and nonempty records are required")
        self.records = tuple(records)
        self.input_shape = tuple(input_shape)
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(records))
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        selected = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        pairs = [load_pair(self.records[i], self.input_shape, self.preprocessing) for i in selected]
        return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def keras_sequence(*args, **kwargs):

    from tensorflow.keras.utils import Sequence

    class VolumeSequence(Sequence):
        def __init__(self):
            self.batches = PatchBatches(*args, **kwargs)

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, index):
            return self.batches[index]

        def on_epoch_end(self):
            self.batches.on_epoch_end()

    return VolumeSequence()
