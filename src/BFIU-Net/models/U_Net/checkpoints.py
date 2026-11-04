from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


MODEL_ID = "c0_plain_unet_3scale_c32_64_128_maxpool_nearest_nobn_relu_v2"
CONTRACT_SCHEMA_VERSION = 4
CONTRACT_DATASET = "__unet_architecture_contract__"
SOURCE_DATASET = "__unet_source_manifest__"
_SOURCE_FILES = ("models.py", "blocks.py", "fusion.py", "checkpoints.py", "__init__.py")


def _json_default(value):

    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_list"):
        return value.as_list()
    raise TypeError("Non-serializable architecture value: {}".format(type(value).__name__))


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _shape(value):
    return [None if x is None else int(x) for x in value]


def _edge_names(layer_config):

    nodes = layer_config.get("inbound_nodes", [])
    return [edge[0] for node in nodes for edge in node]


def get_architecture_contract(model):

    if model.name != MODEL_ID:
        raise ValueError("Model ID mismatch: expected {!r}, got {!r}".format(MODEL_ID, model.name))
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("C0 plain three-scale U-Net requires one NDHWC input and one output")
    if model.inputs[0].shape.rank != 5 or model.outputs[0].shape.rank != 5:
        raise ValueError("C0 plain three-scale U-Net contract requires rank-5 NDHWC tensors")

    config = model.get_config()
    if "layers" not in config:
        raise ValueError("A native Functional model graph is required")
    live_layers = {layer.name: layer for layer in model.layers}
    graph_layers = []
    for layer in config["layers"]:


        entry = json.loads(_json(layer))
        layer_config = entry["config"]
        if entry["class_name"] == "InputLayer":
            for shape_key in ("batch_input_shape", "batch_shape"):
                if shape_key in layer_config:
                    dims = layer_config[shape_key]
                    if len(dims) != 5:
                        raise ValueError("Only rank-5 inputs are supported")

                    layer_config[shape_key] = [None, None, None, None, dims[-1]]
        actual = live_layers[entry["name"]]
        entry["weight_shapes"] = [_shape(weight.shape) for weight in actual.weights]
        entry["weight_dtypes"] = [weight.dtype.name for weight in actual.weights]
        graph_layers.append(entry)

    topology = {
        "layers": graph_layers,
        "input_layers": config["input_layers"],
        "output_layers": config["output_layers"],
    }
    by_name = {layer["name"]: layer for layer in graph_layers}


    encoder_layers = [name for name in ("stem_relu2", "encoder1_relu2", "encoder2_relu2")
                      if name in by_name]
    encoder_widths = [int(live_layers[name].output.shape[-1]) for name in encoder_layers]
    decoder = {entry["name"]: _edge_names(entry) for entry in graph_layers
               if re.match(r"^decoder[01]_concat$", entry["name"])}
    transposed_convs = [entry for entry in graph_layers if entry["class_name"] == "Conv3DTranspose"]
    convolution_entries = [entry for entry in graph_layers
                           if entry["class_name"] in ("Conv3D", "Conv3DTranspose")]
    transposed_details = []
    for entry in transposed_convs:
        layer = live_layers[entry["name"]]
        transposed_details.append({
            "name": entry["name"],
            "source_order": _edge_names(entry),
            "input_channels": int(layer.input.shape[-1]),
            "output_channels": entry["config"]["filters"],
            "config": entry["config"],
        })
    upsampling_layers = [{"name": entry["name"], "class_name": entry["class_name"],
                          "sources": _edge_names(entry), "config": entry["config"]}
                         for entry in graph_layers
                         if entry["class_name"] in ("UpSampling3D", "Conv3DTranspose")]
    downsampling_layers = [{"name": entry["name"], "class_name": entry["class_name"],
                            "sources": _edge_names(entry), "config": entry["config"]}
                           for entry in graph_layers
                           if entry["class_name"] in ("Conv3D", "MaxPooling3D", "AveragePooling3D") and
                           any(stride > 1 for stride in entry["config"]["strides"])]
    head = by_name.get("head", {})
    residual_adds = [entry["name"] for entry in graph_layers if entry["class_name"] == "Add"]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "input_channels": int(model.inputs[0].shape[-1]),
        "branch_channels": encoder_widths,
        "encoder_channels": encoder_widths,
        "encoder_level_count": len(encoder_layers),
        "encoder_output_layers": encoder_layers,
        "hr_branch_count": len([name for name in by_name if name.startswith("transition2_")]),
        "stage_modules": [],
        "bottleneck_count": sum(bool(re.match(r"^bottleneck\d+_conv1$", name)) for name in by_name),
        "total_basicblocks": sum(bool(re.match(r"^.*_bb\d+_conv1$", name)) for name in by_name),
        "residual_add_layers": residual_adds,
        "raw_skip_sources": {name: sources[1] for name, sources in decoder.items()},
        "batch_normalization_layers": [entry["name"] for entry in graph_layers
                                       if entry["class_name"] == "BatchNormalization"],
        "activation_layers": [{"name": entry["name"], "class_name": entry["class_name"],
                               "config": entry["config"]}
                              for entry in graph_layers
                              if entry["class_name"] in ("ReLU", "Activation", "LeakyReLU", "PReLU")],
        "convolution_activations": {entry["name"]: entry["config"]["activation"] for entry in convolution_entries},
        "convolution_bias": {entry["name"]: entry["config"]["use_bias"] for entry in convolution_entries},
        "decoder_source_order": decoder,
        "decoder_concat_channels": {name: int(live_layers[name].output.shape[-1]) for name in decoder},
        "transposed_convolutions": transposed_details,
        "upsampling_layers": upsampling_layers,
        "downsampling_layers": downsampling_layers,
        "head_config": head.get("config"),
        "parameter_count": int(model.count_params()),
        "topology": topology,
        "topology_sha256": _sha(topology),
    }
    return json.loads(_json(contract))


def _source_manifest():
    directory = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in _SOURCE_FILES}
    return {"sha256": _sha(files), "files": files}


def _h5_path(filepath):
    path = Path(os.fspath(filepath)).expanduser()
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Checked weights require an .h5 or .hdf5 file (not a TF checkpoint prefix)")
    return path


def save_weights_checked(model, filepath, *, overwrite=True):

    import h5py

    path = _h5_path(filepath)
    if not path.parent.is_dir():
        raise FileNotFoundError("Checkpoint parent does not exist: {}".format(path.parent))
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    contract = get_architecture_contract(model)
    sources = _source_manifest()
    descriptor, temporary = tempfile.mkstemp(prefix=".hrnet_checked_", suffix=".h5", dir=str(path.parent))
    os.close(descriptor)
    try:
        model.save_weights(temporary, overwrite=True, save_format="h5")
        with h5py.File(temporary, "r+") as handle:
            handle.create_dataset(CONTRACT_DATASET, data=_json(contract), dtype=h5py.string_dtype("utf-8"))
            handle.create_dataset(SOURCE_DATASET, data=_json(sources), dtype=h5py.string_dtype("utf-8"))
            handle.attrs["model_id"] = MODEL_ID
            handle.attrs["architecture_sha256"] = contract["topology_sha256"]
            handle.attrs["contract_sha256"] = _sha(contract)
            handle.attrs["source_sha256"] = sources["sha256"]
            handle.attrs["checked_weights_schema_version"] = CONTRACT_SCHEMA_VERSION
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.flush()
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return contract


def _validate_weight_shapes(handle, model):

    if "layer_names" not in handle.attrs:
        raise ValueError("Not a Keras 2 HDF5 weights file")
    stored = []
    for raw_name in handle.attrs["layer_names"]:
        name = _text(raw_name)
        group = handle[name]
        weights = [_text(value) for value in group.attrs.get("weight_names", [])]
        if weights:
            stored.append((name, [list(group[weight].shape) for weight in weights]))
    expected = [(layer.name, [_shape(weight.shape) for weight in layer.weights])
                for layer in model.layers if layer.weights]
    if stored != expected:
        raise ValueError("HDF5 weight layer/order/shape mismatch before loading")


def load_weights_checked(model, filepath):

    import h5py

    path = _h5_path(filepath)
    expected = get_architecture_contract(model)
    with h5py.File(str(path), "r") as handle:
        if CONTRACT_DATASET not in handle:
            raise ValueError("Missing architecture contract; raw/legacy/other model weights are not checked-compatible")
        try:
            stored = json.loads(_text(handle[CONTRACT_DATASET][()]))
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid architecture contract JSON") from exc
        if not isinstance(stored, dict):
            raise ValueError("Architecture contract must be a JSON object")
        if stored.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError("Architecture contract schema version mismatch")
        if stored.get("model_id") != MODEL_ID or _text(handle.attrs.get("model_id", "")) != MODEL_ID:
            raise ValueError("Architecture model ID mismatch (HR/residual/other model weights rejected)")
        if stored.get("topology_sha256") != _sha(stored.get("topology")):
            raise ValueError("Architecture topology SHA-256 mismatch")
        if _text(handle.attrs.get("architecture_sha256", "")) != stored["topology_sha256"]:
            raise ValueError("Architecture metadata SHA-256 mismatch")
        if _text(handle.attrs.get("contract_sha256", "")) != _sha(stored):
            raise ValueError("Architecture contract SHA-256 mismatch")
        if stored != expected:
            changes = [key for key in sorted(set(stored) | set(expected)) if stored.get(key) != expected.get(key)]
            raise ValueError("Architecture contract mismatch: {}".format(", ".join(changes)))
        if SOURCE_DATASET not in handle:
            raise ValueError("Missing source SHA-256 provenance")
        try:
            source = json.loads(_text(handle[SOURCE_DATASET][()]))
            valid_source = (source["sha256"] == _sha(source["files"]) and
                            source["sha256"] == _text(handle.attrs.get("source_sha256", "")))
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid source SHA-256 provenance") from exc
        if not valid_source:
            raise ValueError("Source SHA-256 provenance mismatch")
        _validate_weight_shapes(handle, model)
    return model.load_weights(str(path))


__all__ = ["MODEL_ID", "CONTRACT_SCHEMA_VERSION", "CONTRACT_DATASET", "SOURCE_DATASET",
           "get_architecture_contract", "save_weights_checked", "load_weights_checked"]
