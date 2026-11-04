from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


MODEL_ID = "hrnet_r5_lowtohigh_e1start_e0skip_c32_64_128_bneck2_tconvalign_nobn_v2"
CONTRACT_SCHEMA_VERSION = 4
CONTRACT_DATASET = "__hrnet_architecture_contract__"
SOURCE_DATASET = "__hrnet_source_manifest__"
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
        raise ValueError("E1-start/E0-skip model requires one NDHWC input and one output")
    if model.inputs[0].shape.rank != 5 or model.outputs[0].shape.rank != 5:
        raise ValueError("E1-start/E0-skip model contract requires rank-5 NDHWC tensors")

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
    modules = {}
    if any(re.match(r"^(?:stage[3-9]_|transition[3-9]_)", name) for name in by_name):
        raise ValueError("This three-level model contains Stage2 only; later stages are invalid")
    for stage in (2,):
        expression = re.compile(r"^stage{}_module(\d+)_fuse_target0_relu$".format(stage))
        indices = sorted(int(match.group(1)) for name in by_name
                         for match in [expression.match(name)] if match)
        if not indices or indices != list(range(1, len(indices) + 1)):
            raise ValueError("Missing or non-contiguous Stage{} modules".format(stage))
        modules[stage] = len(indices)

    block_groups = {}
    for name in by_name:
        match = re.match(r"^(stage2_module\d+_branch[01])_bb(\d+)_conv1$", name)
        if match:
            block_groups.setdefault(match.group(1), []).append(int(match.group(2)))
    block_counts = {name: len(indices) for name, indices in sorted(block_groups.items())}
    unique_counts = sorted(set(block_counts.values()))
    stage_widths = {}
    for stage, count in modules.items():
        stage_widths[str(stage)] = [
            by_name["stage{}_module{}_branch{}_bb1_conv1".format(stage, count, branch)]["config"]["filters"]
            for branch in range(stage)
        ]

    transitions = {}
    fusions = {}
    decoder = {}
    for entry in graph_layers:
        name = entry["name"]
        if re.match(r"^transition2_(?:new_branch1|high_branch0)_conv$", name):
            transitions[name] = _edge_names(entry)
        if re.match(r"^stage2_module\d+_fuse_target[01]_sum$", name):
            fusions[name] = _edge_names(entry)
        if re.match(r"^decoder[01]_concat$", name):
            decoder[name] = _edge_names(entry)


    fusion_modes = {}
    fusion_operators = {}
    lateral_edges = {}
    for module_index in range(1, modules[2] + 1):
        prefix = "stage2_module{}".format(module_index)
        edges = []
        for target in (0, 1):
            sum_name = prefix + "_fuse_target{}_sum".format(target)
            entry = by_name[sum_name]
            sources = _edge_names(entry)
            last_bb = block_counts[prefix + "_branch{}".format(target)]
            expected_self = prefix + "_branch{}_bb{}_relu2".format(target, last_bb)
            if not sources or sources[0] != expected_self:
                raise ValueError("Fusion must read its own processed branch exactly once, first")
            if sources.count(expected_self) != 1:
                raise ValueError("Duplicated self contribution in fusion")
            for source_name in sources[1:]:
                match = re.match(re.escape(prefix) + r"_fuse([01])_to([01])_", source_name)
                if not match or int(match.group(2)) != target:
                    raise ValueError("Fusion input is not the expected directed lateral transform")
                source = int(match.group(1))
                if source == target:
                    raise ValueError("Lateral edge must join distinct branches")
                edges.append([source, target])
            if len(sources) == 1:
                if entry["class_name"] != "Activation" or entry["config"].get("activation") != "linear":
                    raise ValueError("A self-only fusion node must be a linear identity")
            elif entry["class_name"] != "Add":
                raise ValueError("A fusion with lateral input must be an Add")
            fusion_operators[sum_name] = entry["class_name"]
        key = tuple(sorted(tuple(edge) for edge in edges))
        mode = {(): "none", ((1, 0),): "low_to_high", ((0, 1),): "high_to_low"}.get(key)
        if mode is None:
            raise ValueError("This ablation requires no exchange or one directed exchange")
        fusion_modes[prefix] = mode
        lateral_edges[prefix] = sorted(edges)
    unique_fusion_modes = sorted(set(fusion_modes.values()))


    bottleneck_indices = sorted(int(match.group(1)) for name in by_name
                                for match in [re.match(r"^bottleneck(\d+)_conv1$", name)] if match)
    if bottleneck_indices != [1, 2]:
        raise ValueError("This model requires two contiguous shared E1 Bottlenecks")
    bottlenecks = {}
    for index in bottleneck_indices:
        prefix = "bottleneck{}".format(index)
        entries = [by_name[prefix + "_conv{}".format(i)] for i in (1, 2, 3)]
        actual_first = live_layers[prefix + "_conv1"]
        shortcut = by_name.get(prefix + "_shortcut")
        hidden = entries[0]["config"]["filters"]
        output = entries[2]["config"]["filters"]
        bottlenecks[prefix] = {
            "input_sources": _edge_names(entries[0]),
            "input_channels": int(actual_first.input.shape[-1]),
            "main_channels": [entry["config"]["filters"] for entry in entries],
            "planes": hidden,
            "expansion": output / hidden,
            "output_channels": output,
            "kernel_sizes": [entry["config"]["kernel_size"] for entry in entries],
            "strides": [entry["config"]["strides"] for entry in entries],
            "shortcut_type": "projection" if shortcut else "identity",
            "shortcut_sources": _edge_names(shortcut) if shortcut else _edge_names(by_name[prefix + "_add"])[:1],
            "add_source_order": _edge_names(by_name[prefix + "_add"]),
            "output_layer": prefix + "_relu3",
        }
    first_transition = {name: transitions[name] for name in
                        ("transition2_high_branch0_conv", "transition2_new_branch1_conv")}
    if any(sources != ["bottleneck2_relu3"] for sources in first_transition.values()):
        raise ValueError("Both first-transition branches must read bottleneck2_relu3")

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
    head = by_name.get("head", {})
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "input_channels": int(model.inputs[0].shape[-1]),
        "stage_modules": [modules[2]],
        "branch_channels": [by_name["stem_conv2"]["config"]["filters"]] + stage_widths["2"],
        "hr_branch_channels": stage_widths["2"],
        "encoder_level_count": 1 + len(stage_widths["2"]),
        "hr_branch_count": len(stage_widths["2"]),
        "hr_stage_indices": sorted(modules),
        "hr_local_to_encoder_levels": [1, 2],
        "raw_e0_source": _edge_names(by_name["decoder0_concat"])[1],
        "raw_e1_source": _edge_names(by_name["decoder1_concat"])[1],
        "e1_entry_source": _edge_names(by_name["encoder1_conv1"]),
        "e1_entry_strides": by_name["encoder1_conv1"]["config"]["strides"],
        "bottleneck_count": len(bottlenecks),
        "bottlenecks": bottlenecks,
        "first_transition_sources": first_transition,
        "stage_branch_channels": stage_widths,
        "blocks_per_branch": unique_counts[0] if len(unique_counts) == 1 else block_counts,
        "basic_block_counts": block_counts,
        "total_basicblocks": sum(block_counts.values()),
        "batch_normalization_layers": [entry["name"] for entry in graph_layers
                                       if entry["class_name"] == "BatchNormalization"],
        "activation_layers": [{"name": entry["name"], "class_name": entry["class_name"],
                               "config": entry["config"]}
                              for entry in graph_layers
                              if entry["class_name"] in ("ReLU", "Activation", "LeakyReLU", "PReLU")],
        "convolution_activations": {entry["name"]: entry["config"]["activation"] for entry in convolution_entries},
        "convolution_bias": {entry["name"]: entry["config"]["use_bias"] for entry in convolution_entries},
        "transition_sources": transitions,
        "fusion_source_order": fusions,
        "fusion_mode": unique_fusion_modes[0] if len(unique_fusion_modes) == 1 else fusion_modes,
        "module_fusion_modes": fusion_modes,
        "fusion_operators": fusion_operators,
        "lateral_edges": lateral_edges,
        "multiscale_concat_sources": _edge_names(by_name["encoder_multiscale_concat"]),
        "multiscale_concat_channels": int(live_layers["encoder_multiscale_concat"].output.shape[-1]),
        "decoder_source_order": decoder,
        "decoder_concat_channels": {name: int(live_layers[name].output.shape[-1]) for name in decoder},
        "transposed_convolutions": transposed_details,
        "upsampling_layers": upsampling_layers,
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
            raise ValueError("Missing architecture contract; raw/legacy/Step06 weights are not checked-compatible")
        try:
            stored = json.loads(_text(handle[CONTRACT_DATASET][()]))
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid architecture contract JSON") from exc
        if not isinstance(stored, dict):
            raise ValueError("Architecture contract must be a JSON object")
        if stored.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError("Architecture contract schema version mismatch")
        if stored.get("model_id") != MODEL_ID or _text(handle.attrs.get("model_id", "")) != MODEL_ID:
            raise ValueError("Architecture model ID mismatch (different ablation/legacy weights rejected)")
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
