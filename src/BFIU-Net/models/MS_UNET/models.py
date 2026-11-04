from numbers import Integral
from pathlib import Path
import hashlib
import json

from tensorflow.keras import Model, layers


INPUT_SHAPE = (128, 128, 128, 1)
CHANNELS = (32, 64, 128)
MODEL_ID = "ms_unet_3scale_c32_64_128_paper_adaptation_v1"
CHECKPOINT_ATTRIBUTE = "ms_unet_architecture_sha256"


def _input_shape(value):
    try:
        value = tuple(value)
    except TypeError as exc:
        raise ValueError("input_shape must be (D, H, W, C)") from exc
    if len(value) != 4:
        raise ValueError("input_shape must be (D, H, W, C)")
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, Integral) or v <= 0)
           for v in value):
        raise ValueError("input_shape entries must be positive integers or spatial None")
    if value[-1] is None:
        raise ValueError("The number of input channels must be a positive integer")
    if any(v is not None and v % 4 for v in value[:3]):
        raise ValueError("D, H and W must be divisible by 4 (two max-pool stages)")
    return tuple(None if v is None else int(v) for v in value)


def _conv_relu(x, channels, kernel_size, name):
    x = layers.Conv3D(
        channels, kernel_size, strides=1, padding="same",
        data_format="channels_last", use_bias=True,
        kernel_initializer="he_normal", bias_initializer="zeros",
        name=name + "_conv",
    )(x)
    return layers.ReLU(name=name + "_relu")(x)


def _double_conv(x, channels, name):
    x = _conv_relu(x, channels, 3, name + "_1")
    return _conv_relu(x, channels, 3, name + "_2")


def _fused_skip(encoder_features, target):

    width = CHANNELS[target]
    aligned = []
    for source, feature in enumerate(encoder_features):
        prefix = "skip{}_from{}".format(target, source)
        x = feature
        if source < target:
            factor = 2 ** (target - source)
            x = layers.MaxPooling3D(
                pool_size=factor, strides=factor, padding="valid",
                data_format="channels_last", name=prefix + "_down",
            )(x)
        elif source > target:
            factor = 2 ** (source - target)

            x = layers.UpSampling3D(
                size=factor, data_format="channels_last", name=prefix + "_up",
            )(x)
        if source != target:
            x = _conv_relu(x, width, 3, prefix + "_align")
        aligned.append(x)
    x = layers.Concatenate(axis=-1, name="skip{}_concat".format(target))(aligned)
    return _conv_relu(x, width, 1, "skip{}_project".format(target))


def build_ms_unet3d(input_shape=INPUT_SHAPE, *, pretrained_weights=None):

    inputs = layers.Input(shape=_input_shape(input_shape), name="input_volume")
    features = []
    x = inputs
    for level, width in enumerate(CHANNELS):
        if level:
            x = layers.MaxPooling3D(
                pool_size=2, strides=2, padding="valid",
                data_format="channels_last", name="encoder{}_pool".format(level),
            )(x)
        x = _double_conv(x, width, "encoder{}".format(level))
        features.append(x)

    x = features[-1]
    for target in (1, 0):
        skip = _fused_skip(features, target)
        x = layers.UpSampling3D(
            size=2, data_format="channels_last", name="decoder{}_up".format(target),
        )(x)
        x = layers.Concatenate(axis=-1, name="decoder{}_concat".format(target))([skip, x])
        x = _double_conv(x, CHANNELS[target], "decoder{}".format(target))
    outputs = layers.Conv3D(
        1, 1, padding="same", activation="sigmoid", data_format="channels_last",
        dtype="float32", name="fault_probability",
    )(x)
    model = Model(inputs, outputs, name=MODEL_ID)
    if pretrained_weights is not None:
        load_weights_checked(model, pretrained_weights)
    return model


build_unet3d = build_ms_unet3d
build_model = build_ms_unet3d


def unet(pretrained_weights=None, input_size=INPUT_SHAPE):

    return build_ms_unet3d(input_size, pretrained_weights=pretrained_weights)


def intermediate_model(model):

    names = ["encoder{}_2_relu".format(i) for i in range(3)]
    names += ["skip{}_concat".format(i) for i in (0, 1)]
    names += ["skip{}_project_relu".format(i) for i in (0, 1)]
    names += ["decoder{}_concat".format(i) for i in (0, 1)]
    names += ["decoder{}_2_relu".format(i) for i in (0, 1)]
    names += ["fault_probability"]
    return Model(model.inputs, {n: model.get_layer(n).output for n in names},
                 name=MODEL_ID + "_intermediates")


def architecture_sha256(model):

    config = json.loads(model.to_json())["config"]
    config.pop("name", None)
    for entry in config["layers"]:
        if entry["class_name"] == "InputLayer":
            for key in ("batch_input_shape", "batch_shape"):
                if key in entry["config"]:
                    old = entry["config"][key]
                    entry["config"][key] = [None, None, None, None, old[-1]]
    payload = {"graph": config, "weight_shapes": [w.shape.as_list() for w in model.weights]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_weights_checked(model, path):

    import h5py
    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Use an .h5 or .hdf5 weights filename")
    signature = architecture_sha256(model)
    model.save_weights(str(path))
    with h5py.File(str(path), "a") as f:
        f.attrs[CHECKPOINT_ATTRIBUTE] = signature
        f.attrs["ms_unet_model_id"] = MODEL_ID


def load_weights_checked(model, path):

    import h5py
    with h5py.File(str(path), "r") as f:
        stored = f.attrs.get(CHECKPOINT_ATTRIBUTE)
    if isinstance(stored, bytes):
        stored = stored.decode("ascii")
    if stored is None:
        raise ValueError("Missing architecture signature; use save_weights_checked")
    if stored != architecture_sha256(model):
        raise ValueError("Checkpoint architecture mismatch")
    model.load_weights(str(path))
    return model


__all__ = ["INPUT_SHAPE", "CHANNELS", "MODEL_ID", "build_ms_unet3d",
           "build_unet3d", "build_model", "unet", "intermediate_model",
           "architecture_sha256", "save_weights_checked", "load_weights_checked"]
