from numbers import Integral
from pathlib import Path
import hashlib
import json

from tensorflow.keras import Model, layers


INPUT_SHAPE = (128, 128, 128, 1)
CHANNELS = (32, 64, 128)
MODEL_ID = "wnet_3scale_c32_64_128_enc2_dec1_paper_adaptation_v1"
CHECKPOINT_ATTRIBUTE = "wnet_architecture_sha256"


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
        raise ValueError("Input channel count must be a positive integer")
    if any(v is not None and v % 4 for v in value[:3]):
        raise ValueError("D, H and W must be divisible by 4 (two pooling stages)")
    return tuple(None if v is None else int(v) for v in value)


def _conv_relu(x, channels, name):
    x = layers.Conv3D(
        channels, 3, strides=1, padding="same", data_format="channels_last",
        use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros",
        name=name + "_conv",
    )(x)
    return layers.ReLU(name=name + "_relu")(x)


def _upsample(x, name):

    return layers.UpSampling3D(size=2, data_format="channels_last", name=name)(x)


def build_wnet3d(input_shape=INPUT_SHAPE, *, pretrained_weights=None):

    inputs = layers.Input(shape=_input_shape(input_shape), name="input_volume")
    encoders = []
    x = inputs
    for i, width in enumerate(CHANNELS):
        if i:
            x = layers.MaxPooling3D(
                pool_size=2, strides=2, padding="valid", data_format="channels_last",
                name="encoder{}_pool".format(i),
            )(x)
        x = _conv_relu(x, width, "encoder{}_1".format(i))
        x = _conv_relu(x, width, "encoder{}_2".format(i))
        encoders.append(x)


    a = encoders[-1]
    left_decoder = {}
    for i in (1, 0):
        a = _upsample(a, "decoder_a{}_up".format(i))
        a = layers.Concatenate(axis=-1, name="decoder_a{}_concat".format(i))([encoders[i], a])
        a = _conv_relu(a, CHANNELS[i], "decoder_a{}".format(i))
        left_decoder[i] = a


    b = _conv_relu(encoders[-1], CHANNELS[-1], "decoder_b2_seed")
    for i in (1, 0):
        semantic_skip = layers.Add(name="semantic_skip{}_add".format(i))(
            [encoders[i], left_decoder[i]])
        b = _upsample(b, "decoder_b{}_up".format(i))
        b = layers.Concatenate(axis=-1, name="decoder_b{}_concat".format(i))([semantic_skip, b])
        b = _conv_relu(b, CHANNELS[i], "decoder_b{}".format(i))


    outputs = layers.Conv3D(
        1, 1, padding="same", activation="sigmoid", data_format="channels_last",
        dtype="float32", name="fault_probability",
    )(b)
    model = Model(inputs, outputs, name=MODEL_ID)
    if pretrained_weights is not None:
        load_weights_checked(model, pretrained_weights)
    return model


build_unet3d = build_wnet3d  
build_model = build_wnet3d


def wnet(pretrained_weights=None, input_size=INPUT_SHAPE):
    return build_wnet3d(input_size, pretrained_weights=pretrained_weights)


def unet(pretrained_weights=None, input_size=INPUT_SHAPE):

    return wnet(pretrained_weights=pretrained_weights, input_size=input_size)


def intermediate_model(model):

    names = ["encoder{}_2_relu".format(i) for i in range(3)]
    names += ["decoder_b2_seed_relu"]
    for i in (1, 0):
        names += ["decoder_a{}_concat".format(i), "decoder_a{}_relu".format(i),
                  "semantic_skip{}_add".format(i), "decoder_b{}_concat".format(i),
                  "decoder_b{}_relu".format(i)]
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
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def save_weights_checked(model, path):

    import h5py
    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Use an .h5 or .hdf5 weights filename")
    signature = architecture_sha256(model)
    model.save_weights(str(path))
    with h5py.File(str(path), "a") as f:
        f.attrs[CHECKPOINT_ATTRIBUTE] = signature
        f.attrs["wnet_model_id"] = MODEL_ID


def load_weights_checked(model, path):

    import h5py
    with h5py.File(str(path), "r") as f:
        signature = f.attrs.get(CHECKPOINT_ATTRIBUTE)
    if isinstance(signature, bytes):
        signature = signature.decode("ascii")
    if signature is None:
        raise ValueError("Missing architecture signature; use save_weights_checked")
    if signature != architecture_sha256(model):
        raise ValueError("Checkpoint architecture mismatch")
    model.load_weights(str(path))
    return model


__all__ = ["INPUT_SHAPE", "CHANNELS", "MODEL_ID", "build_wnet3d", "build_unet3d",
           "build_model", "wnet", "unet", "intermediate_model", "architecture_sha256",
           "save_weights_checked", "load_weights_checked"]
