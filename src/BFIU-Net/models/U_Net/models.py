import hashlib
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

from tensorflow.keras import Model, layers


INPUT_SHAPE = (128, 128, 128, 1)
BRANCH_CHANNELS = (32, 64, 128)
MODEL_ID = "c0_plain_unet_3scale_c32_64_128_stride2_nearest_transpose_nobn_relu_v1"


def _local_module(name):

    directory = Path(__file__).resolve().parent
    key = "_c0_plain_unet3d_" + hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:20]
    if key not in sys.modules:
        package = types.ModuleType(key)
        package.__path__ = [str(directory)]
        package.__package__ = key
        package.__spec__ = importlib.machinery.ModuleSpec(key, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[key] = package
    return importlib.import_module(key + "." + name)


_blocks = _local_module("blocks")
_fusion = _local_module("fusion")
_checkpoints = _local_module("checkpoints")
get_architecture_contract = _checkpoints.get_architecture_contract
save_weights_checked = _checkpoints.save_weights_checked
load_weights_checked = _checkpoints.load_weights_checked


def _validate_input_shape(input_shape):
    try:
        values = tuple(input_shape)
    except TypeError as exc:
        raise ValueError("input_shape must be (D, H, W, C)") from exc
    if len(values) != 4:
        raise ValueError("input_shape must contain exactly (D, H, W, C)")
    values = tuple(_blocks.positive_integer(v, "input_shape[{}]".format(i))
                   for i, v in enumerate(values))
    if any(v < 4 or v % 4 != 0 for v in values[:3]):
        raise ValueError("Each spatial dimension must be >=4 and divisible by4")
    return values


def build_unet3d(input_shape=INPUT_SHAPE, *, pretrained_weights=None):

    input_shape = _validate_input_shape(input_shape)
    inputs = layers.Input(shape=input_shape, name="input_volume")
    e0 = _blocks.double_conv(inputs, BRANCH_CHANNELS[0], name="stem")
    features = [e0]
    x = e0
    for level, channels in enumerate(BRANCH_CHANNELS[1:], start=1):
        prefix = "encoder{}".format(level)
        x = _blocks.conv3d(x, channels, strides=2, name=prefix + "_conv1")
        x = layers.ReLU(name=prefix + "_relu1")(x)
        x = _blocks.conv3d(x, channels, strides=1, name=prefix + "_conv2")
        x = layers.ReLU(name=prefix + "_relu2")(x)
        features.append(x)
    e1, e2 = features[1:]

    x = _fusion.nearest_skip_concat(e2, e1, name="decoder1")
    x = _blocks.double_conv(x, BRANCH_CHANNELS[1], name="decoder1")
    x = _blocks.conv_transpose3d(x, BRANCH_CHANNELS[0], name="decoder0_deconv")
    x = layers.ReLU(name="decoder0_relu")(x)
    x = _fusion.same_scale_concat(x, e0, name="decoder0_concat")
    x = _blocks.double_conv(x, BRANCH_CHANNELS[0], name="decoder0")
    outputs = _blocks.conv3d(x, 1, kernel_size=3, name="head",
                             dtype="float32", activation="sigmoid")
    model = Model(inputs=inputs, outputs=outputs, name=MODEL_ID)
    if pretrained_weights is not None:
        load_weights_checked(model, pretrained_weights)
    return model


build_plain_unet3d = build_unet3d


def unet(pretrained_weights=None, input_size=INPUT_SHAPE):

    return build_unet3d(input_size, pretrained_weights=pretrained_weights)


__all__ = ["INPUT_SHAPE", "BRANCH_CHANNELS", "MODEL_ID", "build_unet3d",
           "build_plain_unet3d", "unet", "get_architecture_contract",
           "save_weights_checked", "load_weights_checked"]
