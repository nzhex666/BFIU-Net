import hashlib
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

from tensorflow.keras import Model, layers


INPUT_SHAPE = (128, 128, 128, 1)
BRANCH_CHANNELS = (32, 64, 128)
HR_BRANCH_CHANNELS = BRANCH_CHANNELS[1:]
BLOCKS_PER_BRANCH = 2
BOTTLENECK_BLOCKS = 2
BOTTLENECK_PLANES = 64
BOTTLENECK_EXPANSION = 4
DEFAULT_STAGE_MODULES = (1,)
MODEL_ID = "bfiu_net_e1start_e0skip_c32_64_128_bneck2_directtconv_nobn_v3"


def _local_module(name):

    directory = Path(__file__).resolve().parent
    key = "_hrnet_e1start_e0skip_base32_3level_v1_" + hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:20]
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
        raise ValueError("Each spatial dimension must be >= 4 and divisible by 4")
    return values


def _validate_stage_modules(stage_modules):
    try:
        values = tuple(stage_modules)
    except TypeError as exc:
        raise ValueError("stage_modules must be a one-element positive integer sequence, e.g. (1,)") from exc
    if len(values) != 1:
        raise ValueError("This three-level model has only Stage2; stage_modules must contain exactly one count, e.g. (1,), not (1,1)")
    return tuple(_blocks.positive_integer(v, "stage_modules[{}]".format(i))
                 for i, v in enumerate(values))


def build_unet3d(input_shape=INPUT_SHAPE, *, stage_modules=DEFAULT_STAGE_MODULES,
                 pretrained_weights=None):

    input_shape = _validate_input_shape(input_shape)
    stage_modules = _validate_stage_modules(stage_modules)
    inputs = layers.Input(shape=input_shape, name="input_volume")
    stem = _blocks.double_conv(inputs, BRANCH_CHANNELS[0], name="stem")


    e1 = _blocks.conv3d(stem, BRANCH_CHANNELS[1], strides=2, name="encoder1_conv1")
    e1 = layers.ReLU(name="encoder1_relu1")(e1)
    e1 = _blocks.conv3d(e1, BRANCH_CHANNELS[1], name="encoder1_conv2")
    e1 = layers.ReLU(name="encoder1_relu2")(e1)
    shared = e1
    for index in range(1, BOTTLENECK_BLOCKS + 1):
        shared = _blocks.bottleneck_block(
            shared, BOTTLENECK_PLANES, expansion=BOTTLENECK_EXPANSION,
            name="bottleneck{}".format(index)
        )


    high = _blocks.conv3d(shared, HR_BRANCH_CHANNELS[0], strides=1,
                         name="transition2_high_branch0_conv")
    high = layers.ReLU(name="transition2_high_branch0_relu")(high)
    low = _blocks.conv3d(shared, HR_BRANCH_CHANNELS[1], strides=2,
                        name="transition2_new_branch1_conv")
    low = layers.ReLU(name="transition2_new_branch1_relu")(low)
    branches = [high, low]
    for module in range(1, stage_modules[0] + 1):
        branches = _fusion.hr_module(
            branches, HR_BRANCH_CHANNELS,
            name="stage2_module{}".format(module),
            blocks_per_branch=BLOCKS_PER_BRANCH
        )


    aligned_h2 = _blocks.conv_transpose3d(
        branches[1], HR_BRANCH_CHANNELS[1], name="align_h2_deconv")
    aggregated = layers.Concatenate(axis=-1, name="encoder_multiscale_concat")(
        [branches[0], aligned_h2])

    x = layers.Concatenate(axis=-1, name="decoder1_concat")([aggregated, e1])
    x = _blocks.conv_transpose3d(x, BRANCH_CHANNELS[0], name="decoder1_deconv")
    x = layers.ReLU(name="decoder1_relu")(x)

    x = layers.Concatenate(axis=-1, name="decoder0_concat")([x, stem])
    x = _blocks.double_conv(x, BRANCH_CHANNELS[0], name="decoder0")
    outputs = _blocks.conv3d(x, 1, kernel_size=3, name="head", dtype="float32",
                             activation="sigmoid")
    model = Model(inputs=inputs, outputs=outputs, name=MODEL_ID)
    if pretrained_weights is not None:
        load_weights_checked(model, pretrained_weights)
    return model


build_hrnet_unet3d = build_unet3d


def unet(pretrained_weights=None, input_size=INPUT_SHAPE,
         stage_modules=DEFAULT_STAGE_MODULES):

    return build_unet3d(input_size, stage_modules=stage_modules,
                        pretrained_weights=pretrained_weights)


__all__ = [
    "INPUT_SHAPE", "BRANCH_CHANNELS", "HR_BRANCH_CHANNELS", "BLOCKS_PER_BRANCH", "DEFAULT_STAGE_MODULES",
    "BOTTLENECK_BLOCKS", "BOTTLENECK_PLANES", "BOTTLENECK_EXPANSION",
    "MODEL_ID", "build_unet3d", "build_hrnet_unet3d", "unet",
    "get_architecture_contract", "save_weights_checked", "load_weights_checked",
]
