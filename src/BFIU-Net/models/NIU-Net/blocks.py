from numbers import Integral

from tensorflow.keras import layers


def positive_integer(value, name):

    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("{} must be a positive integer, not bool; got {!r}".format(name, value))
    return int(value)


def conv3d(x, channels, kernel_size=3, strides=1, *, name, dtype=None,
           activation=None):

    channels = positive_integer(channels, "channels")
    kernel_size = positive_integer(kernel_size, "kernel_size")
    strides = positive_integer(strides, "strides")
    kwargs = {} if dtype is None else {"dtype": dtype}
    return layers.Conv3D(
        channels, kernel_size, strides=strides, padding="same",
        data_format="channels_last", dilation_rate=1, groups=1,
        activation=activation, use_bias=True,
        kernel_initializer="glorot_uniform", bias_initializer="zeros",
        name=name, **kwargs
    )(x)


def double_conv(x, channels, *, name):

    channels = positive_integer(channels, "channels")
    x = conv3d(x, channels, name=name + "_conv1")
    x = layers.ReLU(name=name + "_relu1")(x)
    x = conv3d(x, channels, name=name + "_conv2")
    return layers.ReLU(name=name + "_relu2")(x)


def conv_transpose3d(x, channels, *, name, dtype=None, activation=None):

    channels = positive_integer(channels, "channels")
    kwargs = {} if dtype is None else {"dtype": dtype}
    return layers.Conv3DTranspose(
        channels, kernel_size=4, strides=2, padding="same", output_padding=None,
        data_format="channels_last", dilation_rate=1, activation=activation,
        use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros",
        name=name, **kwargs
    )(x)


def basic_block(x, channels, *, name):

    channels = positive_integer(channels, "channels")
    if x.shape.rank != 5 or x.shape[-1] != channels:
        raise ValueError("BasicBlock requires NDHWC input with unchanged width {}".format(channels))
    residual = conv3d(x, channels, name=name + "_conv1")
    residual = layers.ReLU(name=name + "_relu1")(residual)
    residual = conv3d(residual, channels, name=name + "_conv2")
    x = layers.Add(name=name + "_add")([x, residual])
    return layers.ReLU(name=name + "_relu2")(x)


def bottleneck_block(x, planes, *, expansion=4, name):

    planes = positive_integer(planes, "planes")
    expansion = positive_integer(expansion, "expansion")
    if x.shape.rank != 5 or x.shape[-1] is None:
        raise ValueError("Bottleneck requires NDHWC input with known channel width")
    channels = planes * expansion
    shortcut = x
    residual = conv3d(x, planes, kernel_size=1, name=name + "_conv1")
    residual = layers.ReLU(name=name + "_relu1")(residual)
    residual = conv3d(residual, planes, kernel_size=3, name=name + "_conv2")
    residual = layers.ReLU(name=name + "_relu2")(residual)
    residual = conv3d(residual, channels, kernel_size=1, name=name + "_conv3")
    if int(x.shape[-1]) != channels:
        shortcut = conv3d(x, channels, kernel_size=1, name=name + "_shortcut")
    x = layers.Add(name=name + "_add")([shortcut, residual])
    return layers.ReLU(name=name + "_relu3")(x)


def transition(branches, channels, *, stage):

    stage = positive_integer(stage, "stage")
    channels = positive_integer(channels, "channels")
    if not branches:
        raise ValueError("transition requires at least one existing branch")
    name = "transition{}_new_branch{}".format(stage, len(branches))
    new_branch = conv3d(branches[-1], channels, strides=2, name=name + "_conv")
    new_branch = layers.ReLU(name=name + "_relu")(new_branch)
    return list(branches) + [new_branch]


__all__ = ["positive_integer", "conv3d", "conv_transpose3d", "double_conv", "basic_block",
           "bottleneck_block", "transition"]
