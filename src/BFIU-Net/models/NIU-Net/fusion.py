from tensorflow.keras import layers

from .blocks import basic_block, conv3d, conv_transpose3d, positive_integer


def transform_edge(source, source_index, target_index, channels, *, name):

    if source_index == target_index:
        return source
    if source_index > target_index:
        x = conv3d(source, channels[target_index], kernel_size=1,
                   name=name + "_project")
        for step in range(1, source_index - target_index + 1):
            x = conv_transpose3d(x, channels[target_index],
                                 name=name + "_up{}_deconv".format(step))
        return x
    x = source
    steps = target_index - source_index
    for step in range(1, steps + 1):
        final = step == steps
        width = channels[target_index] if final else channels[source_index]
        x = conv3d(x, width, strides=2,
                   name=name + "_down{}_conv".format(step))
        if not final:
            x = layers.ReLU(name=name + "_down{}_relu".format(step))(x)
    return x


def hr_module(branches, channels, *, name, blocks_per_branch=2, fusion_mode="none"):

    allowed_modes = {
        "none": (),
        "low_to_high": ((1, 0),),
        "high_to_low": ((0, 1),),
    }
    if fusion_mode not in allowed_modes:
        raise ValueError("fusion_mode must be none, low_to_high, or high_to_low")
    retained_edges = allowed_modes[fusion_mode]
    blocks_per_branch = positive_integer(blocks_per_branch, "blocks_per_branch")
    if blocks_per_branch != 2:
        raise ValueError("This architecture fixes blocks_per_branch=2")
    channels = tuple(positive_integer(c, "branch channel") for c in channels)
    if len(branches) != len(channels) or len(branches) != 2:
        raise ValueError("This model's HR module requires exactly two branches/widths")


    processed = []
    for index, (x, width) in enumerate(zip(branches, channels)):
        for block in range(1, blocks_per_branch + 1):
            x = basic_block(x, width, name="{}_branch{}_bb{}".format(name, index, block))
        processed.append(x)
    u = tuple(processed)

    fused = []
    for target in range(len(u)):

        terms = [u[target]]
        for source in range(len(u)):
            if (source, target) in retained_edges:
                terms.append(transform_edge(
                    u[source], source, target, channels,
                    name="{}_fuse{}_to{}".format(name, source, target)
                ))
        sum_name = "{}_fuse_target{}_sum".format(name, target)
        if len(terms) == 1:
            summed = layers.Activation("linear", name=sum_name)(terms[0])
        else:
            summed = layers.Add(name=sum_name)(terms)
        fused.append(layers.ReLU(name="{}_fuse_target{}_relu".format(name, target))(summed))
    return fused


__all__ = ["hr_module", "transform_edge"]
