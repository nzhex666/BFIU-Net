from tensorflow.keras import layers


def same_scale_concat(decoder, skip, *, name):

    if decoder.shape.rank != 5 or skip.shape.rank != 5:
        raise ValueError("Expected rank-5 NDHWC decoder and skip tensors")
    if tuple(decoder.shape[1:4]) != tuple(skip.shape[1:4]):
        raise ValueError("Decoder and raw skip must have the same spatial shape")
    return layers.Concatenate(axis=-1, name=name)([decoder, skip])


def nearest_skip_concat(decoder, skip, *, name):

    decoder = layers.UpSampling3D(size=(2, 2, 2), data_format="channels_last",
                                  name=name + "_nearest")(decoder)
    return same_scale_concat(decoder, skip, name=name + "_concat")


__all__ = ["same_scale_concat", "nearest_skip_concat"]
