EPSILON = 1e-7


def make_loss(name):

    if name not in ("bce", "balanced_bce"):
        raise ValueError("loss must be bce or balanced_bce")
    import tensorflow as tf

    def probability_loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        predictions = tf.cast(y_pred, tf.float32)
        predictions = tf.clip_by_value(predictions, EPSILON, 1.0 - EPSILON)
        positive = -labels * tf.math.log(predictions)
        negative = -(1.0 - labels) * tf.math.log1p(-predictions)
        if name == "balanced_bce":
            beta = tf.reduce_mean(1.0 - labels)
            return tf.reduce_mean(beta * positive + (1.0 - beta) * negative)
        return tf.reduce_mean(positive + negative)

    probability_loss.__name__ = name
    return probability_loss
