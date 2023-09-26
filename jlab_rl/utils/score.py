import tensorflow as tf

@tf.function
def get_score(pred:tf.Tensor, obs:tf.Tensor) -> tf.Tensor:
    # Difference between top actions
    score2 = tf.reduce_sum(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
        tf.expand_dims(obs, axis=1), tf.expand_dims(obs, axis=0)), axis=-1))) / (obs.shape[0]*(obs.shape[0]-1))

    # Difference between current and top action
    score1 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(
        tf.expand_dims(pred, axis=1), tf.expand_dims(obs, axis=0)), axis=-1)))

    # Total score
    score1 = score1
    score2 = score2
    score = 2 * score1 - score2
    return score, score1, score2


@tf.function
def get_score_1d(pred:tf.Tensor, obs:tf.Tensor) -> tf.Tensor:
    # Difference between top actions
    # score2 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.square(
    #     tf.expand_dims(obs, axis=1) - tf.expand_dims(obs, axis=0)), axis=-1)))

    # # Difference between current and top action
    # score1 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.square(
    #     tf.expand_dims(pred, axis=1) - tf.expand_dims(obs, axis=0)), axis=-1)))

    # Total score
    # score1 = score1
    # score2 = score2
    # score = 2 * score1 - score2
    score = tf.reduce_mean(tf.math.square(tf.sort(pred) - tf.sort(obs)))
    return score, -1, -1