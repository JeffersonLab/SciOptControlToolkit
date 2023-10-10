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
    score = tf.abs((score1 - score2)/score2)
    #score = 2 * score1 - score2
    return score, score1/score2, score2/score2
    #return score, score1, score2

@tf.function
def get_score_2d(training_actions, top_actions, num_actions=2):
    training_actions = tf.split(training_actions, num_or_size_splits=num_actions, axis=1)
    training_actions0, training_actions1 = tf.squeeze(training_actions[0]), tf.squeeze(training_actions[1])
    sorted_indices = tf.argsort(tf.math.atan2(training_actions1, training_actions0))
    training_actions0 = tf.gather(training_actions0, sorted_indices) #tf.sort(training_actions0)
    training_actions1 = tf.gather(training_actions1, sorted_indices) #tf.sort(training_actions1)
    scores0 = tf.math.reduce_sum(tf.math.abs(training_actions0 - top_actions[0]))
    scores1 = tf.math.reduce_sum(tf.math.abs(training_actions1 - top_actions[1]))
    print('self.scores[0]:', scores0)
    print('self.scores[1]:', scores1)
    score = scores0+scores1
    return score, 0, 0

@tf.function
def get_score_1d(training_actions, top_actions, num_actions=1):
    sorted_training = tf.sort(training_actions)
    sorted_top_actions = tf.sort(top_actions)
    score = tf.math.reduce_sum(tf.math.abs(sorted_training - sorted_top_actions))

    return score
# @tf.function
# def split_2d(training_actions, num_actions=2):
    
#     return training_actions

# @tf.function
# def split_1d(training_actions, num_actions=1)
#     return training_actions


# @tf.function
# def get_score_1d(pred:tf.Tensor, obs:tf.Tensor) -> tf.Tensor:
#     # Difference between top actions
#     # score2 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.square(
#     #     tf.expand_dims(obs, axis=1) - tf.expand_dims(obs, axis=0)), axis=-1)))

#     # # Difference between current and top action
#     # score1 = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.square(
#     #     tf.expand_dims(pred, axis=1) - tf.expand_dims(obs, axis=0)), axis=-1)))

#     # Total score
#     # score1 = score1
#     # score2 = score2
#     # score = 2 * score1 - score2
#     score = tf.reduce_mean(tf.math.square(tf.sort(pred) - tf.sort(obs)))
#     return score, -1, -1