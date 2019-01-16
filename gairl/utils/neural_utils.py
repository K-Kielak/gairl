from collections import namedtuple

import tensorflow as tf


DenseLayerParams = namedtuple('DenseLayerParams', 'weights biases')


def gen_rand_weights(size, dtype, trainable=True):
    distribution = tf.truncated_normal(size, stddev=0.00001, dtype=dtype)
    return tf.Variable(distribution, trainable=trainable)


def gen_rand_biases(size, dtype, trainable=True):
    distribution = tf.truncated_normal(size, stddev=0.00001,
                                       mean=0.00003, dtype=dtype)
    return tf.Variable(distribution, trainable=trainable)
