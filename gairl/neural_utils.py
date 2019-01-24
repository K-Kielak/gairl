from collections import namedtuple
from functools import reduce

import tensorflow as tf


DenseLayerParams = namedtuple('DenseLayerParams', 'weights biases')


class DenseNetworkUtils:

    @staticmethod
    def unpack_params(params):
        return reduce(lambda x, y: x.extend(list(y)) or list(x), params, [])

    @staticmethod
    def create_network_params(input_shape, hidden_layers, outputs_num, dtype,
                              trainable=True, name=None):
        """
        Crates tensorflow variables for dense feedforward neural network.
        :param input_shape: tuple of ints; shape of the input to the network.
        :param hidden_layers: tuple of ints; number of nodes in each hidden
            layer of the network.
        :param outputs_num: int; number of outputs network should produce.
        :param dtype: numpy.Dtype; what type should the params have.
        :param trainable: bool; will these params be trainable, i.e.
            if they can be updated by tensorflow training algorithms.
        :param name: string; name scope under which variables will be created.
        :return: list of gairl..DenseLayerParams; tensorflow variables
            representing network weights and biases.
        """
        if not hidden_layers:
            raise AttributeError('DQN has to have some hidden layers!')

        params = []
        layers = [input_shape] + hidden_layers + [outputs_num]
        for i in range(1, len(layers)):
            with tf.name_scope(f'{name}-layer{i}'):
                weights = gen_rand_weights((layers[i - 1], layers[i]), dtype,
                                           trainable=trainable)
                biases = gen_rand_biases((layers[i],), dtype,
                                         trainable=trainable)
                params.append(DenseLayerParams(weights, biases))

        return params

    @staticmethod
    def model_output(input, params, activation_fn, name=None):
        """
        :param input: tf.placeholder; placeholder for network input
        :param params: gairl..DenseLayerParams; tensorflow variables
            representing network weights and biases.
        :param activation_fn: function applied to result of each hidden
            layer of the network.
        :param name: string; name of the resulting tensor
        :return: Final output of the network as a tensorflow tensor
        """
        layer_sum = tf.matmul(input, params[0].weights) + params[0].biases
        activation = activation_fn(layer_sum)

        # Up to len(params) - 1 because last layer doesn't use activation_fn
        for i in range(1, len(params) - 1):
            layer_sum = tf.matmul(activation, params[i].weights) + params[
                i].biases
            activation = activation_fn(layer_sum)

        final_mul = tf.matmul(activation, params[-1].weights)
        return tf.add(final_mul, params[-1].biases, name=name)


def gen_rand_weights(size, dtype, trainable=True):
    distribution = tf.truncated_normal(size, stddev=0.00001, dtype=dtype)
    return tf.Variable(distribution, trainable=trainable, name='weights')


def gen_rand_biases(size, dtype, trainable=True):
    distribution = tf.truncated_normal(size, stddev=0.00001,
                                       mean=0.00003, dtype=dtype)
    return tf.Variable(distribution, trainable=trainable, name='biases')


def create_copy_ops(from_list, to_list):
    return [to_list[i].assign(var) for i, var in enumerate(from_list)]


def summarize_ndarray(arr):
    """
    :param arr: N-dimensional tensor variable
    :return: Summaries containing detailed statistics about the variable
    """
    summaries = []
    mean = tf.reduce_mean(arr)
    summaries.append(tf.summary.scalar('mean', mean))

    stddev = tf.sqrt(tf.reduce_mean(tf.square(arr - mean)))
    summaries.append(tf.summary.scalar('stddev', stddev))
    summaries.append(tf.summary.scalar('max', tf.reduce_max(arr)))
    summaries.append(tf.summary.scalar('min', tf.reduce_min(arr)))
    summaries.append(tf.summary.histogram('histogram', arr))
    return summaries


def summarize_vector(vec):
    """
    :param vec: 1-dimensional tensor variable
    :return: Summaries containing detailed statistics about the variable
    """

    summaries = []
    vector_shape = vec.get_shape().as_list()[0]
    print(vec)
    print(vec.get_shape())
    print(vec.get_shape().as_list())
    for i in range(0, vector_shape):
        summaries.append(tf.summary.scalar(str(i), tf.gather(vec, i)))

    summaries.append(tf.summary.scalar('max', tf.reduce_max(vec)))
    summaries.append(tf.summary.scalar('mean', tf.reduce_mean(vec)))
    return summaries
