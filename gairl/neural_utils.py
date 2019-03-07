from collections import namedtuple
from functools import reduce

import tensorflow as tf


DenseLayerParams = namedtuple('DenseLayerParams', 'weights biases')


class DenseNetworkUtils:

    @staticmethod
    def unpack_params(params):
        return reduce(lambda x, y: x.extend(list(y)) or list(x), params, [])

    @staticmethod
    def create_network_params(input_shape, hidden_layers,
                              outputs_num, dtype,
                              trainable=True, name=None,
                              stddev=1e-5, mean=3e-5):
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
        :param mean: float; mean for the biases distribution.
        :param stddev: float; standard deviation for both weights
            and biases distribution.
        :return: list of gairl..DenseLayerParams; tensorflow variables
            representing network weights and biases.
        """
        params = []
        layers = [input_shape] + list(hidden_layers) + [outputs_num]
        for i in range(1, len(layers)):
            with tf.name_scope(f'{name}_layer{i}'):
                weights = gen_rand_weights((layers[i - 1], layers[i]), dtype,
                                           trainable=trainable, stddev=stddev)
                biases = gen_rand_biases((layers[i],), dtype,
                                         trainable=trainable,
                                         stddev=stddev, mean=mean)
                params.append(DenseLayerParams(weights, biases))

        return params

    @staticmethod
    def model_output(input, params, activation_fn,
                     dropout_prob=1, out_activation_fn=None, name=None):
        """
        :param input: tf.placeholder; placeholder for network input
        :param params: gairl..DenseLayerParams; tensorflow variables
            representing network weights and biases.
        :param activation_fn: function applied to result of each hidden
            layer of the network.
        :param dropout_prob: probability of dropout for the hidden layers.
        :param out_activation_fn: activation function used on the
            network's output, linear if None.
        :param name: string; name of the resulting tensor
        :return: Final output of the network as a tensorflow tensor
        """
        if len(params) == 1:
            mul = tf.matmul(input, params[0].weights)
            out = tf.add(mul, params[0].biases, name=name)
        else:
            layer_sum = tf.matmul(input, params[0].weights) + params[0].biases
            layer_sum = tf.nn.dropout(layer_sum, dropout_prob)
            activation = activation_fn(layer_sum)

            # Up to len(params) - 1 cause last layer uses different activation
            for i in range(1, len(params) - 1):
                layer_sum = tf.matmul(activation, params[i].weights) + params[i].biases
                layer_sum = tf.nn.dropout(layer_sum, dropout_prob)
                activation = activation_fn(layer_sum)

            final_mul = tf.matmul(activation, params[-1].weights)
            out = tf.add(final_mul, params[-1].biases, name=name)

        if out_activation_fn:
            out = out_activation_fn(out)

        return out


def gen_rand_weights(size, dtype, trainable=True, stddev=1e-5):
    distribution = tf.truncated_normal(size, stddev=stddev, dtype=dtype)
    return tf.Variable(distribution, trainable=trainable, name='weights')


def gen_rand_biases(size, dtype, trainable=True, stddev=1e-5, mean=3e-5):
    distribution = tf.truncated_normal(size, dtype=dtype,
                                       stddev=stddev, mean=mean)
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
    for i in range(0, vector_shape):
        summaries.append(tf.summary.scalar(str(i), tf.gather(vec, i)))

    summaries.append(tf.summary.scalar('max', tf.reduce_max(vec)))
    summaries.append(tf.summary.scalar('mean', tf.reduce_mean(vec)))
    return summaries


def normalize(data_batch, data_ranges=None,
              target_ranges=(0, 1), name=None, dtype=tf.float32):
    """
    :param data_batch: tensor with data to normalize
    :param data_ranges: list of tuples of floats; specifies what
            is the range of data in terms of max and min values.
            If single tuple then applies single range to whole data,
            if multiple then for each feature separately.
    :param target_ranges: list of tuples of floats; specifies what
            is the target range of data in terms of max and min values.
            If single tuple then applies single range to whole data,
            if multiple then for each feature separately.
    :param name: name for the final output tensor.
    :param dtype: dtype used for normalization.
    :return: Normalized tensor containing the data
    """
    if data_ranges is None:
        data_ranges = tf.stack((tf.reduce_min(data_batch),
                               tf.reduce_max(data_batch)))

    # Make sure data ranges are appropriately shaped
    data_ranges = tf.convert_to_tensor(data_ranges, dtype=dtype)
    lacking_dims = tf.tile([1], [tf.rank(data_batch) - tf.rank(data_ranges)])
    final_shape = tf.concat((lacking_dims, tf.shape(data_ranges)), 0)
    data_ranges = tf.reshape(data_ranges, final_shape)

    # Make sure target ranges are appropriately shaped
    target_ranges = tf.convert_to_tensor(target_ranges, dtype=dtype)
    lacking_dims = tf.tile([1], [tf.rank(data_batch) - tf.rank(target_ranges)])
    final_shape = tf.concat((lacking_dims, tf.shape(target_ranges)), 0)
    target_ranges = tf.reshape(target_ranges, final_shape)

    data_normed = (data_batch - data_ranges[..., 0])
    data_normed = data_normed / (data_ranges[..., 1] - data_ranges[..., 0])
    data_normed = data_normed * (target_ranges[..., 1] - target_ranges[..., 0])
    return tf.add(data_normed, target_ranges[..., 0], name=name)
