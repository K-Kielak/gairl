import logging
import os
import sys
from functools import reduce
from operator import mul

import tensorflow as tf

from gairl.generators.abstract_generator import AbstractGenerator
from gairl.neural_utils import DenseNetworkUtils as Dnu
from gairl.neural_utils import normalize, summarize_ndarray


MAX_IMGS_TO_VIS = 10
PARAMS_INIT_STDDEV = 1e-3
PARAMS_INIT_MEAN = 0


# TODO add loading
class MultilayerPerceptron(AbstractGenerator):

    def __init__(self,
                 input_shape,
                 data_shape,
                 session,
                 output_directory,
                 name='MultilayerPerceptron',
                 input_ranges=(-1, 1),
                 output_ranges=(-1, 1),
                 dtype=tf.float64,
                 layers=(256, 512, 1024),
                 activation=tf.nn.leaky_relu,
                 final_activation=tf.nn.tanh,
                 norm_range=(-1, 1),
                 dropout=1,
                 optimizer=tf.train.AdamOptimizer(
                     learning_rate=1e-4,
                     epsilon=1.5e-4
                 ),
                 logging_freq=100,
                 logging_level=logging.INFO,
                 max_checkpoints=5,
                 save_freq=1000):
        """
        Initializes multilayer perceptron conditional generator that
        uses L1 loss as a generation obejctive
        :param data_shape: tuple of int; describes size of the
            conditional input used by the MLP.
        :param data_shape: tuple of int; describes the size of the
            data that MLP is supposed to generate.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the model.
        :param input_ranges: list of tuples of floats; specifies what
            is the range of input that is fed to the network in terms
            of max and min values. If single tuple then applies single
            range to whole data, if multiple then for each feature
            separately.
        :param output_ranges: list of tuples of floats; specifies what
            is the range of output that needs to be generated in terms
            of max and min values. If single tuple then applies single
            range to whole data, if multiple then for each feature
            separately.
        :param dtype: tensorflow.DType; type used for the model.
        :param layers: tuple of ints; describes number of nodes
            in each hidden layer of the network.
        :param activation: activation function for hidden layers
            of the network.
        :param final_activation: activation used at the end of the
            network. None for linear.
        :param norm_range: tuple of ints; range of values to which all the
        values internally should be normalized. Should be compatible
            with the range of values produced by final_activation.
        :param dropout: float in (0, 1]; dropout probability for
            hidden layers of the network.
        :param optimizer: tf.trainOptimizer; optimizer for the network.
        :param logging_freq: int; frequency of progress logging and
            writing tensorflow summaries.
        :param logging_level: logging.LEVEL; level of the internal logger.
            None if it already reuses existing logger.
        :param max_checkpoints: int; number of checkpoints to keep.
        :param save_freq: int; how often the model will be saved.
        """
        super().__init__(data_shape)
        self._name = name
        self._sess = session
        self._dtype = dtype
        self._input_shape = input_shape
        self._input_ranges = input_ranges
        self._output_ranges = output_ranges
        self._dropout_val = dropout
        self._logging_freq = logging_freq
        self._save_freq = save_freq

        self._steps_so_far = 0

        # Set up placeholders
        self._input_data = tf.placeholder(shape=(None, *input_shape),
                                          dtype=dtype, name='input')
        self._real_output = tf.placeholder(shape=(None, *data_shape),
                                           dtype=dtype, name='real_output')
        self._dropout_ph = tf.placeholder(shape=(), dtype=dtype,
                                          name='dropout')

        # Preprocess placeholders
        batch_size = tf.shape(self._input_data)[0]
        flat_in_size = reduce(mul, input_shape)
        flat_in = tf.reshape(self._input_data, (batch_size, flat_in_size),
                             name='flat_input')
        input_preproc = normalize(flat_in,
                                  data_ranges=self._input_ranges,
                                  target_ranges=norm_range,
                                  name='input_preproc',
                                  dtype=dtype)

        flat_out_size = reduce(mul, data_shape)
        flat_real_out = tf.reshape(self._real_output,
                                   (batch_size, flat_out_size),
                                   'flat_real_output')
        real_output_preproc = normalize(flat_real_out,
                                        data_ranges=self._output_ranges,
                                        target_ranges=norm_range,
                                        name='real_output_preproc',
                                        dtype=dtype)

        # Create network
        self._params = Dnu.create_network_params(flat_in_size,
                                                 layers,
                                                 flat_out_size,
                                                 dtype,
                                                 name='network_params',
                                                 stddev=PARAMS_INIT_STDDEV,
                                                 mean=PARAMS_INIT_MEAN)
        generated_out = Dnu.model_output(input_preproc,
                                         self._params,
                                         activation,
                                         dropout_prob=self._dropout_ph,
                                         out_activation_fn=final_activation,
                                         name='flat_generated_output')
        denorm_generated_out = normalize(generated_out,
                                         data_ranges=norm_range,
                                         target_ranges=self._output_ranges,
                                         name='denorm_generated_out',
                                         dtype=dtype)
        self._generated_output = tf.reshape(denorm_generated_out,
                                            (batch_size, *data_shape),
                                            name='generated_output')

        # Define objective
        gen_real_diff = tf.abs(generated_out - real_output_preproc,
                               name='absolute_difference')
        self._loss = tf.reduce_mean(gen_real_diff, name='l1_loss')

        self._train_step = optimizer.minimize(
            self._loss,
            var_list=Dnu.unpack_params(self._params)
        )

        self._add_summaries()
        # Initialize vars
        init_ops = tf.variables_initializer(Dnu.unpack_params(self._params) +
                                            optimizer.variables())
        self._sess.run(init_ops)
        self._set_up_outputs(output_directory, logging_level, max_checkpoints)

        self._logger.info(
            f'\nCreating MLP with:\n'
            f'Input shape {input_shape}\n'
            f'Output shape {data_shape}\n'
            f'Dtype: {dtype}\n'
            f'Network layers: {layers}\n'
            f'Activation function: {activation.__name__}\n'
            f'Dropout probability: {dropout}\n'
            f'Optimizer: {optimizer.__class__.__name__}\n'
        )

    def _add_summaries(self):
        training_summs = []
        with tf.name_scope(f'network'):
            for i, layer in enumerate(self._params):
                with tf.name_scope(f'{i}/weights'):
                    training_summs.extend(summarize_ndarray(layer.weights))
                with tf.name_scope(f'{i}/biases'):
                    training_summs.extend(summarize_ndarray(layer.biases))
        with tf.name_scope('losses/'):
            training_summs.append(tf.summary.scalar('loss', self._loss))

        self._training_summary = tf.summary.merge(training_summs)

        vis_shape = [tf.shape(self._real_output)[0], *self._data_shape]
        while len(vis_shape) < 4:
            vis_shape.append(1)
        vis_data = tf.reshape(self._real_output, vis_shape)
        self._visualise_summary = tf.summary.image('generated-imgs',
                                                   vis_data,
                                                   max_outputs=MAX_IMGS_TO_VIS)

    def _set_up_outputs(self, output_dir, logging_level, max_checkpoints):
        os.mkdir(output_dir)
        sumaries_dir = os.path.join(output_dir, 'tensorboard')
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.mkdir(checkpoints_dir)
        self._ckpt_path = os.path.join(checkpoints_dir, 'ckpt')
        self._saver = tf.train.Saver(max_to_keep=max_checkpoints)

        self._logger = logging.getLogger(self._name)
        logs_filepath = os.path.join(output_dir, 'logs.log')
        formatter = logging.Formatter('%(asctime)s:%(name)s:'
                                      '%(levelname)s: %(message)s')
        file_handler = logging.FileHandler(logs_filepath)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging_level)

        self._summary_writer = tf.summary.FileWriter(sumaries_dir,
                                                     self._sess.graph)

    def train_step(self, expected_output, condition=None):
        assert condition is not None, 'Condition for MLP has to exist!'
        assert condition.shape[1:] == self._input_shape, \
            f'Expected ({self._input_shape}) and received ' \
            f'({condition.shape[1:]}) data shapes do not match'
        assert expected_output.shape[1:] == self._data_shape, \
            f'Expected ({self._data_shape}) and received ' \
            f'({expected_output.shape[1]}) sizes do not match'
        assert condition.shape[0] == expected_output.shape[0], \
            'You need to pass the same amount of labels as data!'

        # Train network
        self._sess.run(self._train_step, feed_dict={
            self._input_data: condition,
            self._real_output: expected_output,
            self._dropout_ph: self._dropout_val
        })

        self._steps_so_far += 1

        # Save model
        if self._steps_so_far % self._save_freq == 0:
            self._logger.info('Saving the model\n')
            self._saver.save(self._sess, self._ckpt_path,
                             global_step=self._steps_so_far)

        if self._steps_so_far % self._logging_freq == 0:
            self._log_step(expected_output, condition)

    def _log_step(self, expected_output, condition):
        train_summ, loss = \
            self._sess.run(
                [self._training_summary, self._loss], feed_dict={
                    self._input_data: condition,
                    self._real_output: expected_output,
                    self._dropout_ph: 1
                })
        self._summary_writer.add_summary(train_summ, self._steps_so_far)

        self._logger.info(
            f'Current step: {self._steps_so_far}\n'
            f'Loss: {loss}\n'
            '\n--------------------------------------------------\n'
        )

    def visualize_data(self, data):
        self._logger.info('Visualising data\n')
        vis_summ = self._sess.run(self._visualise_summary, feed_dict={
            self._real_output: data,
            self._dropout_ph: 1
        })
        self._summary_writer.add_summary(vis_summ, self._steps_so_far)

    def generate(self, how_many, condition=None):
        assert condition is not None, 'Condition for MLP has to exist!'
        assert condition.shape[1:] == self._input_shape, \
            f'Expected ({self._input_shape}) and received ' \
            f'({condition.shape[1:]}) data shapes do not match'
        assert how_many == condition.shape[0], \
            'You need to pass the same amount of labels as you expect' \
            'generated images!'

        return self._sess.run(self._generated_output, feed_dict={
            self._input_data: condition,
            self._dropout_ph: 1
        })

    def calculate_l1_loss(self, expected_output, condition):
        assert condition is not None, 'Condition for MLP has to exist!'
        assert expected_output.shape[1:] == self._data_shape, \
            f'Expected ({self._data_shape}) and received ' \
            f'({expected_output.shape[1:]}) data shapes do not match'
        assert expected_output.shape[0] == condition.shape[0], \
            'You need to pass the same amount of labels as data!'

        return self._sess.run(self._loss, feed_dict={
            self._real_output: expected_output,
            self._input_data: condition,
            self._dropout_ph: 1
        })
