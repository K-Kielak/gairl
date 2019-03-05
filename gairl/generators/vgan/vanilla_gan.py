import logging
import os
import sys
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf

from gairl.neural_utils import DenseNetworkUtils as Dnu
from gairl.neural_utils import summarize_ndarray


MAX_IMGS_TO_VIS = 10
PARAMS_INIT_STDDEV = 1e-3
PARAMS_INIT_MEAN = 0


# TODO add loading
class VanillaGAN:

    def __init__(self,
                 data_shape,
                 noise_size,
                 session,
                 output_directory,
                 name='VanillaGAN',
                 cond_in_size=None,
                 dtype=tf.float64,
                 g_layers=(256, 512, 1024),
                 g_activation=tf.nn.leaky_relu,
                 g_dropout=1,
                 g_optimizer=tf.train.AdamOptimizer(
                     learning_rate=1e-4,
                     epsilon=1.5e-4
                 ),
                 d_layers=(1024, 512, 256),
                 d_activation=tf.nn.leaky_relu,
                 d_dropout=0.8,
                 d_optimizer=tf.train.GradientDescentOptimizer(
                     learning_rate=1e-4
                 ),
                 k=1,
                 logging_freq=100,
                 logging_level=logging.INFO,
                 max_checkpoints=5,
                 save_freq=1000):
        """
        Initializes feed-forward version of vanilla GAN
        :param noise_size: int; describes the size of the noise that
            will be fed as an input to the generator.
        :param data_shape: tuple of int; describes the size of the
            data that GAN is supposed to generate.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the model.
        :param cond_in_size: int; describes size of the conditional
            input used for GAN, None or 0 if non-conditional GAN.
        :param is_cond_one_hot: bool; describes if conditional input
            is a discrete one-hot encoded label or continous data.
        :param dtype: tensorflow.DType; type used for the model.
        :param g_layers: tuple of ints; describes number of nodes
            in each hidden layer of the generator network.
        :param g_activation: activation function for hidden layers
            of the generator network.
        :param g_dropout: float in (0, 1]; dropout probability for
            hidden layers of the generator network.
        :param g_optimizer: tf.trainOptimizer; optimizer for the
            generator network.
        :param d_layers: tuple of ints; describes number of nodes
            in each hidden layer of the discriminator network.
        :param d_activation: activation function for hidden layers
            if the discriminator network.
        :param d_dropout: float in (0, 1]; dropout probability for
            hidden layers of the discriminator network.
        :param d_optimizer: tf.trainOptimizer; optimizer for the
            discriminator network.
        :param k: int > 0; number of iterations of the discriminator
            per generator iteration.
        :param logging_freq: int; frequency of progress logging and
            writing tensorflow summaries.
        :param logging_level: logging.LEVEL; level of the internal logger.
        :param max_checkpoints: int; number of checkpoints to keep.
        :param save_freq: int; how often the model will be saved.
        """
        self._name = name
        self._sess = session
        self._data_shape = data_shape
        self._cond_in_size = cond_in_size if cond_in_size else 0
        self._flat_data_size = reduce(mul, data_shape)
        self._noise_size = noise_size
        self._dtype = dtype
        self._g_optimizer = g_optimizer
        self._g_activation = g_activation
        self._d_optimizer = d_optimizer
        self._d_activation = d_activation
        self._k = k
        self._logging_freq = logging_freq
        self._save_freq = save_freq

        self._steps_so_far = 0

        # Set up inputs
        self._noise = tf.placeholder(shape=(None, self._noise_size),
                                     dtype=dtype, name='noise')
        self._real_data = tf.placeholder(shape=(None, self._flat_data_size),
                                         dtype=dtype, name='real_data')
        self._g_condition = tf.placeholder(shape=(None, self._cond_in_size,),
                                           dtype=dtype, name='labels')
        self._d_condition = tf.placeholder(shape=(None, self._cond_in_size,),
                                           dtype=dtype, name='labels')

        # Create networks
        self._create_generator_network(g_layers, g_activation,
                                       g_dropout, dtype)
        self._create_discriminator_network(d_layers, d_activation,
                                           d_dropout, dtype)

        # Define objectives
        self._define_generator_objective(g_optimizer)
        self._define_discriminator_objective(d_optimizer)

        self._add_summaries()
        self._initialize_vars()
        self._set_up_outputs(output_directory, logging_level, max_checkpoints)

        self._logger.info(
            f'\nCreating GAN with:\n'
            f'Data shape {data_shape}\n'
            f'Noise size {noise_size}\n'
            f'Dtype: {dtype}\n'
            f'Generator layers: {g_layers}\n'
            f'Generator activation function: {g_activation.__name__}\n'
            f'Generator dropout probability: {g_dropout}\n'
            f'Generator optimizer: {g_optimizer.__class__.__name__}\n'
            f'Discriminator layers: {d_layers}\n'
            f'Discriminator activation function: {d_activation.__name__}\n'
            f'Discriminator dropout probability: {d_dropout}\n'
            f'Discriminator optimizer: {d_optimizer.__class__.__name__}\n'
            f'K: {k}\n'
            f'Labels num: {cond_in_size}'
        )

    def _create_generator_network(self, layers, activation, dropout, dtype):
        self._g_params = Dnu.create_network_params(self._noise_size +
                                                   self._cond_in_size,
                                                   layers,
                                                   self._flat_data_size,
                                                   dtype,
                                                   name='generator_params',
                                                   stddev=PARAMS_INIT_STDDEV,
                                                   mean=PARAMS_INIT_MEAN)
        gen_in = tf.concat([self._noise, self._g_condition], axis=1)
        self._generated_data = Dnu.model_output(gen_in,
                                                self._g_params,
                                                activation,
                                                dropout_prob=dropout,
                                                out_activation_fn=tf.nn.tanh,
                                                name='generator_out')

    def _create_discriminator_network(self, layers, activation, dropout, dtype):
        self._d_params = Dnu.create_network_params(self._flat_data_size +
                                                   self._cond_in_size,
                                                   layers,
                                                   1,  # 0 - fake, 1 - real
                                                   dtype,
                                                   name='discriminator_params',
                                                   stddev=PARAMS_INIT_STDDEV,
                                                   mean=PARAMS_INIT_MEAN)
        fake_discrim_in = tf.concat([self._generated_data,
                                     self._d_condition], axis=1)
        self._fake_discrim = Dnu.model_output(fake_discrim_in,
                                              self._d_params,
                                              activation,
                                              dropout_prob=dropout,
                                              out_activation_fn=tf.nn.sigmoid,
                                              name='fake_discrimination')
        self._fake_discrim_mean = tf.reduce_mean(self._fake_discrim)

        real_discrim_in = tf.concat([self._real_data,
                                     self._d_condition], axis=1)
        self._real_discrim = Dnu.model_output(real_discrim_in,
                                              self._d_params,
                                              activation,
                                              dropout_prob=dropout,
                                              out_activation_fn=tf.nn.sigmoid,
                                              name='real_discrimination')
        self._real_discrim_mean = tf.reduce_mean(self._real_discrim)

    def _define_generator_objective(self, optimizer):
        # Generator objective: min log(1 - fake_discrim) =~
        # =~ max log(fake_discrim) = min -log(fake_discrim)
        unpacked_g_params = Dnu.unpack_params(self._g_params) + \
                            optimizer.variables()
        fake_discrim_log = tf.log(self._fake_discrim,
                                  name='fake_discrimination_log')
        self._g_loss = tf.negative(tf.reduce_mean(fake_discrim_log),
                                   name='g_loss')
        self._g_train_step = optimizer.minimize(self._g_loss,
                                                var_list=unpacked_g_params)

    def _define_discriminator_objective(self, optimizer):
        # Discriminator objective:
        # max log(real_discrim) + log(1 - fake_discrim) =
        # = min -(log(real_discrim) - log(1 - fake_discrim))
        unpacked_d_params = Dnu.unpack_params(self._d_params) + \
                            optimizer.variables()
        real_discrim_log = tf.log(self._real_discrim,
                                  name='real_discrimination_log')
        self._real_d_loss = tf.negative(tf.reduce_mean(real_discrim_log),
                                        name='real_data_d_loss')
        fake_rev_log = tf.log(tf.ones_like(self._fake_discrim) -
                              self._fake_discrim)
        self._fake_d_loss = tf.negative(tf.reduce_mean(fake_rev_log),
                                        name='fake_data_d_loss')
        self._d_loss = tf.add(self._real_d_loss, self._fake_d_loss,
                              name='d_loss')
        self._d_train_step = optimizer.minimize(self._d_loss,
                                                var_list=unpacked_d_params)

    def _add_summaries(self):
        training_summs = []
        with tf.name_scope(f'generator-network'):
            for i, layer in enumerate(self._g_params):
                with tf.name_scope(f'{i}/weights'):
                    training_summs.extend(summarize_ndarray(layer.weights))
                with tf.name_scope(f'{i}/biases'):
                    training_summs.extend(summarize_ndarray(layer.biases))
        with tf.name_scope(f'discriminator-network'):
            for i, layer in enumerate(self._d_params):
                with tf.name_scope(f'{i}/weights'):
                    training_summs.extend(summarize_ndarray(layer.weights))
                with tf.name_scope(f'{i}/biases'):
                    training_summs.extend(summarize_ndarray(layer.biases))
        with tf.name_scope('losses'):
            training_summs.append(tf.summary.scalar('generator-loss',
                                                    self._g_loss))
            training_summs.append(tf.summary.scalar('fake-discriminator-loss',
                                                    self._fake_d_loss))
            training_summs.append(tf.summary.scalar('real-discriminator-loss',
                                                    self._real_d_loss))
            training_summs.append(tf.summary.scalar('discriminator-loss',
                                                    self._d_loss))
            gen_real_diff = tf.abs(self._generated_data - self._real_data)

            l1_loss = tf.reduce_sum(tf.reduce_mean(gen_real_diff, axis=1))
            training_summs.append(tf.summary.scalar('L1-loss', l1_loss))
        with tf.name_scope('discriminations'):
            training_summs.append(tf.summary.scalar('real_avg',
                                                    self._real_discrim_mean))
            training_summs.append(tf.summary.scalar('fake_avg',
                                                    self._fake_discrim_mean))

        self._training_summary = tf.summary.merge(training_summs)
        vis_shape = [tf.shape(self._real_data)[0], *self._data_shape]
        while len(vis_shape) < 4:
            vis_shape.append(1)
        vis_data = tf.reshape(self._real_data, vis_shape)
        self._visualise_summary = tf.summary.image('generated-imgs',
                                                   vis_data,
                                                   max_outputs=MAX_IMGS_TO_VIS)

    def _initialize_vars(self):
        init_ops = tf.variables_initializer(
            Dnu.unpack_params(self._g_params) +
            Dnu.unpack_params(self._d_params) +
            self._g_optimizer.variables() +
            self._d_optimizer.variables()
        )
        self._sess.run(init_ops)

    def _set_up_outputs(self, output_dir, logging_level, max_checkpoints):
        os.mkdir(output_dir)
        logs_filepath = os.path.join(output_dir, 'logs.log')
        sumaries_dir = os.path.join(output_dir, 'tensorboard')
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.mkdir(checkpoints_dir)
        self._ckpt_path = os.path.join(checkpoints_dir, 'ckpt')
        self._saver = tf.train.Saver(max_to_keep=max_checkpoints)

        self._logger = logging.getLogger(self._name)
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

    def train_step(self, data_batch, noise_batch,
                   g_condition=None, d_condition=None):
        assert data_batch.shape[1:] == self._data_shape, \
            f'Expected ({self._data_shape}) and received ' \
            f'({data_batch.shape[1:]}) data shapes do not match'
        assert noise_batch.shape[1] == self._noise_size, \
            f'Expected ({self._noise_size}) and received ' \
            f'({noise_batch.shape[1]}) noise sizes do not match'
        assert data_batch.shape[0] == noise_batch.shape[0], \
            'You need to pass the same amount of noise as data!'
        if g_condition is None:
            g_condition = np.zeros((data_batch.shape[0], 0))
            d_condition = g_condition
        elif d_condition is None:
            d_condition = g_condition
        assert data_batch.shape[0] == g_condition.shape[0], \
            'You need to pass the same amount of labels as data!'
        assert data_batch.shape[0] == d_condition.shape[0], \
            'You need to pass the same amount of labels as data!'

        batch_size = len(data_batch)
        flat_data = data_batch.reshape(batch_size, self._flat_data_size)
        norm_data = np.interp(flat_data, (flat_data.min(), flat_data.max()),
                              (-1, +1))

        # Train Discriminator
        self._sess.run(self._d_train_step, feed_dict={
                           self._noise: noise_batch,
                           self._real_data: norm_data,
                           self._g_condition: g_condition,
                           self._d_condition: d_condition
                       })
        self._steps_so_far += 1

        # Train Generator
        if self._steps_so_far % self._k == 0:
            self._sess.run(self._g_train_step, feed_dict={
                                self._noise: noise_batch,
                                self._g_condition: g_condition,
                                self._d_condition: d_condition
                           })

        # Save model
        if self._steps_so_far % self._save_freq == 0:
            self._logger.info('Saving the model\n')
            self._saver.save(self._sess, self._ckpt_path,
                             global_step=self._steps_so_far)

        if self._steps_so_far % self._logging_freq == 0:
            self._log_step(norm_data, noise_batch, g_condition, d_condition)

    def _log_step(self, norm_data_batch, noise_batch,
                  g_condition, d_condition):
        train_summ, fake_discrim_mean, real_discrim_mean = \
            self._sess.run([self._training_summary, self._fake_discrim_mean,
                            self._real_discrim_mean], feed_dict={
                               self._noise: noise_batch,
                               self._real_data: norm_data_batch,
                               self._g_condition: g_condition,
                               self._d_condition: d_condition
                           })
        self._summary_writer.add_summary(train_summ, self._steps_so_far)

        self._logger.info(
            f'Current step: {self._steps_so_far}\n'
            f'Fake discrimination mean: {fake_discrim_mean}\n'
            f'Real discrimination mean: {real_discrim_mean}\n'
            '\n--------------------------------------------------\n'
        )

    def visualize_data(self, data):
        self._logger.info('Visualising data\n')
        vis_summ = self._sess.run(self._visualise_summary, feed_dict={
            self._real_data: data,
        })
        self._summary_writer.add_summary(vis_summ, self._steps_so_far)

    def generate(self, noise_batch, g_condition=None):
        assert (noise_batch.shape[1] == self._noise_size), \
            f'Expected ({self._noise_size}) and received ' \
            f'({noise_batch.shape[1]}) noise sizes do not match'
        if g_condition is None:
            g_condition = np.zeros((noise_batch.shape[0], 0))
        assert noise_batch.shape[0] == g_condition.shape[0], \
            'You need to pass the same amount of labels as noise!'

        return self._sess.run(self._generated_data, feed_dict={
                                  self._noise: noise_batch,
                                  self._g_condition: g_condition
                              })
