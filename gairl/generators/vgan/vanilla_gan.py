import logging
import os
import sys
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf

from gairl.neural_utils import DenseNetworkUtils as Dnu
from gairl.neural_utils import summarize_ndarray


IMGS_TO_VIS = 5


# TODO add loading
class VanillaGAN:

    def __init__(self,
                 data_shape,
                 noise_size,
                 session,
                 output_directory,
                 name='VanillaGAN',
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
                 visualisation_freq=1000,
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
        :param visualisation_freq: int; frequency of saving generator
            generator images with their discrimination.
        :param logging_level: logging.LEVEL; level of the internal logger.
        :param max_checkpoints: int; number of checkpoints to keep.
        :param save_freq: int; how often the model will be saved.
        """
        self._name = name
        self._sess = session
        self._data_shape = data_shape
        self._flat_data_size = reduce(mul, data_shape)
        self._g_optimizer = g_optimizer
        self._d_optimizer = d_optimizer
        self._k = k
        self._logging_freq = logging_freq
        self._visual_freq = visualisation_freq
        self._save_freq = save_freq

        self._steps_so_far = 0

        # Set up input placeholders
        self._noise = tf.placeholder(shape=(None, noise_size),
                                     dtype=dtype, name='noise')
        self._real_data = tf.placeholder(shape=(None, self._flat_data_size),
                                         dtype=dtype, name='real_data')

        # Set up generator network
        self._g_params = Dnu.create_network_params(noise_size,
                                                   g_layers,
                                                   self._flat_data_size,
                                                   dtype,
                                                   name='generator_params',
                                                   stddev=1e-3, mean=0)
        self._generated_data = Dnu.model_output(self._noise,
                                                self._g_params,
                                                g_activation,
                                                dropout_prob=g_dropout,
                                                out_activation_fn=tf.nn.tanh,
                                                name='generator_out')

        # Set up discriminator network
        self._d_params = Dnu.create_network_params(self._flat_data_size,
                                                   d_layers,
                                                   1,  # 0 - fake, 1 - real
                                                   dtype,
                                                   name='discriminator_params',
                                                   stddev=1e-3, mean=0)
        self._fake_discrim = Dnu.model_output(self._generated_data,
                                              self._d_params,
                                              d_activation,
                                              dropout_prob=d_dropout,
                                              out_activation_fn=tf.nn.sigmoid,
                                              name='fake_discrimination')
        self._fake_discrim_mean = tf.reduce_mean(self._fake_discrim)
        self._real_discrim = Dnu.model_output(self._real_data,
                                              self._d_params,
                                              d_activation,
                                              dropout_prob=d_dropout,
                                              out_activation_fn=tf.nn.sigmoid,
                                              name='real_discrimination')
        self._real_discrim_mean = tf.reduce_mean(self._real_discrim)

        # Generator objective: min log(1 - fake_discrim) =~
        # =~ max log(fake_discrim) = min -log(fake_discrim)
        unpacked_g_params = Dnu.unpack_params(self._g_params) + \
                            g_optimizer.variables()
        fake_discrim_log = tf.log(self._fake_discrim,
                                  name='fake_discrimination_log')
        self._g_loss = tf.negative(tf.reduce_mean(fake_discrim_log),
                                   name='g_loss')
        self._g_train_step = g_optimizer.minimize(self._g_loss,
                                                  var_list=unpacked_g_params)

        # Discriminator objective:
        # max log(real_discrim) + log(1 - fake_discrim) = min -...
        unpacked_d_params = Dnu.unpack_params(self._d_params) + \
                            d_optimizer.variables()
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
        self._d_train_step = d_optimizer.minimize(self._d_loss,
                                                  var_list=unpacked_d_params)

        self._add_summaries()
        self._initialize_vars()
        self._set_up_outputs(output_directory, logging_level, max_checkpoints)

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
        with tf.name_scope('discriminations'):
            training_summs.append(tf.summary.scalar('real_avg',
                                                    self._real_discrim_mean))
            training_summs.append(tf.summary.scalar('fake_avg',
                                                    self._fake_discrim_mean))

        visualise_summs = []
        # Prepare images to visualize
        images_to_vis = self._generated_data[:IMGS_TO_VIS]
        images_shape = [tf.shape(images_to_vis)[0], *self._data_shape]
        while len(images_shape) < 4:
            images_shape.append([1])
        images_to_vis = tf.reshape(images_to_vis, images_shape)
        # Add top row depending on discrimination
        discriminations = self._fake_discrim[:images_shape[0]]
        discriminations = tf.reshape(discriminations, [images_shape[0], 1, 1, 1])
        discriminations = tf.tile(discriminations, [1, 1, images_shape[1], 1])
        images_to_vis = tf.concat([discriminations, images_to_vis], axis=1)
        with tf.name_scope('generated-imgs-discriminated'):
            visualise_summs.append(tf.summary.image('generated-imgs',
                                                    images_to_vis,
                                                    max_outputs=IMGS_TO_VIS))

        self._training_summary = tf.summary.merge(training_summs)
        self._visualise_summary = tf.summary.merge(visualise_summs)

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

    def train_step(self, data_batch, noise_batch):
        batch_size = len(data_batch)
        flat_data = data_batch.reshape(batch_size, self._flat_data_size)
        norm_data = np.interp(flat_data, (flat_data.min(), flat_data.max()),
                              (-1, +1))

        # Train Discriminator
        self._sess.run(self._d_train_step, feed_dict={
                           self._noise: noise_batch,
                           self._real_data: norm_data
                       })
        self._steps_so_far += 1

        # Train Generator
        if self._steps_so_far % self._k == 0:
            self._sess.run(self._g_train_step, feed_dict={
                                self._noise: noise_batch
                           })

        # Save model
        if self._steps_so_far % self._save_freq == 0:
            self._logger.info('Saving the model\n')
            self._saver.save(self._sess, self._ckpt_path,
                             global_step=self._steps_so_far)

        if self._steps_so_far % self._logging_freq == 0:
            self._log_step(norm_data, noise_batch)

    def _log_step(self, norm_data_batch, noise_batch):
        train_summ, fake_discrim_mean, real_discrim_mean = \
            self._sess.run([self._training_summary, self._fake_discrim_mean,
                            self._real_discrim_mean], feed_dict={
                               self._noise: noise_batch,
                               self._real_data: norm_data_batch
                           })
        self._summary_writer.add_summary(train_summ, self._steps_so_far)

        if self._steps_so_far % self._visual_freq < self._logging_freq:
            self._logger.info('Visualising generated data\n')
            vis_summ = self._sess.run(self._visualise_summary, feed_dict={
                                          self._noise: noise_batch,
                                          self._real_data: norm_data_batch
                                      })
            self._summary_writer.add_summary(vis_summ, self._steps_so_far)

        self._logger.info(
            f'Current step: {self._steps_so_far}\n'
            f'Fake discrimination mean: {fake_discrim_mean}\n'
            f'Real discrimination mean: {real_discrim_mean}\n'
            '\n--------------------------------------------------\n'
        )