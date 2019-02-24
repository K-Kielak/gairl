import logging

import tensorflow as tf

from gairl.generators.vgan.vanilla_gan import VanillaGAN
from gairl.generators.vgan.vanilla_gan import PARAMS_INIT_STDDEV
from gairl.generators.vgan.vanilla_gan import PARAMS_INIT_MEAN
from gairl.neural_utils import DenseNetworkUtils as Dnu


class WassersteinGAN(VanillaGAN):

    def __init__(self,
                 data_shape,
                 noise_size,
                 session,
                 output_directory,
                 name='WassersteinGAN',
                 labels_num=None,
                 dtype=tf.float64,
                 g_layers=(256, 512, 1024),
                 g_activation=tf.nn.leaky_relu,
                 g_dropout=1,
                 g_optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                 d_layers=(1024, 512, 256),
                 d_activation=tf.nn.leaky_relu,
                 d_dropout=0.8,
                 d_optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                 k=5,
                 clip_bounds=(-0.01, 0.01),
                 logging_freq=100,
                 visualisation_freq=1000,
                 logging_level=logging.INFO,
                 max_checkpoints=5,
                 save_freq=1000):
        """
        Initializes feed-forward version of Wasserstein GAN.
        :param noise_size: int; describes the size of the noise that
            will be fed as an input to the generator.
        :param data_shape: tuple of int; describes the size of the
            data that GAN is supposed to generate.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the model.
        :param labels_num: int; describes number of labels used for
            conditional GAN, None or 0 if non-conditional GAN.
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
        assert len(clip_bounds) == 2, 'There should be minimum ' \
                                      'clip and maximum clip.'
        assert clip_bounds[0] < clip_bounds[1], 'Minimum clip should be' \
                                                'smaller than maximum clip.'
        self._clip_bounds = clip_bounds
        super().__init__(data_shape,
                         noise_size,
                         session,
                         output_directory,
                         name=name,
                         labels_num=labels_num,
                         dtype=dtype,
                         g_layers=g_layers,
                         g_activation=g_activation,
                         g_dropout=g_dropout,
                         g_optimizer=g_optimizer,
                         d_layers=d_layers,
                         d_activation=d_activation,
                         d_dropout=d_dropout,
                         d_optimizer=d_optimizer,
                         k=k,
                         logging_freq=logging_freq,
                         visualisation_freq=visualisation_freq,
                         logging_level=logging_level,
                         max_checkpoints=max_checkpoints,
                         save_freq=save_freq)
        self._logger.info(
            f'It is a Wasserstein GAN with:\n'
            f'Clip bounds: {clip_bounds}'
        )

    def _create_discriminator_network(self, layers, activation, dropout, dtype):
        self._d_params = Dnu.create_network_params(self._flat_data_size +
                                                   self._labels_num,
                                                   layers,
                                                   1,  # < 0 - fake, > 0 - real
                                                   dtype,
                                                   name='discriminator_params',
                                                   stddev=PARAMS_INIT_STDDEV,
                                                   mean=PARAMS_INIT_MEAN)
        fake_discrim_in = tf.concat([self._generated_data,
                                     self._labels_onehot], axis=1)
        self._fake_discrim = Dnu.model_output(fake_discrim_in,
                                              self._d_params,
                                              activation,
                                              dropout_prob=dropout,
                                              out_activation_fn=None,  # Linear
                                              name='fake_discrimination')
        self._fake_discrim_mean = tf.reduce_mean(self._fake_discrim)

        real_discrim_in = tf.concat([self._real_data,
                                     self._labels_onehot], axis=1)
        self._real_discrim = Dnu.model_output(real_discrim_in,
                                              self._d_params,
                                              activation,
                                              dropout_prob=dropout,
                                              out_activation_fn=None,  # Linear
                                              name='real_discrimination')
        self._real_discrim_mean = tf.reduce_mean(self._real_discrim)

    def _define_generator_objective(self, optimizer):
        # Generator objective: max fake_discrim = min -fake_discrim
        unpacked_g_params = Dnu.unpack_params(self._g_params)
        self._g_loss = tf.negative(self._fake_discrim_mean, name='g_loss')
        self._g_train_step = optimizer.minimize(self._g_loss,
                                                var_list=unpacked_g_params +
                                                         optimizer.variables())

    def _define_discriminator_objective(self, optimizer):
        # Discriminator objective: max (real_discrim + fake_discrim) =
        # = min (fake_discrim - real_discrim)
        unpacked_d_params = Dnu.unpack_params(self._d_params)
        self._real_d_loss = tf.negative(self._real_discrim_mean,
                                        name='real_data_d_loss')
        self._fake_d_loss = tf.identity(self._fake_discrim_mean,
                                        name='fake_data_d_loss')
        self._d_loss = tf.add(self._real_d_loss, self._fake_d_loss,
                              name='d_loss')
        loss_minimization = optimizer.minimize(self._d_loss,
                                               var_list=unpacked_d_params +
                                                        optimizer.variables())
        # Make sure minimization happens before clipping
        with tf.get_default_graph().control_dependencies([loss_minimization]):
            clip_ops = []
            for param in unpacked_d_params:
                clip = tf.clip_by_value(param,
                                        clip_value_min=self._clip_bounds[0],
                                        clip_value_max=self._clip_bounds[1])
                clip_ops.append(tf.assign(param, clip))

        # Group training step ops (minimization + clipping)
        self._d_train_step = tf.group(loss_minimization, *clip_ops)
