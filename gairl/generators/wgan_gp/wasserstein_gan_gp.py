import logging

import tensorflow as tf

from gairl.generators.wgan.wasserstein_gan import WassersteinGAN
from gairl.neural_utils import DenseNetworkUtils as Dnu


class WassersteinGANGP(WassersteinGAN):

    def __init__(self,
                 data_shape,
                 session,
                 output_directory,
                 name='WassersteinGANGP',
                 noise_size=100,
                 cond_in_size=None,
                 data_ranges=(-1, 1),
                 dtype=tf.float64,
                 g_layers=(256, 512, 1024),
                 g_activation=tf.nn.leaky_relu,
                 g_dropout=1,
                 g_optimizer=tf.train.AdamOptimizer(
                     learning_rate=1e-4,
                     beta1=0.5,
                     beta2=0.9
                 ),
                 d_layers=(1024, 512, 256),
                 d_activation=tf.nn.leaky_relu,
                 d_dropout=0.8,
                 d_optimizer=tf.train.AdamOptimizer(
                     learning_rate=1e-4,
                     beta1=0.5,
                     beta2=0.9
                 ),
                 k=5,
                 penalty_coeff=10,
                 logging_freq=100,
                 logging_level=logging.INFO,
                 max_checkpoints=5,
                 save_freq=1000):
        """
        Initializes feed-forward version of Wasserstein GAN with Gradient Penalty
        :param data_shape: tuple of int; describes the size of the
            data that GAN is supposed to generate.
        :param cond_in_size: int; describes size of the conditional
            input used for GAN, None or 0 if non-conditional GAN.
        :param data_ranges: tuple of ints; specifies what is the range of
            data that needs to be generated in terms of max and min values.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the model.
        :param noise_size: int; describes the size of the noise that
            will be fed as an input to the generator.
        :param cond_in_size: int; describes size of the conditional
            input used for GAN, None or 0 if non-conditional GAN.
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
        self._penalty_coeff = penalty_coeff
        super().__init__(data_shape,
                         session,
                         output_directory,
                         name=name,
                         noise_size=noise_size,
                         cond_in_size=cond_in_size,
                         data_ranges=data_ranges,
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
                         logging_level=logging_level,
                         max_checkpoints=max_checkpoints,
                         save_freq=save_freq)
        self._logger.info(
            f'It uses gradient penalty with coefficient: {penalty_coeff}'
        )

    def _define_discriminator_objective(self, optimizer):
        # Discriminator objective: max (real_discrim + fake_discrim) =
        # = min (fake_discrim - real_discrim)
        unpacked_d_params = Dnu.unpack_params(self._d_params)

        # Calculate standard loss
        self._real_d_loss = tf.negative(self._real_discrim_mean,
                                        name='real_data_d_loss')
        self._fake_d_loss = tf.identity(self._fake_discrim_mean,
                                        name='fake_data_d_loss')
        non_penalized_loss = tf.add(self._real_d_loss, self._fake_d_loss,
                                    name='non_penalized_loss')

        # Calculate gradient penalty
        differences = tf.subtract(self._generated_data_flat,
                                  self._real_data_preproc,
                                  name='differences')
        random_scaling = tf.random_uniform(
            shape=[tf.shape(self._real_data_preproc)[0], 1],
            dtype=self._dtype, minval=0, maxval=1
        )
        interpolates = tf.add(self._real_data_preproc,
                              random_scaling*differences,
                              name='interpolates')
        labeled_interpolates = tf.concat([interpolates,
                                          self._d_condition], axis=1)
        interpolates_discrim = Dnu.model_output(labeled_interpolates,
                                                self._d_params,
                                                self._d_activation,
                                                name='interpolates_discrim')
        # TODO shouldn't we propagate labels further as well?
        interp_grads = tf.gradients(interpolates_discrim, [interpolates],
                                    name='interpolates_gradients')[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(interp_grads),
                                       reduction_indices=[1]), name='slopes')
        gradient_penalty = tf.reduce_mean((slopes-1.)**2,
                                          name='gradient_penalty')
        gradient_penalty = tf.scalar_mul(self._penalty_coeff, gradient_penalty)

        self._d_loss = tf.add(non_penalized_loss, gradient_penalty,
                              name='d_loss')
        self._d_train_step = optimizer.minimize(self._d_loss,
                                                var_list=unpacked_d_params +
                                                         optimizer.variables())
