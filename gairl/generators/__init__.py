from gairl.generators.mlp import multilayer_perceptron_config as mlp_conf
from gairl.generators.mlp.multilayer_perceptron import MultilayerPerceptron
from gairl.generators.vgan import vanilla_gan_config as vgan_conf
from gairl.generators.vgan.vanilla_gan import VanillaGAN
from gairl.generators.wgan.wasserstein_gan import WassersteinGAN
from gairl.generators.wgan import wasserstein_gan_config as wgan_conf
from gairl.generators.wgan_gp.wasserstein_gan_gp import WassersteinGANGP
from gairl.generators.wgan_gp import wasserstein_gan_gp_config as wgan_gp_conf


def create_generator(generator_name,
                     data_shape,
                     session,
                     name=None,
                     data_ranges=(-1, 1),
                     conditional_shape=None,
                     conditional_ranges=(-1, 1),
                     output_dir=None,
                     layers=None,
                     dropout=None):
    if generator_name not in _STR_TO_GENERATOR.keys():
        raise AttributeError(f"There's no generator like {generator_name}.")

    if generator_name == 'mlp':
        if not conditional_shape:
            raise AttributeError('Multilayer Perceptron can work '
                                 'only in a conditional mode.')
        return _create_multilayer_perceptron(conditional_shape,
                                             data_shape,
                                             session,
                                             name=name,
                                             input_ranges=conditional_ranges,
                                             output_ranges=data_ranges,
                                             output_dir=output_dir,
                                             layers=layers,
                                             dropout=dropout)

    creation_method = _STR_TO_GENERATOR[generator_name]
    return creation_method(data_shape, session,
                           data_ranges=data_ranges,
                           conditional_shape=conditional_shape,
                           conditional_ranges=conditional_ranges,
                           name=name,
                           output_dir=output_dir,
                           layers=layers,
                           dropout=dropout)


def _create_vanilla_gan(data_shape,
                        session,
                        name=None,
                        data_ranges=(-1, 1),
                        conditional_shape=None,
                        conditional_ranges=(-1, 1),
                        output_dir=None,
                        layers=None,
                        dropout=None):
    output_dir = output_dir if output_dir else vgan_conf.OUTPUT_DIRECTORY
    name = name if name else 'VanillaGAN'
    g_layers = layers if layers else vgan_conf.G_LAYERS
    g_dropout = dropout if dropout else vgan_conf.G_DROPOUT
    d_layers = layers if layers else vgan_conf.D_LAYERS
    d_dropout = dropout if dropout else vgan_conf.D_DROPOUT
    return VanillaGAN(data_shape,
                      session,
                      output_dir,
                      name=name,
                      noise_size=vgan_conf.NOISE_SIZE,
                      data_ranges=data_ranges,
                      conditional_shape=conditional_shape,
                      conditional_ranges=conditional_ranges,
                      dtype=vgan_conf.DTYPE,
                      g_layers=g_layers,
                      g_activation=vgan_conf.G_ACTIVATION,
                      g_dropout=g_dropout,
                      g_optimizer=vgan_conf.G_OPTIMIZER,
                      d_layers=d_layers,
                      d_activation=vgan_conf.D_ACTIVATION,
                      d_dropout=d_dropout,
                      d_optimizer=vgan_conf.D_OPTIMIZER,
                      k=vgan_conf.K,
                      logging_freq=vgan_conf.LOGGING_FREQ,
                      logging_level=vgan_conf.LOGGING_LEVEL,
                      max_checkpoints=vgan_conf.MAX_CHECKPOINTS,
                      save_freq=vgan_conf.SAVE_FREQ)


def _create_wasserstein_gan(data_shape,
                            session,
                            name=None,
                            data_ranges=(-1, 1),
                            conditional_shape=None,
                            conditional_ranges=(-1, 1),
                            output_dir=None,
                            layers=None,
                            dropout=None):
    output_dir = output_dir if output_dir else wgan_conf.OUTPUT_DIRECTORY
    name = name if name else 'WassersteinGAN'
    g_layers = layers if layers else wgan_conf.G_LAYERS
    g_dropout = dropout if dropout else wgan_conf.G_DROPOUT
    d_layers = layers if layers else wgan_conf.D_LAYERS
    d_dropout = dropout if dropout else wgan_conf.D_DROPOUT
    return WassersteinGAN(data_shape,
                          session,
                          output_dir,
                          name=name,
                          noise_size=vgan_conf.NOISE_SIZE,
                          data_ranges=data_ranges,
                          conditional_shape=conditional_shape,
                          conditional_ranges=conditional_ranges,
                          dtype=wgan_conf.DTYPE,
                          g_layers=g_layers,
                          g_activation=wgan_conf.G_ACTIVATION,
                          g_dropout=g_dropout,
                          g_optimizer=wgan_conf.G_OPTIMIZER,
                          d_layers=d_layers,
                          d_activation=wgan_conf.D_ACTIVATION,
                          d_dropout=d_dropout,
                          d_optimizer=wgan_conf.D_OPTIMIZER,
                          k=wgan_conf.K,
                          clip_bounds=wgan_conf.CLIP_BOUNDS,
                          logging_freq=wgan_conf.LOGGING_FREQ,
                          logging_level=wgan_conf.LOGGING_LEVEL,
                          max_checkpoints=wgan_conf.MAX_CHECKPOINTS,
                          save_freq=wgan_conf.SAVE_FREQ)


def _create_wasserstein_gan_gp(data_shape,
                               session,
                               name=None,
                               data_ranges=(-1, 1),
                               conditional_shape=None,
                               conditional_ranges=(-1, 1),
                               output_dir=None,
                               layers=None,
                               dropout=None):
    output_dir = output_dir if output_dir else wgan_gp_conf.OUTPUT_DIRECTORY
    name = name if name else 'WassersteinGAN with GP'
    g_layers = layers if layers else wgan_gp_conf.G_LAYERS
    g_dropout = dropout if dropout else wgan_gp_conf.G_DROPOUT
    d_layers = layers if layers else wgan_gp_conf.D_LAYERS
    d_dropout = dropout if dropout else wgan_gp_conf.D_DROPOUT
    return WassersteinGANGP(data_shape,
                            session,
                            output_dir,
                            name=name,
                            noise_size=vgan_conf.NOISE_SIZE,
                            data_ranges=data_ranges,
                            conditional_shape=conditional_shape,
                            conditional_ranges=conditional_ranges,
                            dtype=wgan_gp_conf.DTYPE,
                            g_layers=g_layers,
                            g_activation=wgan_gp_conf.G_ACTIVATION,
                            g_dropout=g_dropout,
                            g_optimizer=wgan_gp_conf.G_OPTIMIZER,
                            d_layers=d_layers,
                            d_activation=wgan_gp_conf.D_ACTIVATION,
                            d_dropout=d_dropout,
                            d_optimizer=wgan_gp_conf.D_OPTIMIZER,
                            k=wgan_gp_conf.K,
                            penalty_coeff=wgan_gp_conf.PENALTY_COEFF,
                            logging_freq=wgan_gp_conf.LOGGING_FREQ,
                            logging_level=wgan_gp_conf.LOGGING_LEVEL,
                            max_checkpoints=wgan_gp_conf.MAX_CHECKPOINTS,
                            save_freq=wgan_gp_conf.SAVE_FREQ)


def _create_multilayer_perceptron(input_shape,
                                  output_shape,
                                  session,
                                  name=None,
                                  input_ranges=(-1, 1),
                                  output_ranges=(-1, 1),
                                  output_dir=None,
                                  layers=None,
                                  dropout=None):
    output_dir = output_dir if output_dir else mlp_conf.OUTPUT_DIRECTORY
    name = name if name else 'Multilayer Perceptron'
    layers = layers if layers else mlp_conf.LAYERS
    dropout = dropout if dropout else mlp_conf.DROPOUT
    return MultilayerPerceptron(input_shape,
                                output_shape,
                                session,
                                output_dir,
                                name=name,
                                input_ranges=input_ranges,
                                output_ranges=output_ranges,
                                dtype=mlp_conf.DTYPE,
                                layers=layers,
                                activation=mlp_conf.ACTIVATION,
                                final_activation=mlp_conf.FINAL_ACTIVATION,
                                norm_range=mlp_conf.NORM_RANGE,
                                dropout=dropout,
                                optimizer=mlp_conf.OPTIMIZER,
                                logging_freq=mlp_conf.LOGGING_FREQ,
                                logging_level=mlp_conf.LOGGING_LEVEL,
                                max_checkpoints=mlp_conf.MAX_CHECKPOINTS,
                                save_freq=mlp_conf.SAVE_FREQ)


_STR_TO_GENERATOR = {
    'vgan': _create_vanilla_gan,
    'wgan': _create_wasserstein_gan,
    'wgan_gp': _create_wasserstein_gan_gp,
    'mlp': _create_multilayer_perceptron
}
