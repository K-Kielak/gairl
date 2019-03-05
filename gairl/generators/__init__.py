from gairl.generators.vgan import vanilla_gan_config as vgan_conf
from gairl.generators.vgan.vanilla_gan import VanillaGAN
from gairl.generators.wgan.wasserstein_gan import WassersteinGAN
from gairl.generators.wgan import wasserstein_gan_config as wgan_conf
from gairl.generators.wgan_gp.wasserstein_gan_gp import WassersteinGANGP
from gairl.generators.wgan_gp import wasserstein_gan_gp_config as wgan_gp_conf


def create_gan(gan_name,
               data_shape,
               noise_size,
               session,
               name=None,
               cond_in_size=None,
               data_range=(-1, 1),
               output_dir=None,
               separate_logging=True):
    if gan_name not in _STR_TO_GAN.keys():
        raise AttributeError(f"There's no agent like {gan_name}. You "
                             f"can choose only from {_STR_TO_GAN.keys()}")

    creation_method = _STR_TO_GAN[gan_name]
    if name:
        return creation_method(data_shape, noise_size, session,
                               cond_in_size=cond_in_size,
                               data_range=data_range,
                               name=name,
                               output_dir=output_dir,
                               separate_logging=separate_logging)

    return creation_method(data_shape, noise_size, session,
                           cond_in_size=cond_in_size,
                           data_range=data_range,
                           output_dir=output_dir,
                           separate_logging=separate_logging)


def _create_vanilla_gan(data_shape,
                        noise_size,
                        session,
                        name='VanillaGAN',
                        cond_in_size=None,
                        data_range=(-1, 1),
                        output_dir=None,
                        separate_logging=True):
    if not output_dir:
        output_dir = vgan_conf.OUTPUT_DIRECTORY

    logging_level = vgan_conf.LOGGING_LEVEL if separate_logging else None
    return VanillaGAN(data_shape,
                      noise_size,
                      session,
                      output_dir,
                      name=name,
                      cond_in_size=cond_in_size,
                      data_range=data_range,
                      dtype=vgan_conf.DTYPE,
                      g_layers=vgan_conf.G_LAYERS,
                      g_activation=vgan_conf.G_ACTIVATION,
                      g_dropout=vgan_conf.G_DROPOUT,
                      g_optimizer=vgan_conf.G_OPTIMIZER,
                      d_layers=vgan_conf.D_LAYERS,
                      d_activation=vgan_conf.D_ACTIVATION,
                      d_dropout=vgan_conf.D_DROPOUT,
                      d_optimizer=vgan_conf.D_OPTIMIZER,
                      k=vgan_conf.K,
                      logging_freq=vgan_conf.LOGGING_FREQ,
                      logging_level=logging_level,
                      max_checkpoints=vgan_conf.MAX_CHECKPOINTS,
                      save_freq=vgan_conf.SAVE_FREQ)


def _create_wasserstein_gan(data_shape,
                            noise_size,
                            session,
                            name='WassersteinGAN',
                            cond_in_size=None,
                            data_range=(-1, 1),
                            output_dir=None,
                            separate_logging=True):
    if not output_dir:
        output_dir = wgan_conf.OUTPUT_DIRECTORY

    logging_level = wgan_conf.LOGGING_LEVEL if separate_logging else None
    return WassersteinGAN(data_shape,
                          noise_size,
                          session,
                          output_dir,
                          name=name,
                          cond_in_size=cond_in_size,
                          data_range=data_range,
                          dtype=wgan_conf.DTYPE,
                          g_layers=wgan_conf.G_LAYERS,
                          g_activation=wgan_conf.G_ACTIVATION,
                          g_dropout=wgan_conf.G_DROPOUT,
                          g_optimizer=wgan_conf.G_OPTIMIZER,
                          d_layers=wgan_conf.D_LAYERS,
                          d_activation=wgan_conf.D_ACTIVATION,
                          d_dropout=wgan_conf.D_DROPOUT,
                          d_optimizer=wgan_conf.D_OPTIMIZER,
                          k=wgan_conf.K,
                          clip_bounds=wgan_conf.CLIP_BOUNDS,
                          logging_freq=wgan_conf.LOGGING_FREQ,
                          logging_level=logging_level,
                          max_checkpoints=wgan_conf.MAX_CHECKPOINTS,
                          save_freq=wgan_conf.SAVE_FREQ)


def _create_wasserstein_gan_gp(data_shape,
                               noise_size,
                               session,
                               name='WassersteinGAN with GP',
                               cond_in_size=None,
                               data_range=(-1, 1),
                               output_dir=None,
                               separate_logging=True):
    if not output_dir:
        output_dir = wgan_gp_conf.OUTPUT_DIRECTORY

    logging_level = wgan_gp_conf.LOGGING_LEVEL if separate_logging else None
    return WassersteinGANGP(data_shape,
                            noise_size,
                            session,
                            output_dir,
                            name=name,
                            cond_in_size=cond_in_size,
                            data_range=data_range,
                            dtype=wgan_gp_conf.DTYPE,
                            g_layers=wgan_gp_conf.G_LAYERS,
                            g_activation=wgan_gp_conf.G_ACTIVATION,
                            g_dropout=wgan_gp_conf.G_DROPOUT,
                            g_optimizer=wgan_gp_conf.G_OPTIMIZER,
                            d_layers=wgan_gp_conf.D_LAYERS,
                            d_activation=wgan_gp_conf.D_ACTIVATION,
                            d_dropout=wgan_gp_conf.D_DROPOUT,
                            d_optimizer=wgan_gp_conf.D_OPTIMIZER,
                            k=wgan_gp_conf.K,
                            penalty_coeff=wgan_gp_conf.PENALTY_COEFF,
                            logging_freq=wgan_gp_conf.LOGGING_FREQ,
                            logging_level=logging_level,
                            max_checkpoints=wgan_gp_conf.MAX_CHECKPOINTS,
                            save_freq=wgan_gp_conf.SAVE_FREQ)


_STR_TO_GAN = {
    'vgan': _create_vanilla_gan,
    'wgan': _create_wasserstein_gan,
    'wgan_gp': _create_wasserstein_gan_gp
}
