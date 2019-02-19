from gairl.generators.vgan import vanilla_gan_config as vgan_conf
from gairl.generators.vgan.vanilla_gan import VanillaGAN
from gairl.generators.wgan.wasserstein_gan import WassersteinGAN
from gairl.generators.wgan import wasserstein_gan_config as wgan_conf


def create_gan(gan_name, data_shape, noise_size, session):
    if gan_name not in _STR_TO_GAN.keys():
        raise AttributeError(f"There's no agent like {gan_name}. You "
                             f"can choose only from {_STR_TO_GAN.keys()}")

    creation_method = _STR_TO_GAN[gan_name]
    return creation_method(data_shape, noise_size, session)


def _create_vanilla_gan(data_shape, noise_size, session):
    return VanillaGAN(data_shape,
                      noise_size,
                      session,
                      vgan_conf.OUTPUT_DIRECTORY,
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
                      visualisation_freq=vgan_conf.VISUALIZATION_FREQ,
                      logging_level=vgan_conf.LOGGING_LEVEL,
                      max_checkpoints=vgan_conf.MAX_CHECKPOINTS,
                      save_freq=vgan_conf.SAVE_FREQ)


def _create_wasserstein_gan(data_shape, noise_size, session):
    return WassersteinGAN(data_shape,
                          noise_size,
                          session,
                          wgan_conf.OUTPUT_DIRECTORY,
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
                          visualisation_freq=wgan_conf.VISUALIZATION_FREQ,
                          logging_level=wgan_conf.LOGGING_LEVEL,
                          max_checkpoints=wgan_conf.MAX_CHECKPOINTS,
                          save_freq=wgan_conf.SAVE_FREQ)


_STR_TO_GAN = {
    'vgan': _create_vanilla_gan,
    'wgan': _create_wasserstein_gan
}
