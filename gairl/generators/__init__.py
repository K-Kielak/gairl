from gairl.generators.vgan import vanilla_gan_config as gan_conf
from gairl.generators.vgan.vanilla_gan import VanillaGAN


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
                      gan_conf.OUTPUT_DIRECTORY,
                      dtype=gan_conf.DTYPE,
                      g_layers=gan_conf.G_LAYERS,
                      g_activation=gan_conf.G_ACTIVATION,
                      g_dropout=gan_conf.G_DROPOUT,
                      g_optimizer=gan_conf.G_OPTIMIZER,
                      d_layers=gan_conf.D_LAYERS,
                      d_activation=gan_conf.D_ACTIVATION,
                      d_dropout=gan_conf.D_DROPOUT,
                      d_optimizer=gan_conf.D_OPTIMIZER,
                      k=gan_conf.K,
                      logging_freq=gan_conf.LOGGING_FREQ,
                      visualisation_freq=gan_conf.VISUALIZATION_FREQ,
                      logging_level=gan_conf.LOGGING_LEVEL,
                      max_checkpoints=gan_conf.MAX_CHECKPOINTS,
                      save_freq=gan_conf.SAVE_FREQ)


_STR_TO_GAN = {
    'gan': _create_vanilla_gan,
}
