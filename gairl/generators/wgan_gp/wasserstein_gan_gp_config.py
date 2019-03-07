import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'cond_mnist',
                                'wgan_gp', 'test')
NOISE_SIZE = 100
DTYPE = tf.float64
G_LAYERS = (64, 64, 64)
G_ACTIVATION = tf.nn.leaky_relu
G_DROPOUT = 1
G_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9)
D_LAYERS = (64, 64, 64)
D_ACTIVATION = tf.nn.leaky_relu
D_DROPOUT = 1
D_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9)
K = 10
PENALTY_COEFF = 10
LOGGING_FREQ = 500
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
