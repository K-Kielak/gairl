import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'cond_mnist',
                                'wgan', 'k10ddrop1adam2e-4,adam2e-4,3x1024clip0.05')
NOISE_SIZE = 100
DTYPE = tf.float64
G_LAYERS = (256, 512, 1024)
G_ACTIVATION = tf.nn.leaky_relu
G_DROPOUT = 1
G_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9)
D_LAYERS = (1024, 1024, 1024)
D_ACTIVATION = tf.nn.leaky_relu
D_DROPOUT = 1
D_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9)
K = 10
CLIP_BOUNDS = (-0.05, 0.05)
LOGGING_FREQ = 50
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
