import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'cond_mnist',
                                'vgan', 'test')
NOISE_SIZE = 100
DTYPE = tf.float64
G_LAYERS = (64, 64, 64)
G_ACTIVATION = tf.nn.leaky_relu
G_DROPOUT = 1
G_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4)
D_LAYERS = (64, 64, 64)
D_ACTIVATION = tf.nn.leaky_relu
D_DROPOUT = 0.8
D_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4)
K = 1
LOGGING_FREQ = 500
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
