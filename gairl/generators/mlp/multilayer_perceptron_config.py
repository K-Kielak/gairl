import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'cond_mnist',
                                'mlp', 'test')
DTYPE = tf.float64
LAYERS = (64, 64, 64)
ACTIVATION = tf.nn.leaky_relu
FINAL_ACTIVATION = tf.nn.tanh
NORM_RANGE = (-1, 1)
DROPOUT = 1
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4)
LOGGING_FREQ = 500
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
