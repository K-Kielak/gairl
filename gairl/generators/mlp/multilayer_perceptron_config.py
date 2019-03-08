import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'cond_mnist',
                                'mlp', 'test')
DTYPE = tf.float64
LAYERS = (16, 16)
ACTIVATION = tf.nn.leaky_relu
FINAL_ACTIVATION = None
NORM_RANGE = (0, 1)
DROPOUT = 0.75
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=2e-4)
LOGGING_FREQ = 50000
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
