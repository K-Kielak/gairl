import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'generation', 'mnist',
                                'wgan', 'test')
DTYPE = tf.float64
G_LAYERS = (256, 512, 1024)
G_ACTIVATION = tf.nn.leaky_relu
G_DROPOUT = 1
G_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5)
D_LAYERS = (1024, 512, 256)
D_ACTIVATION = tf.nn.leaky_relu
D_DROPOUT = 0.8
D_OPTIMIZER = tf.train.RMSPropOptimizer(learning_rate=5e-5)
K = 5
CLIP_BOUNDS = (-0.01, 0.01)
LOGGING_FREQ = 10
VISUALIZATION_FREQ = 1000
LOGGING_LEVEL = logging.INFO
MAX_CHECKPOINTS = 5
SAVE_FREQ = 1000
