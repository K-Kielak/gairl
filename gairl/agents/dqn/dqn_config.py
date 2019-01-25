import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR
from gairl.memory.replay_buffer import ReplayBuffer


HIDDEN_LAYERS = [64]
OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'dqn')
DTYPE = tf.float64
ACTIVATION_FN = tf.nn.leaky_relu
OPTIMIZER = tf.train.AdamOptimizer()
GRADIENT_CLIP = 5
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1
EPSILON_WARMUP = 1000
EPSILON_END = 0.05
EPSILON_PERIOD = 10000
REPLAY_BUFFER = ReplayBuffer(100000, 500, 32)
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 2000
LOGGING_FREQ = 1000
