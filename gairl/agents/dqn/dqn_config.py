import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR
from gairl.memory.replay_buffer import ReplayBuffer


HIDDEN_LAYERS = [24, 24]
OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'dqn-2424gdsc5e-3batch256')
DTYPE = tf.float64
ACTIVATION_FN = tf.nn.leaky_relu
OPTIMIZER = tf.train.GradientDescentOptimizer(learning_rate=5e-3)
GRADIENT_CLIP = 1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1
EPSILON_WARMUP = 1000
EPSILON_END = 0.05
EPSILON_PERIOD = 10000
REPLAY_BUFFER = ReplayBuffer(10000, 512, 256)
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 500
LOGGING_FREQ = 500
