import tensorflow as tf

from gairl.memory.replay_buffer import ReplayBuffer


HIDDEN_LAYERS = [8, 16, 8, 4]
DTYPE = tf.float64
ACTIVATION_FN = tf.nn.leaky_relu
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=6.25e-5, epsilon=1.5e-4)
GRADIENT_CLIP = 1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 0.9
EPSILON_WARMUP = 10000
EPSILON_END = 0.05
EPSILON_PERIOD = 1000000
REPLAY_BUFFER = ReplayBuffer(100000, 1000, 32)
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 10000
