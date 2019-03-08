import logging
import os

import tensorflow as tf

from gairl.config import OUTPUTS_DIR


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'reinforcement', 'acrobot',
                                'gairl', '20kf50kg,dqn,mlp16,16d0.75')
GENERATIVE_MODEL_STR = 'mlp'
RL_AGENT_STR = 'dqn'
DTYPE = tf.float64
MODEL_FREE_STEPS = 20000
MODEL_TRAINING_STEPS = 50000
MODEL_MEMORY_SIZE = 1000000
MODEL_BATCH_SIZE = 256
MODEL_TEST_SIZE = 0.2
MODEL_BASED_STEPS = 10000
LOGGING_FREQUENCY = 500
LOGGING_LEVEL = logging.INFO
