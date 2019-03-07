import logging
import os

from gairl.config import OUTPUTS_DIR
from gairl.memory.replay_buffer import ReplayBuffer


OUTPUT_DIRECTORY = os.path.join(OUTPUTS_DIR, 'reinforcement', 'acrobot',
                                'gairl', 'dqn,mlp-default')
GENERATIVE_MODEL_STR = 'mlp'
RL_AGENT_STR = 'dqn'
REPLAY_BUFFER = ReplayBuffer(100000, 512, 256)
MODEL_FREE_STEPS = 1000
MODEL_TRAINING_STEPS = 10000
MODEL_BASED_STEPS = 10000
LOGGING_LEVEL = logging.INFO
