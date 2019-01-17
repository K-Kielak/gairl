import os
import logging


PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'outputs')

AGENT_STR = 'dqn'
RENDER = True
LOGS_VERBOSITY = logging.INFO