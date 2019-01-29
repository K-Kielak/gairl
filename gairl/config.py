import os


PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'outputs')

AGENT_STR = 'dqn'
DELAY_BETWEEN_RENDERS = 0.1  # So render can be slowed down [s]
RENDER = True
