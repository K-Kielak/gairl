import os


# Directories
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'outputs')
RESOURCES_DIR = os.path.join(PROJECT_ROOT_DIR, 'resources')

# Used gan
GENERATOR_STR = 'wgan_gp'
GENERATOR_VIS_FREQ = 5000

# Used reinforcement learning agent
AGENT_STR = 'gairl'

# OpenAI Gym config
DELAY_BETWEEN_RENDERS = 0  # So render can be slowed down [s]
RENDER = True
