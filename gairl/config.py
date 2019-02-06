import os


PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'outputs')

AGENT_STR = 'rainbowdqn'
DELAY_BETWEEN_RENDERS = 0  # So render can be slowed down [s]
RENDER = True
