from gym.envs.classic_control import CartPoleEnv

from gairl.agents.random_agent import RandomAgent


env = CartPoleEnv()
agent = RandomAgent(env.action_space.n,
                    env.observation_space.shape,
                    env.observation_space.dtype)


EPISODES_NUM = 1000
MAX_STEPS_PER_EPISODE = 1000

RENDER = True
