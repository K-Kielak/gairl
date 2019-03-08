import time

import numpy as np
import tensorflow as tf
from gym.envs.classic_control import AcrobotEnv

from gairl.agents import create_agent
from gairl.config import AGENT_STR, DELAY_BETWEEN_RENDERS, RENDER


EPISODES_NUM = 5000
REWARD_RANGE = (-1, 0)


def main():
    env = AcrobotEnv()
    with tf.Session() as sess:
        s_ranges = np.concatenate((np.vstack(env.observation_space.low),
                                   np.vstack(env.observation_space.high)), 1)
        a_ranges = [[0, 1]] * env.action_space.n
        agent = create_agent(AGENT_STR, env.action_space.n,
                             env.observation_space.shape[0], sess,
                             state_ranges=s_ranges,
                             action_ranges=a_ranges,
                             reward_range=REWARD_RANGE)
        for e in range(EPISODES_NUM):
            observation = env.reset()
            action = agent.step(observation)
            while True:
                if RENDER:
                    env.render()
                    time.sleep(DELAY_BETWEEN_RENDERS)

                observation, reward, done, _ = env.step(action)
                if done:
                    agent.step(observation, reward, is_terminal=True)
                    break
                else:
                    action = agent.step(observation, reward)


if __name__ == '__main__':
    main()
