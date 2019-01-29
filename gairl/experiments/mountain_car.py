import time

import tensorflow as tf
from gym.envs.classic_control import MountainCarEnv

from gairl.agents import create_agent
from gairl.config import AGENT_STR, DELAY_BETWEEN_RENDERS, RENDER


EPISODES_NUM = 10000
MAX_STEPS_PER_EPISODE = 200


def main():
    env = MountainCarEnv()
    with tf.Session() as sess:
        agent = create_agent(AGENT_STR, env.action_space.n,
                             env.observation_space.shape[0], sess)
        for e in range(EPISODES_NUM):
            observation = env.reset()
            action = agent.step(observation)
            for t in range(MAX_STEPS_PER_EPISODE + 1):
                if RENDER:
                    env.render()
                    time.sleep(DELAY_BETWEEN_RENDERS)

                observation, reward, done, _ = env.step(action)
                if done or t == MAX_STEPS_PER_EPISODE:
                    agent.step(observation, reward, is_terminal=True)
                    break
                else:
                    action = agent.step(observation, reward)


if __name__ == '__main__':
    main()
