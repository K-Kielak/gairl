import time

import tensorflow as tf
from gym.envs.box2d import LunarLander

from gairl.agents import create_agent
from gairl.experiments.reinforcement.config import AGENT_STR, \
    DELAY_BETWEEN_RENDERS, RENDER

EPISODES_NUM = 5000


def main():
    env = LunarLander()
    with tf.Session() as sess:
        agent = create_agent(AGENT_STR, env.action_space.n,
                             env.observation_space.shape[0], sess)
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
