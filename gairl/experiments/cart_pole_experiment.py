import tensorflow as tf
from gym.envs.classic_control import CartPoleEnv

from gairl.agents import create_agent
from gairl.config import AGENT_STR, LOGS_VERBOSITY, RENDER


EPISODES_NUM = 100000
MAX_STEPS_PER_EPISODE = 200


def main():
    tf.logging.set_verbosity(LOGS_VERBOSITY)
    env = CartPoleEnv()
    with tf.Session() as sess:
        agent = create_agent(AGENT_STR, env.action_space.n,
                             env.observation_space.shape[0], sess)
        for e in range(EPISODES_NUM):
            observation = env.reset()
            action = agent.step(observation)
            for t in range(MAX_STEPS_PER_EPISODE):
                if RENDER:
                    env.render()

                observation, reward, done, _ = env.step(action)
                if done:
                    agent.step(observation, reward, is_terminal=True)
                    break
                else:
                    action = agent.step(observation, reward)


if __name__ == '__main__':
    main()
