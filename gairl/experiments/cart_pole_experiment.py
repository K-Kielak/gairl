import tensorflow as tf
from gym.envs.classic_control import CartPoleEnv

from gairl.agents import create_agent


AGENT_STR = 'dqn'
EPISODES_NUM = 100000
EPISODE_END_REWARD = -100
MAX_STEPS_PER_EPISODE = 1000
RENDER = True


def main():
    env = CartPoleEnv()
    with tf.Session() as sess:
        agent = create_agent(AGENT_STR, env.action_space.n,
                             env.observation_space.shape, sess)
        for e in range(EPISODES_NUM):
            observation = env.reset()
            action = agent.step(observation)
            for t in range(MAX_STEPS_PER_EPISODE):
                if RENDER:
                    env.render()

                observation, reward, done, _ = env.step(action)
                if done:
                    agent.step(observation, EPISODE_END_REWARD)
                    break
                else:
                    # The longer it stays the better rewards
                    action = agent.step(observation, reward + t)


if __name__ == '__main__':
    main()
