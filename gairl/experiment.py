from gym.envs.classic_control import CartPoleEnv

from gairl.agents import create_agent
from gairl.config import AGENT_STR, EPISODES_NUM, MAX_STEPS_PER_EPISODE, RENDER


def main():
    env = CartPoleEnv()
    agent = create_agent(AGENT_STR, env.action_space.n,
                         env.observation_space.shape)
    for e in range(EPISODES_NUM):
        observation = env.reset()
        action = agent.step(observation)
        for t in range(MAX_STEPS_PER_EPISODE):
            if RENDER:
                env.render()

            observation, reward, done, _ = env.step(action)
            action = agent.step(observation, reward)
            if done:
                break


if __name__ == '__main__':
    main()
