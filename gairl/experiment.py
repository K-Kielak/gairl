from gairl.config import agent, env
from gairl.config import EPISODES_NUM, MAX_STEPS_PER_EPISODE, RENDER


def main():
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
