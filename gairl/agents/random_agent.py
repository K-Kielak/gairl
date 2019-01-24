from random import randrange

from gairl.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, actions_num, state_size):
        super().__init__(actions_num, state_size)

    def step(self, state, reward=None):
        return randrange(self._actions_num)
