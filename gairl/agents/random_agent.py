from random import randint

from gairl.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, actions_num, state_shape, state_dtype):
        super().__init__(actions_num, state_shape, state_dtype)

    def step(self, state, reward=None):
        return randint(0, self._actions_num - 1)
