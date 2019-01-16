from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    def __init__(self, actions_num, state_shape):
        assert actions_num > 0, 'actions_num has to be higher than 0'
        assert isinstance(state_shape, tuple), 'state_shape has to be a tuple'

        self._actions_num = actions_num
        self._state_shape = state_shape

    @abstractmethod
    def step(self, state, reward=None):
        """
        Asks agent to decide what action to choose given the state.
        :param state: numpy.array of self._state_shape shape
            and self._state_dtype dtype; Current state the agent is in.
        :param reward: float; Recent reward returned by the environment.
            None if beginning of a new training episode.
        :return: int a in [0,1,...,self.actions_num-1]; representing
            chosen action.
        """
        pass
