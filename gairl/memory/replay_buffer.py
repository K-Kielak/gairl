from collections import deque
from random import sample

import numpy as np


class ReplayBuffer:

    def __init__(self, max_capacity, min_capacity, replay_batch_size):
        """
        :param max_capacity: int; maximum number of experience
            samples that can be stored by the buffer.
        :param min_capacity: int; the minimum amount of experience
            samples that are needed to replay the experience.
        :param replay_batch_size: int; how many samples are returned
            when replaying the experience.
        """
        assert min_capacity >= replay_batch_size, \
            f'min_capacity ({min_capacity}) has to be equal or ' \
            f'higher than replay_batch_size ({replay_batch_size})'

        assert max_capacity >= min_capacity, \
            f'max_capacity ({max_capacity} has to be equal or ' \
            f'higher than min_capacity ({min_capacity})'

        self._max_capacity = max_capacity
        self._min_capacity = min_capacity
        self._replay_batch_size = replay_batch_size
        self._buffer = deque()

    def add_experience(self, start_state, action,
                       reward, next_state, is_terminal):
        """
        Takes experience s_{t}, a_{t}, r_{t+1}, s_{t+1}, it_{t+1}
            and saves it to the buffer.
        """
        self._buffer.appendleft((start_state, action, reward,
                                 next_state, is_terminal))
        if len(self._buffer) > self._max_capacity:
            self._buffer.pop()

    def replay_experience(self):
        """
        :return: self._replay_batch_size randomly sampled experience
            tuples (s_{t}, a_{t}, r_{t+1}, s_{t+1}, it_{t+1}) from the buffer.
        """
        if len(self._buffer) < self._min_capacity:
            return None

        samples = sample(self._buffer, self._replay_batch_size)
        return np.array(samples)
