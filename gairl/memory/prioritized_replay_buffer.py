from random import uniform

import numpy as np

from gairl.memory.replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self,
                 max_capacity,
                 min_capacity,
                 replay_batch_size,
                 epsilon=1e-5,
                 alpha=0.6,
                 init_beta=0.4,
                 beta_annealing_period=100000):
        """
        :param epsilon: float; constant added to priority of every
            sample to prevent 0 probabilities of samples replay.
        :param alpha: float; constant determining how much
            prioritization is used. 0 - uniform, 1 - fully prioritized.
        :param init_beta: float; initial value for beta variable. Beta
            corresponds to bias reducing weights of often seen samples.
        :param beta_annealing_period: int; how many replays beta will
            linearly anneal from init_beta to 1.
        """
        assert 0. <= alpha <= 1, f'alpha ({alpha}) has to be within ' \
                                 f'[0, 1] range.'
        assert 0. <= init_beta <= 1, f'init_beta ({init_beta}) has to be ' \
                                     f'within [0, 1] range.'
        super().__init__(max_capacity, min_capacity, replay_batch_size)
        self._epsilon = epsilon
        self._alpha = alpha
        self._beta = init_beta
        self._beta_step = (1 - init_beta) / beta_annealing_period
        self._buffer = _SumTree(max_capacity)

    def __str__(self):
        return f'PrioritizedReplayBuffer(' \
               f'max_capacity={self._max_capacity}, ' \
               f'min_capacity={self._min_capacity}, ' \
               f'replay_batch_size={self._replay_batch_size}, ' \
               f'epsilon={self._epsilon}, ' \
               f'alpha={self._alpha}, ' \
               f'beta={self._beta}, ' \
               f'beta_step={self._beta_step})'

    def add_experience(self, start_state, action,
                       reward, next_state, is_terminal):
        exp_tuple = (start_state, action, reward, next_state, is_terminal)
        priority = self._buffer.priorities_range[1] ** self._alpha
        self._buffer.add(exp_tuple, priority)

    def replay_experience(self, return_if_not_enough=False):
        replay_size = self._replay_batch_size
        if len(self._buffer) < self._min_capacity:
            if not return_if_not_enough:
                return None

            replay_size = len(self._buffer)

        self._beta = min([1, self._beta + self._beta_step])

        samples = []
        indices = []
        is_weights = []
        min_priority = self._buffer.priorities_range[0]
        max_isw = (1 / len(self._buffer) / min_priority) ** self._beta
        range_size = self._buffer.total_priority / replay_size
        for i in range(replay_size):
            start = int(i * range_size)
            end = int((i + 1) * range_size)
            data, index, priority = self._buffer.get_data(uniform(start, end))
            isw = (1 / len(self._buffer) / priority) ** self._beta
            norm_isw = isw / max_isw
            samples.append(data)
            indices.append(index)
            is_weights.append(norm_isw)

        return np.array(samples), indices, is_weights

    def update_priorities(self, indices, new_priorities):
        new_priorities = [p + self._epsilon for p in new_priorities]
        for i, p in zip(indices, new_priorities):
            self._buffer.update_priority(i, p ** self._alpha)

    @property
    def prioritized(self):
        return True


class _SumTree:

    def __init__(self, max_capacity):
        assert max_capacity >= 2, f'max_capacity ({max_capacity}) of ' \
                                  f'a sumtree needs to be at least 2'
        assert _is_pow2(max_capacity), f'max_capacity ({max_capacity})' \
                                       f'needs to be equal to 2^x for ' \
                                       f'any int x > 0'

        self._curr_capacity = 0
        self._max_capacity = max_capacity
        self._tree = [0] * (2 * max_capacity - 1)
        self._data = [0] * max_capacity
        self._data_index = 0

        self._min_priority = 1
        self._min_priorities_num = 0
        self._max_priority = 1
        self._max_priorities_num = 0

    def __len__(self):
        return self._curr_capacity

    def add(self, data, priority):
        self.update_priority(self._data_index, priority)
        self._data[self._data_index] = data
        self._data_index += 1
        self._curr_capacity = min(self._curr_capacity + 1, self._max_capacity)
        if self._data_index >= self._max_capacity:
            self._data_index = 0

    def update_priority(self, data_index, priority):
        tree_index = data_index + self._max_capacity - 1
        self._update_minmax_priorities(tree_index, priority)
        prior_diff = priority - self._tree[tree_index]
        self._tree[tree_index] += prior_diff
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self._tree[tree_index] += prior_diff

    def _update_minmax_priorities(self, tree_index, new_priority):
        # If new value equals to min or max, increment counts
        if new_priority == self._max_priority:
            self._max_priorities_num += 1
        if new_priority == self._min_priority:
            self._min_priorities_num += 1

        # Replace min or max with new smaller/higher value
        if new_priority > self._max_priority:
            self._max_priority = new_priority
            self._max_priorities_num = 1
        if new_priority < self._min_priority:
            self._min_priority = new_priority
            self._min_priorities_num = 1

        # If replaced value equals to min or max, decrement counts
        if self._tree[tree_index] == self._min_priority:
            self._min_priorities_num -= 1
        if self._tree[tree_index] == self._max_priority:
            self._max_priorities_num -= 1

        # If there are no more nodes with min or max, find new values
        if self._max_priorities_num == 0:  # max does not exist anymore
            data_nodes = self._tree[-self._max_capacity:] + [new_priority]
            data_nodes = [p for p in data_nodes
                          if p != self._max_priority]
            self._max_priority = max(data_nodes)
            self._max_priorities_num = data_nodes.count(self._max_priority)

        if self._min_priorities_num == 0:  # min does not exist anymore
            data_nodes = self._tree[-self._max_capacity:] + [new_priority]
            data_nodes = [p for p in data_nodes
                          if p > 0 and p != self._min_priority]
            self._min_priority = min(data_nodes)
            self._min_priorities_num = data_nodes.count(self._min_priority)

    def get_data(self, v):
        assert v <= self.total_priority

        tree_index = 0
        left_child = 2 * tree_index + 1
        right_child = left_child + 1
        while left_child < len(self._tree):
            if v <= self._tree[left_child]:
                tree_index = left_child
            else:
                tree_index = right_child
                v -= self._tree[left_child]

            left_child = 2 * tree_index + 1
            right_child = left_child + 1

        data_index = tree_index - (self._max_capacity - 1)
        return self._data[data_index], data_index, self._tree[tree_index]

    @property
    def total_priority(self):
        return self._tree[0]

    @property
    def priorities_range(self):
        return self._min_priority, self._max_priority


def _is_pow2(n):
    while n > 1:
        if n % 2 != 0:
            return False

        n /= 2

    return True
