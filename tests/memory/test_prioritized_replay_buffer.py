import random

import pytest
from hamcrest import assert_that, close_to, equal_to, none

from gairl.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
from gairl.memory.prioritized_replay_buffer import _SumTree


def test_init_valid():
    # When
    rep_buff = PrioritizedReplayBuffer(8, 3, 3, epsilon=1e-4,
                                       alpha=1, init_beta=0,
                                       beta_annealing_period=1000000)

    # Then
    assert_that(rep_buff._epsilon, equal_to(1e-4))
    assert_that(rep_buff._alpha, equal_to(1))
    assert_that(rep_buff._beta, equal_to(0))
    assert_that(rep_buff._beta_step, equal_to(0.000001))


def test_init_invalid_alpha():
    # When/Then
    with pytest.raises(AssertionError):
        PrioritizedReplayBuffer(8, 3, 3, epsilon=1e-4,
                                alpha=1.0001, init_beta=0.6,
                                beta_annealing_period=1000000)


def test_init_invalid_beta():
    # When/Then
    with pytest.raises(AssertionError):
        PrioritizedReplayBuffer(8, 3, 3, epsilon=1e-4,
                                alpha=0.45, init_beta=-0.00000001,
                                beta_annealing_period=1000000)


def test_replay_experience():
    # Given
    random.seed(7812411)
    rep_buff = PrioritizedReplayBuffer(8, 3, 3, epsilon=1e-4,
                                       alpha=1, init_beta=0.4,
                                       beta_annealing_period=1000000)
    tree = _SumTree(8)
    tree._data = [
        ('s1', 'a1', 'r2', 's2', 't2'),
        ('s2', 'a2', 'r3', 's3', 't3'),
        ('s3', 'a3', 'r4', 's4', 't4'),
        ('s4', 'a4', 'r5', 's5', 't5'),
        ('s5', 'a5', 'r6', 's6', 't6'),
        ('s6', 'a6', 'r7', 's7', 't7'),
        ('s7', 'a7', 'r8', 's8', 't8'),
        ('s8', 'a8', 'r9', 's9', 't9'),
    ]
    tree._tree = [
        18,
        8, 10,
        5, 3, 7, 3,
        2, 3, 1, 2, 4, 3, 1, 2
    ]
    tree._min_priority = 1
    tree._min_priorities_num = 2
    tree._max_priority = 4
    tree._max_priorities_num = 1
    tree._curr_capacity = 8
    rep_buff._buffer = tree

    # When
    replay1 = rep_buff.replay_experience()
    replay2 = rep_buff.replay_experience()

    # Then
    assert_that(replay1[0].tolist(), equal_to([
        ['s2', 'a2', 'r3', 's3', 't3'],
        ['s4', 'a4', 'r5', 's5', 't5'],
        ['s7', 'a7', 'r8', 's8', 't8']
    ]))
    assert_that(replay1[1], equal_to([1, 3, 6]))
    assert_that(replay1[2][0], close_to(0.64439, 0.00001))
    assert_that(replay1[2][1], close_to(0.75785, 0.00001))
    assert_that(replay1[2][2], close_to(1, 0.00001))

    assert_that(replay2[0].tolist(), equal_to([
        ['s2', 'a2', 'r3', 's3', 't3'],
        ['s5', 'a5', 'r6', 's6', 't6'],
        ['s7', 'a7', 'r8', 's8', 't8']
    ]))
    assert_that(replay2[1], equal_to([1, 4, 6]))
    assert_that(replay2[2][0], close_to(0.64439, 0.00001))
    assert_that(replay2[2][1], close_to(0.57434, 0.00001))
    assert_that(replay2[2][2], close_to(1, 0.00001))


def test_replay_exerience_not_enough():
    # Given
    rep_buff = PrioritizedReplayBuffer(8, 6, 3, epsilon=1e-4,
                                       alpha=1, init_beta=0.4,
                                       beta_annealing_period=1000000)
    rep_buff.add_experience('s1', 'a1', 'r2', 's2', 't2')
    rep_buff.add_experience('s2', 'a2', 'r3', 's3', 't3')
    rep_buff.add_experience('s3', 'a3', 'r4', 's4', 't4')
    rep_buff.add_experience('s4', 'a4', 'r5', 's5', 't5')
    rep_buff.add_experience('s5', 'a5', 'r6', 's6', 't6')

    # When
    replay = rep_buff.replay_experience()

    # Then
    assert_that(replay, none())
