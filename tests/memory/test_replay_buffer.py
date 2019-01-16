import random

import pytest
from hamcrest import assert_that, contains, equal_to, none

from gairl.memory.replay_buffer import ReplayBuffer


def test_init_valid1():
    # When
    rep_buff = ReplayBuffer(5, 5, 5)

    # Then
    assert_that(rep_buff._max_capacity, equal_to(5))
    assert_that(rep_buff._min_capacity, equal_to(5))
    assert_that(rep_buff._replay_batch_size, equal_to(5))


def test_init_valid2():
    # When
    rep_buff = ReplayBuffer(7, 5, 2)

    # Then
    assert_that(rep_buff._max_capacity, equal_to(7))
    assert_that(rep_buff._min_capacity, equal_to(5))
    assert_that(rep_buff._replay_batch_size, equal_to(2))


def test_init_batch_size_higher_than_min_cap():
    # When/Then
    with pytest.raises(AssertionError):
        ReplayBuffer(7, 5, 6)


def test_init_min_cap_higher_than_max_cap():
    # When/Then
    with pytest.raises(AssertionError):
        ReplayBuffer(7, 8, 3)


def test_add_experience_max_not_exceeded():
    # Given
    rep_buff = ReplayBuffer(4, 2, 2)

    # When
    rep_buff.add_experience('s1', 'a1', 'r2', 's2')
    rep_buff.add_experience('s2', 'a2', 'r3', 's3')
    rep_buff.add_experience('s3', 'a3', 'r4', 's4')
    rep_buff.add_experience('s4', 'a4', 'r5', 's5')

    # Then
    assert_that(rep_buff._buffer, contains(
        ('s4', 'a4', 'r5', 's5'),
        ('s3', 'a3', 'r4', 's4'),
        ('s2', 'a2', 'r3', 's3'),
        ('s1', 'a1', 'r2', 's2'),
    ))


def test_add_experience_max_exceeded():
    # Given
    rep_buff = ReplayBuffer(3, 2, 2)

    # When
    rep_buff.add_experience('s1', 'a1', 'r2', 's2')
    rep_buff.add_experience('s2', 'a2', 'r3', 's3')
    rep_buff.add_experience('s3', 'a3', 'r4', 's4')
    rep_buff.add_experience('s4', 'a4', 'r5', 's5')
    rep_buff.add_experience('s5', 'a5', 'r6', 's6')

    # Then
    assert_that(rep_buff._buffer, contains(
        ('s5', 'a5', 'r6', 's6'),
        ('s4', 'a4', 'r5', 's5'),
        ('s3', 'a3', 'r4', 's4')
    ))


def test_replay_experience():
    # Given
    random.seed(9812312)
    rep_buff = ReplayBuffer(3, 2, 2)

    # When
    rep_buff.add_experience('s1', 'a1', 'r2', 's2')
    rep_buff.add_experience('s2', 'a2', 'r3', 's3')
    rep_buff.add_experience('s3', 'a3', 'r4', 's4')
    rep_buff.add_experience('s4', 'a4', 'r5', 's5')
    rep_buff.add_experience('s5', 'a5', 'r6', 's6')

    # Then
    assert_that(rep_buff.replay_experience().tolist(), contains(
        ['s3', 'a3', 'r4', 's4'],
        ['s5', 'a5', 'r6', 's6']
    ))
    assert_that(rep_buff.replay_experience().tolist(), contains(
        ['s5', 'a5', 'r6', 's6'],
        ['s4', 'a4', 'r5', 's5']
    ))


def test_replay_exerience_not_enough():
    # Given
    rep_buff = ReplayBuffer(7, 6, 3)

    # When
    rep_buff.add_experience('s1', 'a1', 'r2', 's2')
    rep_buff.add_experience('s2', 'a2', 'r3', 's3')
    rep_buff.add_experience('s3', 'a3', 'r4', 's4')
    rep_buff.add_experience('s4', 'a4', 'r5', 's5')
    rep_buff.add_experience('s5', 'a5', 'r6', 's6')

    # Then
    assert_that(rep_buff.replay_experience(), none())
