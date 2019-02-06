import pytest
from hamcrest import assert_that, equal_to

from gairl.memory.prioritized_replay_buffer import _SumTree


def test_init_valid():
    # When
    tree = _SumTree(8)

    # Then
    assert_that(tree.total_priority, equal_to(0))
    assert_that(tree.priorities_range, equal_to((1, 1)))
    assert_that(tree._data, equal_to([0]*8))
    assert_that(tree._tree, equal_to([0]*15))


def test_init_capacity_not_power_2():
    # When / Then
    with pytest.raises(AssertionError):
        _SumTree(10)


def test_add_not_full():
    # Given
    tree = _SumTree(16)

    # When
    tree.add((1, 'a', 1.))
    tree.add(('b', 2, 2.))
    tree.add(195)
    tree.add((3, 3., 'c'))
    tree.add('d')
    tree.add(19287412.214121)
    tree.add(0)

    # Then
    assert_that(tree.priorities_range, equal_to((1, 1)))
    assert_that(tree._data, equal_to([
        (1, 'a', 1.), ('b', 2, 2.), 195,
        (3, 3., 'c'), 'd', 19287412.214121, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ]))
    assert_that(tree._tree, equal_to([
        7,
        7, 0,
        4, 3, 0, 0,
        2, 2, 2, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]))


def test_add_overflow():
    # Given
    tree = _SumTree(4)

    # When
    tree.add((1, 'a', 1.))
    tree.add(('b', 2, 2.))
    tree.add(195)
    tree.add((3, 3., 'c'))
    tree.add('d')
    tree.add(19287412.214121)
    tree.add(0)

    # Then
    assert_that(tree.priorities_range, equal_to((1, 1)))
    assert_that(tree._data, equal_to(['d', 19287412.214121, 0, (3, 3., 'c')]))
    assert_that(tree._tree, equal_to([4, 2, 2, 1, 1, 1, 1]))


def test_get_not_full():
    # Given
    tree = _SumTree(16)
    tree._data = [
        (1, 'a', 1.), ('b', 2, 2.), 195,
        (3, 3., 'c'), 'd', 19287412.214121, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    tree._tree = [
        22,
        22, 0,
        7, 15, 0, 0,
        4, 3, 6, 9, 0, 0, 0, 0,
        1, 3, 1, 2, 1, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]

    # When
    data1 = tree.get_data(9.3)
    data2 = tree.get_data(10.7)
    data3 = tree.get_data(5.1)
    data4 = tree.get_data(0)
    data5 = tree.get_data(22)
    data6 = tree.get_data(13.001)
    data7 = tree.get_data(1.9)

    # Then
    assert_that(data1, equal_to((19287412.214121, 5, 5)))
    assert_that(data2, equal_to((19287412.214121, 5, 5)))
    assert_that(data3, equal_to(((3, 3., 'c'), 3, 2)))
    assert_that(data4, equal_to(((1, 'a', 1.), 0, 1)))
    assert_that(data5, equal_to((0, 6, 9)))
    assert_that(data6, equal_to((0, 6, 9)))
    assert_that(data7, equal_to((('b', 2, 2.), 1, 3)))


def test_get_overflow():
    # Given
    tree = _SumTree(4)
    tree._data = ['d', 19287412.214121, 0, (3, 3., 'c')]
    tree._tree = [
        15,
        5, 10,
        4, 1, 7, 3
    ]

    # When
    data1 = tree.get_data(0.31)
    data2 = tree.get_data(4.7)
    data3 = tree.get_data(11.9999)
    data4 = tree.get_data(12.1)
    data5 = tree.get_data(15)

    # Then
    assert_that(data1, equal_to(('d', 0, 4)))
    assert_that(data2, equal_to((19287412.214121, 1, 1)))
    assert_that(data3, equal_to((0, 2, 7)))
    assert_that(data4, equal_to(((3, 3., 'c'), 3, 3)))
    assert_that(data5, equal_to(((3, 3., 'c'), 3, 3)))


def test_get_higher_than_total():
    # Given
    tree = _SumTree(4)
    tree._data = ['d', 19287412.214121, 0, (3, 3., 'c')]
    tree._tree = [
        15,
        5, 10,
        4, 1, 7, 3
    ]

    # When
    with pytest.raises(AssertionError):
        tree.get_data(15.001)


def test_update_priority_no_maxmin_change():
    # Given
    tree = _SumTree(8)
    tree._data = ['d', 19287412.214121, 0, (3, 3., 'c')]
    tree._tree = [
        16,
        6, 10,
        4, 2, 7, 3,
        1, 3, 1, 1, 4, 3, 1, 2
    ]
    tree._min_priority = 1
    tree._min_priorities_num = 4
    tree._max_priority = 4
    tree._max_priorities_num = 1

    # When
    tree.update_priority(0, 2)
    tree.update_priority(2, 2)
    tree.update_priority(5, 2)
    tree.update_priority(6, 2)

    # Then
    assert_that(tree.priorities_range, equal_to((1, 4)))
    assert_that(tree._max_priorities_num, equal_to(1))
    assert_that(tree._min_priorities_num, equal_to(1))
    assert_that(tree._data, equal_to(['d', 19287412.214121, 0, (3, 3., 'c')]))
    assert_that(tree._tree, equal_to([
        18,
        8, 10,
        5, 3, 6, 4,
        2, 3, 2, 1, 4, 2, 2, 2
    ]))


def test_update_priority_maxmin_run_out():
    # Given
    tree = _SumTree(8)
    tree._data = ['d', 19287412.214121, 0, (3, 3., 'c')]
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

    # When
    tree.update_priority(4, 3)
    tree.update_priority(2, 2)
    tree.update_priority(6, 3)
    tree.update_priority(1, 3)
    tree.update_priority(3, 2)
    tree.update_priority(5, 2)

    # Then
    assert_that(tree.priorities_range, equal_to((2, 3)))
    assert_that(tree._max_priorities_num, equal_to(3))
    assert_that(tree._min_priorities_num, equal_to(5))
    assert_that(tree._data, equal_to(['d', 19287412.214121, 0, (3, 3., 'c')]))
    assert_that(tree._tree, equal_to([
        19,
        9, 10,
        5, 4, 5, 5,
        2, 3, 2, 2, 3, 2, 3, 2
    ]))


def test_update_priority_maxmin_overwrite():
    tree = _SumTree(8)
    tree._data = ['d', 19287412.214121, 0, (3, 3., 'c')]
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

    # When
    tree.update_priority(1, 5)
    tree.update_priority(4, 0.5)
    tree.update_priority(3, 1)
    tree.update_priority(7, 5)

    # Then
    assert_that(tree.priorities_range, equal_to((0.5, 5)))
    assert_that(tree._max_priorities_num, equal_to(2))
    assert_that(tree._min_priorities_num, equal_to(1))
    assert_that(tree._data, equal_to(['d', 19287412.214121, 0, (3, 3., 'c')]))
    assert_that(tree._tree, equal_to([
        18.5,
        9, 9.5,
        7, 2, 3.5, 6,
        2, 5, 1, 1, 0.5, 3, 1, 5
    ]))
