import pytest
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from gairl.neural_utils import normalize


@pytest.mark.parametrize('data_batch, data_ranges, '
                         'target_ranges, expected', [
                             (np.array([[1, 2, 3, 4, 5],
                                        [5, 4, 3, 2, 1],
                                        [3, 5, 1, 4, 2]]),
                              None,
                              (-1, 1),
                              np.array([[-1, -0.5, 0, 0.5, 1],
                                        [1, 0.5, 0, -0.5, -1],
                                        [0, 1, -1, 0.5, -0.5]])),
                             (np.array([[1, 2, 3, 4, 5],
                                        [5, 4, 3, 2, 1],
                                        [3, 5, 1, 4, 2]]),
                              (1, 5),
                              (0, 1),
                              np.array([[0, 0.25, 0.5, 0.75, 1],
                                        [1, 0.75, 0.5, 0.25, 0],
                                        [0.5, 1, 0, 0.75, 0.25]])),
                             (np.array([[1, 2, 3, 4, 5],
                                        [5, 4, 3, 2, 1],
                                        [3, 5, 1, 4, 2]]),
                              ((1, 5), (3, 5), (1, 3), (2, 4), (-5, 5)),
                              (0, 100),
                              np.array([[0, -50, 100, 100, 100],
                                        [100, 50, 100, 0, 60],
                                        [50, 100, 0, 100, 70]])),
                             (np.array([[1, 2, 3, 4, 5],
                                        [5, 4, 3, 2, 1],
                                        [3, 5, 1, 4, 2]]),
                              ((1, 5), (3, 5), (1, 3), (2, 4), (-5, 5)),
                              ((0, 1), (-5, 0), (-1, 1), (0, 0.1), (500, 1000)),
                              np.array([[0, -7.5, 1, 0.1, 1000],
                                        [1, -2.5, 1, 0, 800],
                                        [0.5, 0, -1, 0.1, 850]]))
                         ])
def test_normalize(data_batch, data_ranges, target_ranges, expected):
    # Given
    sess = tf.InteractiveSession()

    # When
    data_batch = tf.constant(data_batch, dtype=tf.float32)
    normalized_data = normalize(data_batch,
                                data_ranges=data_ranges,
                                target_ranges=target_ranges)

    # Then
    assert_array_almost_equal(sess.run(normalized_data), expected, decimal=5)
