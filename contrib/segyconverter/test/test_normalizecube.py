# pylint: disable=import-error

"""
Test data normalization in listric2 data
"""
import os
import numpy as np
import utils.normalize_cube as norm_cube
import pytest

INPUT_FOLDER = './contrib/segyconverter/test/test_data'
MAX_RANGE = 1
MIN_RANGE = 0
K = 12
# Global standard deviation of normalsegy volume
STDDEV = 11.557824


def test_normalize_cube():
    """
        Test method that normalize one cube by checking if normalized values are within [max, min] range.
    """
    cube = np.load(os.path.join(INPUT_FOLDER, 'normalsegy.npy'))
    min_clip, max_clip, scale = norm_cube.compute_statistics(STDDEV, MAX_RANGE, K)
    norm_block = norm_cube.normalize_cube(cube, min_clip, max_clip, scale, MIN_RANGE, MAX_RANGE)
    assert np.amax(norm_block) <= MAX_RANGE
    assert np.amin(norm_block) >= MIN_RANGE


def test_norm_value():
    # Check if normalized value is calculated correctly
    min_clip = -18469.875210304104
    max_clip = 18469.875210304104
    scale = 2.707110872741882e-05
    input_value = 2019
    expected_norm_value = 0.5546565685206586
    norm_v = norm_cube.norm_value(input_value, min_clip, max_clip, MIN_RANGE, MAX_RANGE, scale)
    assert norm_v == pytest.approx(expected_norm_value, rel=1e-3)

    # Check if normalized value is within [MIN_RANGE, MAX_RANGE]
    cube = np.load(os.path.join(INPUT_FOLDER, 'normalsegy.npy'))
    v = cube[10, 50, 5]
    min_clip, max_clip, scale = norm_cube.compute_statistics(STDDEV, MAX_RANGE, K)
    norm_v = norm_cube.norm_value(v, min_clip, max_clip, MIN_RANGE, MAX_RANGE, scale)
    assert norm_v <= MAX_RANGE
    assert norm_v >= MIN_RANGE

    pytest.raises(Exception, norm_cube.norm_value, v, min_clip * 10, max_clip * 10, MIN_RANGE, MAX_RANGE, scale * 10)


def test_compute_statistics():
    # Check if statistics are calculated correctly for provided stddev, max_range and k values
    expected_min_clip = -138.693888
    expected_max_clip = 138.693888
    expected_scale = 0.003605061529459755
    min_clip, max_clip, scale = norm_cube.compute_statistics(STDDEV, MAX_RANGE, K)
    assert expected_min_clip == pytest.approx(min_clip, rel=1e-3)
    assert expected_max_clip == pytest.approx(max_clip, rel=1e-3)
    assert expected_scale == pytest.approx(scale, rel=1e-3)
    # Testing division by zero
    pytest.raises(Exception, norm_cube.compute_statistics, STDDEV, MAX_RANGE, 0)
    pytest.raises(Exception, norm_cube.compute_statistics, 0, MAX_RANGE, 0)


def test_main():
    cube = np.load(os.path.join(INPUT_FOLDER, 'normalsegy.npy'))
    norm_block = norm_cube.main(cube, STDDEV, K, MIN_RANGE, MAX_RANGE)
    assert np.amax(norm_block) <= MAX_RANGE
    assert np.amin(norm_block) >= MIN_RANGE

    pytest.raises(Exception, norm_cube.main, cube, STDDEV, 0, MIN_RANGE, MAX_RANGE)
    pytest.raises(Exception, norm_cube.main, cube, 0, K, MIN_RANGE, MAX_RANGE)

    invalid_cube = np.empty_like(cube)
    invalid_cube[:] = np.nan
    pytest.raises(Exception, norm_cube.main, invalid_cube, STDDEV, 0, MIN_RANGE, MAX_RANGE)
