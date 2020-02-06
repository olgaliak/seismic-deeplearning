"""
Test that the current scripts can run from the command line
"""
import os
import numpy as np
import convert_segy
import test_util


MAX_RANGE = 1
MIN_RANGE = 0
ERROR_EXIT_CODE = 99
TESTFILE = "./contrib/segyconverter/test/test_data/volume1-label.segy"


def test_convert_segy_generates_single_npy(tmpdir):
    # Setup
    prefix = 'volume1'
    input_file = TESTFILE
    output_dir = tmpdir.strpath
    metadata_only = False
    iline = 189
    xline = 193
    cube_size = -1
    stride = 128
    normalize = True
    inputpath = ""

    # Test
    convert_segy.main(input_file, output_dir, prefix, iline, xline,
                      metadata_only, stride, cube_size, normalize, inputpath)

    # Validate
    npy_files = test_util.get_npy_files(tmpdir.strpath)
    assert(len(npy_files) == 1)

    min_val, max_val = _get_min_max(tmpdir.strpath)
    assert (min_val >= MIN_RANGE)
    assert (max_val <= MAX_RANGE)


def test_convert_segy_generates_multiple_npy_files(tmpdir):
    """
    Run process_all_files and checks that it returns with 0 exit code
    :param function filedir: fixture for setup and cleanup
    """

    # Setup
    prefix = 'volume1'
    input_file = TESTFILE
    output_dir = tmpdir.strpath
    metadata_only = False
    iline = 189
    xline = 193
    cube_size = 128
    stride = 128
    normalize = True
    inputpath = ""
    
    # Test
    convert_segy.main(input_file, output_dir, prefix, iline, xline,
                      metadata_only, stride, cube_size, normalize, inputpath)

    # Validate
    npy_files = test_util.get_npy_files(tmpdir.strpath)
    assert(len(npy_files) == 2)


def test_convert_segy_normalizes_data(tmpdir):
    """
    Run process_all_files and checks that it returns with 0 exit code
    :param function filedir: fixture for setup and cleanup
    """

    # Setup
    prefix = 'volume1'
    input_file = TESTFILE
    output_dir = tmpdir.strpath
    metadata_only = False
    iline = 189
    xline = 193
    cube_size = 128
    stride = 128
    normalize = True
    inputpath = ""
    
    # Test
    convert_segy.main(input_file, output_dir, prefix, iline, xline,
                      metadata_only, stride, cube_size, normalize, inputpath)

    # Validate
    npy_files = test_util.get_npy_files(tmpdir.strpath)
    assert(len(npy_files) == 2)
    min_val, max_val = _get_min_max(tmpdir.strpath)
    assert (min_val >= MIN_RANGE)
    assert (max_val <= MAX_RANGE)


def _get_min_max(outputdir):
    """
    Check # of npy files in directory
    :param str outputdir: directory to check for npy files
    :returns: min_val, max_val of values in npy files
    :rtype: int, int
    """
    min_val = 0
    max_val = 0
    npy_files = test_util.get_npy_files(outputdir)
    for file in npy_files:
        data = np.load(os.path.join(outputdir, file))
        this_min = np.amin(data)
        this_max = np.amax(data)
        if this_min < min_val:
            min_val = this_min
        if this_max > max_val:
            max_val = this_max
    return min_val, max_val
