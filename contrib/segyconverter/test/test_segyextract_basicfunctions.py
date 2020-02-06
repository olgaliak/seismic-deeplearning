"""
Test the extract functions against a variety of SEGY files and trace_header scenarioes
"""
import pandas as pd
import utils.segyextract as segyextract


def test_remove_duplicates_should_keep_order():
    # setup
    list_with_dups = [1, 2, 3, 3, 5, 8, 4, 2]
    # test
    result = segyextract._remove_duplicates(list_with_dups)
    # validate
    expected_result = [1, 2, 3, 5, 8, 4]
    assert all([a == b for a, b in zip(result, expected_result)])


def test_identify_fast_direction_should_handle_xline_sequence_1():
    # setup
    df = pd.DataFrame({'i': [101, 102, 102, 102, 103, 103], 'j': [301, 301, 302, 303, 301, 302]})
    # test
    segyextract._identify_fast_direction(df, 'fast', 'slow')
    # validate
    assert(df.keys()[0] == 'fast')
    assert(df.keys()[1] == 'slow')


def test_identify_fast_direction_should_handle_xline_sequence_2():
    # setup
    df = pd.DataFrame({'i': [101, 102, 102, 102, 102, 102], 'j': [301, 301, 302, 303, 304, 305]})
    # test
    segyextract._identify_fast_direction(df, 'fast', 'slow')
    # validate
    assert(df.keys()[0] == 'fast')
    assert(df.keys()[1] == 'slow')
