"""Test the extract functions against a variety of SEGY files and trace_header scenarioes
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import scripts.prepare_dutchf3 as prep_dutchf3

# Setup
OUTPUT = None
ILINE = XLINE = DEPTH = 100
ALINE = np.zeros((ILINE, XLINE, DEPTH))
STRIDE = 50
PATCH = 100
PER_VAL = 0.2
LOG_CONFIG = None


def test_get_aline_range_step_one():

    """check if it includes the step in the range if step = 1
    """
    SLICE_STEPS = 1

    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_get_aline_range_step_zero():

    """check if a ValueError exception is raised when slice_steps = 0
    """
    with pytest.raises(ValueError, match=r'range\(\) arg 3 must not be zero'):
        SLICE_STEPS = 0

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_negative_step():

    """check if a ValueError exception is raised when slice_steps = -1
    """
    with pytest.raises(ValueError, match='slice_steps cannot be a negative number'):
        SLICE_STEPS = -1

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_float_step():

    """check if a ValueError exception is raised when slice_steps = 1.1
    """
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        SLICE_STEPS = 1.

        # Test
        output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
        output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

        assert output_iline
        assert output_xline


def test_get_aline_range_single_digit_step():

    """check if it includes the step in the range if 1 < step < 10
    """
    SLICE_STEPS = 1
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_get_aline_range_double_digit_step():

    """check if it includes the step in the range if step > 10
    """
    SLICE_STEPS = 17
    # Test
    output_iline = prep_dutchf3._get_aline_range(ILINE, PER_VAL, SLICE_STEPS)
    output_xline = prep_dutchf3._get_aline_range(XLINE, PER_VAL, SLICE_STEPS)

    assert str(output_iline[0].step) == str(SLICE_STEPS)
    assert str(output_xline[0].step) == str(SLICE_STEPS)


def test_prepare_dutchf3_patch_step_1():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 1

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_patch_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,
                                           slice_steps=SLICE_STEPS, stride=STRIDE,
                                           patch=PATCH, per_val=PER_VAL,
                                           log_config=LOG_CONFIG)

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + '/patch_train.txt', header=None, names=['row', 'a', 'b'])
        patch_train = pd.DataFrame(patch_train.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test
        i_patch_train = pd.DataFrame([x for x in patch_train.x if (patch_train.aline == 'i').bool])
        x_patch_train = pd.DataFrame([x for x in patch_train.y if (patch_train.aline == 'x').bool])

        modules_i = pd.DataFrame([int(x) % SLICE_STEPS for x in i_patch_train])
        modules_x = pd.DataFrame([int(x) % SLICE_STEPS for x in x_patch_train])

        zero_i = modules_i.sum(axis=1)
        zero_x = modules_x.sum(axis=1)

        assert int(zero_i) == 0
        assert int(zero_x) == 0


def test_prepare_dutchf3_patch_step_2():

    """check a complete run for the script in case further changes are needed
    """
    SLICE_STEPS = 2

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script
        prep_dutchf3.split_patch_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,
                                           slice_steps=SLICE_STEPS, stride=STRIDE,
                                           patch=PATCH, per_val=PER_VAL,
                                           log_config=LOG_CONFIG)

        # reading the file and splitting the data
        patch_train = pd.read_csv(output + '/patch_train.txt', header=None, names=['row', 'a', 'b'])
        patch_train = pd.DataFrame(patch_train.row.str.split('_').tolist(), columns=['aline', 'x', 'y', 'z'])

        # test
        i_patch_train = pd.DataFrame([x for x in patch_train.x if (patch_train.aline == 'i').bool])
        x_patch_train = pd.DataFrame([x for x in patch_train.y if (patch_train.aline == 'x').bool])

        modules_i = pd.DataFrame([int(x) % SLICE_STEPS for x in i_patch_train])
        modules_x = pd.DataFrame([int(x) % SLICE_STEPS for x in x_patch_train])

        zero_i = modules_i.sum(axis=1)
        zero_x = modules_x.sum(axis=1)

        assert int(zero_i) == 0
        assert int(zero_x) == 0


def test_prepare_dutchf3_section_step_1():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 1

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_section_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,
                                                slice_steps=SLICE_STEPS, per_val=PER_VAL, log_config=LOG_CONFIG)

        # reading the file and splitting the data
        section_train = pd.read_csv(output + '/section_train.txt', header=None, names=['row', 'a'])
        section_train = pd.DataFrame(section_train.row.str.split('_').tolist(), columns=['aline', 'section'])

        print(section_train)

        # test
        i_section_train = pd.DataFrame([x for x in section_train.section if (section_train.aline == 'i').bool])
        x_section_train = pd.DataFrame([x for x in section_train.section if (section_train.aline == 'x').bool])

        modules_i = pd.DataFrame([int(x) % SLICE_STEPS for x in i_section_train])
        modules_x = pd.DataFrame([int(x) % SLICE_STEPS for x in x_section_train])

        zero_i = modules_i.sum(axis=1)
        zero_x = modules_x.sum(axis=1)

        assert int(zero_i) == 0
        assert int(zero_x) == 0


def test_prepare_dutchf3_section_step_2():

    """check a complete run for the script in case further changes are needed
    """
    # setting a value to SLICE_STEPS as needed to test the values
    SLICE_STEPS = 2

    # use a temp dir that will be discarded at the end of the execution
    with tempfile.TemporaryDirectory() as tmpdirname:

        # saving the file to be used by the script
        label_file = tmpdirname + '/label_file.npy'
        np.save(label_file, ALINE)

        # stting the output directory to be used by the script
        output = tmpdirname + '/split'

        # calling the main function of the script without SLICE_STEPS, to check default value
        prep_dutchf3.split_section_train_val(data_dir=tmpdirname, output_dir=output, label_file=label_file,
                                                slice_steps=SLICE_STEPS, per_val=PER_VAL, log_config=LOG_CONFIG)

        # reading the file and splitting the data
        section_train = pd.read_csv(output + '/section_train.txt', header=None, names=['row', 'a'])
        section_train = pd.DataFrame(section_train.row.str.split('_').tolist(), columns=['aline', 'section'])

        print(section_train)

        # test
        i_section_train = pd.DataFrame([x for x in section_train.section if (section_train.aline == 'i').bool])
        x_section_train = pd.DataFrame([x for x in section_train.section if (section_train.aline == 'x').bool])

        modules_i = pd.DataFrame([int(x) % SLICE_STEPS for x in i_section_train])
        modules_x = pd.DataFrame([int(x) % SLICE_STEPS for x in x_section_train])

        zero_i = modules_i.sum(axis=1)
        zero_x = modules_x.sum(axis=1)

        assert int(zero_i) == 0
        assert int(zero_x) == 0
