"""
Utility functions for pytest
"""
import numpy as np
import os


def is_npy(s):
    """
    Filter check for npy files
    :param str s: file path
    :returns: True if npy
    :rtype: bool
    """
    if (s.find(".npy") == -1):
        return False
    else:
        return True


def get_npy_files(outputdir):
    """
    List npy files
    :param str outputdir: location of npy files
    :returns: npy_files
    :rtype: list
    """
    npy_files = os.listdir(outputdir)
    npy_files = list(filter(is_npy, npy_files))
    npy_files.sort()
    return npy_files


def build_volume(n_points, npy_files, file_location):
    """
    Rebuild volume from npy files. This only works for a vertical column of
    npy files. If there is a cube of files, then a new algorithm will be required to
    stitch them back together

    :param int n_points: size of cube expected in npy_files
    :param list npy_files: list of files to load into vertical volume
    :param str file_location: directory for npy files to add to array
    :returns: numpy array created by stacking the npy_file arrays vertically (third axis)
    :rtype: numpy.array
    """
    full_volume_from_file = np.zeros((n_points, n_points, n_points * len(npy_files)), dtype=np.float32)
    for i, file in enumerate(npy_files):
        data = np.load(os.path.join(file_location, file))
        full_volume_from_file[:, :, n_points * i:n_points * (i + 1)] = data
    return full_volume_from_file
