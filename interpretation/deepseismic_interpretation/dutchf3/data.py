# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import warnings
import segyio
from os import path
import scipy

# bugfix for scipy imports
import scipy.misc
import numpy as np
import torch
from toolz import curry
from torch.utils import data
import logging
from deepseismic_interpretation.dutchf3.utils.batch import (
    interpolate_to_fit_data,
    parse_labels_in_image,
    get_coordinates_for_slice,
    get_grid,
    augment_flip,
    augment_rot_xy,
    augment_rot_z,
    augment_stretch,
    rand_int,
    trilinear_interpolation,
)


def _train_data_for(data_dir):
    return path.join(data_dir, "train", "train_seismic.npy")


def _train_labels_for(data_dir):
    return path.join(data_dir, "train", "train_labels.npy")


def _test1_data_for(data_dir):
    return path.join(data_dir, "test_once", "test1_seismic.npy")


def _test1_labels_for(data_dir):
    return path.join(data_dir, "test_once", "test1_labels.npy")


def _test2_data_for(data_dir):
    return path.join(data_dir, "test_once", "test2_seismic.npy")


def _test2_labels_for(data_dir):
    return path.join(data_dir, "test_once", "test2_labels.npy")


def readSEGY(filename):
    """[summary]
    Read the segy file and return the data as a numpy array and a dictionary describing what has been read in.

    Arguments:
        filename {str} -- .segy file location.
    
    Returns:
        [type] -- 3D segy data as numy array and a dictionary with metadata information
    """

    # TODO: we really need to add logging to this repo
    print("Loading data cube from", filename, "with:")

    # Read full data cube
    data = segyio.tools.cube(filename)

    # Put temporal axis first
    data = np.moveaxis(data, -1, 0)

    # Make data cube fast to acess
    data = np.ascontiguousarray(data, "float32")

    # Read meta data
    segyfile = segyio.open(filename, "r")
    print("  Crosslines: ", segyfile.xlines[0], ":", segyfile.xlines[-1])
    print("  Inlines:    ", segyfile.ilines[0], ":", segyfile.ilines[-1])
    print("  Timeslices: ", "1", ":", data.shape[0])

    # Make dict with cube-info
    data_info = {}
    data_info["crossline_start"] = segyfile.xlines[0]
    data_info["inline_start"] = segyfile.ilines[0]
    data_info["timeslice_start"] = 1  # Todo: read this from segy
    data_info["shape"] = data.shape
    # Read dt and other params needed to do create a new

    return data, data_info


def read_labels(fname, data_info):
    """
    Read labels from an image.

    Args:
        fname: filename of labelling mask (image)
        data_info: dictionary describing the data

    Returns:
        list of labels and list of coordinates
    """

    # Alternative writings for slice-type
    inline_alias = ["inline", "in-line", "iline", "y"]
    crossline_alias = ["crossline", "cross-line", "xline", "x"]
    timeslice_alias = ["timeslice", "time-slice", "t", "z", "depthslice", "depth"]

    label_imgs = []
    label_coordinates = {}

    # Find image files in folder

    tmp = fname.split("/")[-1].split("_")
    slice_type = tmp[0].lower()
    tmp = tmp[1].split(".")
    slice_no = int(tmp[0])

    if slice_type not in inline_alias + crossline_alias + timeslice_alias:
        print("File:", fname, "could not be loaded.", "Unknown slice type")
        return None

    if slice_type in inline_alias:
        slice_type = "inline"
    if slice_type in crossline_alias:
        slice_type = "crossline"
    if slice_type in timeslice_alias:
        slice_type = "timeslice"

    # Read file
    print("Loading labels for", slice_type, slice_no, "with")
    img = scipy.misc.imread(fname)
    img = interpolate_to_fit_data(img, slice_type, slice_no, data_info)
    label_img = parse_labels_in_image(img)

    # Get coordinates for slice
    coords = get_coordinates_for_slice(slice_type, slice_no, data_info)

    # Loop through labels in label_img and append to label_coordinates
    for cls in np.unique(label_img):
        if cls > -1:
            if str(cls) not in label_coordinates.keys():
                label_coordinates[str(cls)] = np.array(np.zeros([3, 0]))
            inds_with_cls = label_img == cls
            cords_with_cls = coords[:, inds_with_cls.ravel()]
            label_coordinates[str(cls)] = np.concatenate((label_coordinates[str(cls)], cords_with_cls), 1)
            print(" ", str(np.sum(inds_with_cls)), "labels for class", str(cls))
    if len(np.unique(label_img)) == 1:
        print(" ", 0, "labels", str(cls))

    # Add label_img to output
    label_imgs.append([label_img, slice_type, slice_no])

    return label_imgs, label_coordinates


def get_random_batch(
    data_cube,
    label_coordinates,
    im_size,
    batch_size,
    index,
    random_flip=False,
    random_stretch=None,
    random_rot_xy=None,
    random_rot_z=None,
):
    """
    Returns a batch of augmented samples with center pixels randomly drawn from label_coordinates

    Args:
        data_cube: 3D numpy array with floating point velocity values
        label_coordinates: 3D coordinates of the labeled training slice
        im_size: size of the 3D voxel which we're cutting out around each label_coordinate
        batch_size: size of the batch
        index: element index of this element in a batch
        random_flip: bool to perform random voxel flip
        random_stretch: bool to enable random stretch
        random_rot_xy: bool to enable random rotation of the voxel around dim-0 and dim-1
        random_rot_z: bool to enable random rotation around dim-2

    Returns:
        a tuple of batch numpy array array of data with dimension
        (batch, 1, data_cube.shape[0], data_cube.shape[1], data_cube.shape[2]) and the associated labels as an array
        of size (batch).
    """

    # always generate only one datapoint - batch_size controls class balance
    num_batch_size = 1

    # Make 3 im_size elements
    if isinstance(im_size, int):
        im_size = [im_size, im_size, im_size]

    # Output arrays
    batch = np.zeros([num_batch_size, 1, im_size[0], im_size[1], im_size[2]])
    ret_labels = np.zeros([num_batch_size])

    class_keys = list(label_coordinates)
    n_classes = len(class_keys)

    # We seek to have a balanced batch with equally many samples from each class.
    # get total number of samples per class
    samples_per_class = batch_size // n_classes
    # figure out index relative to zero (not sequentially counting points)
    index = index - batch_size * (index // batch_size)
    # figure out which class to sample for this datapoint
    class_ind = index // samples_per_class

    # Start by getting a grid centered around (0,0,0)
    grid = get_grid(im_size)

    # Apply random flip
    if random_flip:
        grid = augment_flip(grid)

    # Apply random rotations
    if random_rot_xy:
        grid = augment_rot_xy(grid, random_rot_xy)
    if random_rot_z:
        grid = augment_rot_z(grid, random_rot_z)

    # Apply random stretch
    if random_stretch:
        grid = augment_stretch(grid, random_stretch)

    # Pick random location from the label_coordinates for this class:
    coords_for_class = label_coordinates[class_keys[class_ind]]
    random_index = rand_int(0, coords_for_class.shape[1])
    coord = coords_for_class[:, random_index : random_index + 1]

    # Move grid to be centered around this location
    grid += coord

    # Interpolate samples at grid from the data:
    sample = trilinear_interpolation(data_cube, grid)

    # Insert in output arrays
    ret_labels[0] = class_ind
    batch[0, 0, :, :, :] = np.reshape(sample, (im_size[0], im_size[1], im_size[2]))

    return batch, ret_labels


class SectionLoader(data.Dataset):
    """
    Base class for section data loader
    :param str data_dir: Root directory for training/test data
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, split="train", is_transform=True, augmentations=None,
                 seismic_path=None, label_path=None):
        self.split = split
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6
        self.sections = list()

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, index):

        section_name = self.sections[index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, int(number), :]
            lbl = self.labels[:, int(number), :]

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl

    def transform(self, img, lbl):
        # to be in the BxCxHxW that PyTorch uses:
        lbl = np.expand_dims(lbl, 0)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


class VoxelLoader(data.Dataset):
    def __init__(
        self, root_path, filename, window_size=65, split="train", n_classes=2, gen_coord_list=False, len=None,
    ):

        assert split == "train" or split == "val"

        # location of the file
        self.root_path = root_path
        self.split = split
        self.n_classes = n_classes
        self.window_size = window_size
        self.coord_list = None
        self.filename = filename
        self.full_filename = path.join(root_path, filename)

        # Read 3D cube
        # NOTE: we cannot pass this data manually as serialization of data into each python process is costly,
        # so each worker has to load the data on its own.
        self.data, self.data_info = readSEGY(self.full_filename)
        if len:
            self.len = len
        else:
            self.len = self.data.size
        self.labels = None

        if gen_coord_list:
            # generate a list of coordinates to index the entire voxel
            # memory footprint of this isn't large yet, so not need to wrap as a generator
            nx, ny, nz = self.data.shape
            x_list = range(self.window_size, nx - self.window_size)
            y_list = range(self.window_size, ny - self.window_size)
            z_list = range(self.window_size, nz - self.window_size)

            print("-- generating coord list --")
            # TODO: is there any way to use a generator with pyTorch data loader?
            self.coord_list = list(itertools.product(x_list, y_list, z_list))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        # TODO: can we specify a pixel mathematically by index?
        pixel = self.coord_list[index]
        x, y, z = pixel
        # TODO: current bottleneck - can we slice out voxels any faster
        small_cube = self.data[
            x - self.window : x + self.window + 1,
            y - self.window : y + self.window + 1,
            z - self.window : z + self.window + 1,
        ]

        return small_cube[np.newaxis, :, :, :], pixel

    # TODO: do we need a transformer for voxels?
    """
    def transform(self, img, lbl):
        # to be in the BxCxHxW that PyTorch uses:
        lbl = np.expand_dims(lbl, 0)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()
    """


class TrainSectionLoader(SectionLoader):
    """
    Training data loader for sections
    :param str data_dir: Root directory for training/test data
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, split="train", is_transform=True, augmentations=None,
                 seismic_path=None, label_path=None):
        super(TrainSectionLoader, self).__init__(
            data_dir, split=split, is_transform=is_transform, augmentations=augmentations,
            seismic_path=seismic_path, label_path=label_path
        )

        if seismic_path is not None and label_path is not None:
            # Load npy files (seismc and corresponding labels) from provided
            # location (path)
            if not path.isfile(seismic_path):
                raise Exception(f"{seismic_path} does not exist")
            if not path.isfile(label_path):
                raise Exception(f"{label_path} does not exist")
            self.seismic = np.load(seismic_path)
            self.labels = np.load(label_path)
        else:
            self.seismic = np.load(_train_data_for(self.data_dir))
            self.labels = np.load(_train_labels_for(self.data_dir))

        # reading the file names for split
        txt_path = path.join(self.data_dir, "splits", "section_" + split + ".txt")
        file_list = tuple(open(txt_path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.sections = file_list


class TrainSectionLoaderWithDepth(TrainSectionLoader):
    """
    Section data loader that includes additional channel for depth
    :param str data_dir: Root directory for training/test data
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, split="train", is_transform=True, augmentations=None,
                 seismic_path=None, label_path=None):
        super(TrainSectionLoaderWithDepth, self).__init__(
            data_dir, split=split, is_transform=is_transform, augmentations=augmentations,
            seismic_path=seismic_path, label_path=label_path
        )
        self.seismic = add_section_depth_channels(self.seismic)  # NCWH

    def __getitem__(self, index):

        section_name = self.sections[index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, :, int(number), :]
            lbl = self.labels[:, int(number), :]

            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl


class TrainVoxelWaldelandLoader(VoxelLoader):
    def __init__(
        self, root_path, filename, split="train", window_size=65, batch_size=None, len=None,
    ):
        super(TrainVoxelWaldelandLoader, self).__init__(
            root_path, filename, split=split, window_size=window_size, len=len
        )

        label_fname = None
        if split == "train":
            label_fname = path.join(self.root_path, "inline_339.png")
        elif split == "val":
            label_fname = path.join(self.root_path, "inline_405.png")
        else:
            raise Exception("undefined split")

        self.class_imgs, self.coordinates = read_labels(label_fname, self.data_info)

        self.batch_size = batch_size if batch_size else 1

    def __getitem__(self, index):
        # print(index)
        batch, labels = get_random_batch(
            self.data,
            self.coordinates,
            self.window_size,
            self.batch_size,
            index,
            random_flip=True,
            random_stretch=0.2,
            random_rot_xy=180,
            random_rot_z=15,
        )

        return batch, labels


# TODO: write TrainVoxelLoaderWithDepth
TrainVoxelLoaderWithDepth = TrainVoxelWaldelandLoader


class TestSectionLoader(SectionLoader):
    """
    Test data loader for sections
    :param str data_dir: Root directory for training/test data
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, split = "test1", is_transform = True, augmentations = None,
                 seismic_path = None, label_path = None):
        super(TestSectionLoader, self).__init__(
           data_dir, split=split, is_transform=is_transform, augmentations=augmentations,
        )

        if "test1" in self.split:
            self.seismic = np.load(_test1_data_for(self.data_dir))
            self.labels = np.load(_test1_labels_for(self.data_dir))
        elif "test2" in self.split:
            self.seismic = np.load(_test2_data_for(self.data_dir))
            self.labels = np.load(_test2_labels_for(self.data_dir))
        elif seismic_path is not None and label_path is not None:
            # Load npy files (seismc and corresponding labels) from provided
            # location (path)
            if not path.isfile(seismic_path):
                raise Exception(f"{seismic_path} does not exist")
            if not path.isfile(label_path):
                raise Exception(f"{label_path} does not exist")
            self.seismic = np.load(seismic_path)
            self.labels = np.load(label_path)

        # We are in test mode. Only read the given split. The other one might not
        # be available.
        txt_path = path.join(self.data_dir, "splits", "section_" + split + ".txt")
        file_list = tuple(open(txt_path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.sections = file_list


class TestSectionLoaderWithDepth(TestSectionLoader):
    """
    Test data loader for sections that includes additional channel for depth
    :param str data_dir: Root directory for training/test data
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, split="test1", is_transform=True, augmentations=None,
                 seismic_path = None, label_path = None):
        super(TestSectionLoaderWithDepth, self).__init__(
            data_dir, split=split, is_transform=is_transform, augmentations=augmentations,
            seismic_path = seismic_path, label_path = label_path
        )
        self.seismic = add_section_depth_channels(self.seismic)  # NCWH

    def __getitem__(self, index):

        section_name = self.sections[index]
        direction, number = section_name.split(sep="_")

        if direction == "i":
            im = self.seismic[int(number), :, :, :]
            lbl = self.labels[int(number), :, :]
        elif direction == "x":
            im = self.seismic[:, :, int(number), :]
            lbl = self.labels[:, int(number), :]

            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        return im, lbl


class TestVoxelWaldelandLoader(VoxelLoader):
    def __init__(self, data_dir, split="test"):
        super(TestVoxelWaldelandLoader, self).__init__(data_dir, split=split)


# TODO: write TestVoxelLoaderWithDepth
TestVoxelLoaderWithDepth = TestVoxelWaldelandLoader


def _transform_WH_to_HW(numpy_array):
    assert len(numpy_array.shape) >= 2, "This method needs at least 2D arrays"
    return np.swapaxes(numpy_array, -2, -1)


class PatchLoader(data.Dataset):
    """
    Base Data loader for the patch-based deconvnet
    :param str data_dir: Root directory for training/test data
    :param int stride: training data stride
    :param int patch_size: Size of patch for training
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(self, data_dir, stride=30, patch_size=99, is_transform=True, augmentations=None,
                 seismic_path=None, label_path=None):
        self.data_dir = data_dir
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6
        self.patches = list()
        self.patch_size = patch_size
        self.stride = stride

    def pad_volume(self, volume):
        """
        Only used for train/val!! Not test.
        """
        return np.pad(volume, pad_width=self.patch_size, mode="constant", constant_values=255)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):

        patch_name = self.patches[index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift

        if direction == "i":
            im = self.seismic[idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
            lbl = self.labels[idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
        elif direction == "x":
            im = self.seismic[idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size]
            lbl = self.labels[idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size]

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        # to be in the BxCxHxW that PyTorch uses:
        lbl = np.expand_dims(lbl, 0)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long()


class TestPatchLoader(PatchLoader):
    """
    Test Data loader for the patch-based deconvnet
    :param str data_dir: Root directory for training/test data
    :param int stride: training data stride
    :param int patch_size: Size of patch for training
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    """
    def __init__(self, data_dir, stride=30, patch_size=99, is_transform=True, augmentations=None,
                 txt_path=None):
        super(TestPatchLoader, self).__init__(
            data_dir, stride=stride, patch_size=patch_size, is_transform=is_transform, augmentations=augmentations,
        )
        ## Warning: this is not used or tested
        raise NotImplementedError("This class is not correctly implemented.")
        self.seismic = np.load(_train_data_for(self.data_dir))
        self.labels = np.load(_train_labels_for(self.data_dir))

        # We are in test mode. Only read the given split. The other one might not
        # be available.
        # If txt_path is not provided, it will be assumed as below. Otherwise, provided path will be used for 
        # loading txt file and create patches.
        if not txt_path:
            self.split = "test1"  # TODO: Fix this can also be test2
            txt_path = path.join(self.data_dir, "splits", "patch_" + self.split + ".txt")
        patch_list = tuple(open(txt_path, "r"))
        patch_list = [id_.rstrip() for id_ in patch_list]
        self.patches = patch_list


class TrainPatchLoader(PatchLoader):
    """
    Train data loader for the patch-based deconvnet
    :param str data_dir: Root directory for training/test data
    :param int stride: training data stride
    :param int patch_size: Size of patch for training
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(
        self, data_dir, split="train", stride=30, patch_size=99, is_transform=True, augmentations=None,
        seismic_path=None, label_path=None
    ):
        super(TrainPatchLoader, self).__init__(
            data_dir, stride=stride, patch_size=patch_size, is_transform=is_transform, augmentations=augmentations,
            seismic_path=seismic_path, label_path=label_path
        )
        # self.seismic = self.pad_volume(np.load(seismic_path))
        # self.labels = self.pad_volume(np.load(labels_path))
        warnings.warn("This no longer pads the volume")
        if seismic_path is not None and label_path is not None:
            # Load npy files (seismc and corresponding labels) from provided
            # location (path)
            if not path.isfile(seismic_path):
                raise Exception(f"{seismic_path} does not exist")
            if not path.isfile(label_path):
                raise Exception(f"{label_path} does not exist")
            self.seismic = np.load(seismic_path)
            self.labels = np.load(label_path)
        else:
            self.seismic = np.load(_train_data_for(self.data_dir))
            self.labels = np.load(_train_labels_for(self.data_dir))
        # We are in train/val mode. Most likely the test splits are not saved yet,
        # so don't attempt to load them.
        self.split = split
        # reading the file names for split
        txt_path = path.join(self.data_dir, "splits", "patch_" + split + ".txt")
        patch_list = tuple(open(txt_path, "r"))
        patch_list = [id_.rstrip() for id_ in patch_list]
        self.patches = patch_list


class TrainPatchLoaderWithDepth(TrainPatchLoader):
    """
    Train data loader for the patch-based deconvnet with patch depth channel
    :param str data_dir: Root directory for training/test data
    :param int stride: training data stride
    :param int patch_size: Size of patch for training
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(
        self, data_dir, split="train", stride=30, patch_size=99, is_transform=True, augmentations=None,
        seismic_path=None, label_path=None
    ):
        super(TrainPatchLoaderWithDepth, self).__init__(
            data_dir, split=split, stride=stride, patch_size=patch_size, is_transform=is_transform, augmentations=augmentations,
            seismic_path=seismic_path, label_path=label_path
        )

    def __getitem__(self, index):

        patch_name = self.patches[index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift

        if direction == "i":
            im = self.seismic[idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
            lbl = self.labels[idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
        elif direction == "x":
            im = self.seismic[idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size]
            lbl = self.labels[idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size]
        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        # TODO: Add check for rotation augmentations and raise warning if found
        if self.augmentations is not None:
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]

        im = add_patch_depth_channels(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def _transform_CHW_to_HWC(numpy_array):
    return np.moveaxis(numpy_array, 0, -1)


def _transform_HWC_to_CHW(numpy_array):
    return np.moveaxis(numpy_array, -1, 0)


class TrainPatchLoaderWithSectionDepth(TrainPatchLoader):
    """
    Train data loader for the patch-based deconvnet section depth channel
    :param str data_dir: Root directory for training/test data
    :param int stride: training data stride
    :param int patch_size: Size of patch for training
    :param str split: split file to use for loading patches
    :param bool is_transform: Transform patch to dimensions expected by PyTorch
    :param list augmentations: Data augmentations to apply to patches
    :param str seismic_path: Override file path for seismic data
    :param str label_path: Override file path for label data
    """
    def __init__(
        self, data_dir, split="train", stride=30, patch_size=99, is_transform=True, augmentations=None,
        seismic_path=None, label_path=None
    ):
        super(TrainPatchLoaderWithSectionDepth, self).__init__(
            data_dir,
            split=split,
            stride=stride,
            patch_size=patch_size,
            is_transform=is_transform,
            augmentations=augmentations,
            seismic_path = seismic_path,
            label_path = label_path
        )
        self.seismic = add_section_depth_channels(self.seismic)

    def __getitem__(self, index):

        patch_name = self.patches[index]
        direction, idx, xdx, ddx = patch_name.split(sep="_")

        # Shift offsets the padding that is added in training
        # shift = self.patch_size if "test" not in self.split else 0
        # TODO: Remember we are cancelling the shift since we no longer pad
        shift = 0
        idx, xdx, ddx = int(idx) + shift, int(xdx) + shift, int(ddx) + shift
        if direction == "i":
            im = self.seismic[idx, :, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
            lbl = self.labels[idx, xdx : xdx + self.patch_size, ddx : ddx + self.patch_size]
        elif direction == "x":
            im = self.seismic[idx : idx + self.patch_size, :, xdx, ddx : ddx + self.patch_size]
            lbl = self.labels[idx : idx + self.patch_size, xdx, ddx : ddx + self.patch_size]
            im = np.swapaxes(im, 0, 1)  # From WCH to CWH

        im, lbl = _transform_WH_to_HW(im), _transform_WH_to_HW(lbl)

        if self.augmentations is not None:
            im = _transform_CHW_to_HWC(im)
            augmented_dict = self.augmentations(image=im, mask=lbl)
            im, lbl = augmented_dict["image"], augmented_dict["mask"]
            im = _transform_HWC_to_CHW(im)

        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl
    
    def __repr__(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        ratio = counts/np.sum(counts)
        return "\n".join(f"{lbl}: {cnt} [{rat}]"for lbl, cnt, rat in zip(unique, counts, ratio))


_TRAIN_PATCH_LOADERS = {
    "section": TrainPatchLoaderWithSectionDepth,
    "patch": TrainPatchLoaderWithDepth,
}

_TRAIN_SECTION_LOADERS = {"section": TrainSectionLoaderWithDepth}

_TRAIN_VOXEL_LOADERS = {"voxel": TrainVoxelLoaderWithDepth}


def get_patch_loader(cfg):
    assert cfg.TRAIN.DEPTH in [
        "section",
        "patch",
        "none",
    ], f"Depth {cfg.TRAIN.DEPTH} not supported for patch data. \
            Valid values: section, patch, none."
    return _TRAIN_PATCH_LOADERS.get(cfg.TRAIN.DEPTH, TrainPatchLoader)


def get_section_loader(cfg):
    assert cfg.TRAIN.DEPTH in [
        "section",
        "none",
    ], f"Depth {cfg.TRAIN.DEPTH} not supported for section data. \
        Valid values: section, none."
    return _TRAIN_SECTION_LOADERS.get(cfg.TRAIN.DEPTH, TrainSectionLoader)


def get_voxel_loader(cfg):
    assert cfg.TRAIN.DEPTH in [
        "voxel",
        "none",
    ], f"Depth {cfg.TRAIN.DEPTH} not supported for section data. \
        Valid values: voxel, none."
    return _TRAIN_SECTION_LOADERS.get(cfg.TRAIN.DEPTH, TrainVoxelWaldelandLoader)


_TEST_LOADERS = {"section": TestSectionLoaderWithDepth}


def get_test_loader(cfg):
    logger = logging.getLogger(__name__)
    logger.info(f"Test loader {cfg.TRAIN.DEPTH}")
    return _TEST_LOADERS.get(cfg.TRAIN.DEPTH, TestSectionLoader)


def add_patch_depth_channels(image_array):
    """Add 2 extra channels to a 1 channel numpy array
    One channel is a linear sequence from 0 to 1 starting from the top of the image to the bottom
    The second channel is the product of the input channel and the 'depth' channel
    
    Args:
        image_array (np.array): 1D Numpy array
    
    Returns:
        [np.array]: 3D numpy array
    """
    h, w = image_array.shape
    image = np.zeros([3, h, w])
    image[0] = image_array
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image


def add_section_depth_channels(sections_numpy):
    """Add 2 extra channels to a 1 channel section
    One channel is a linear sequence from 0 to 1 starting from the top of the section to the bottom
    The second channel is the product of the input channel and the 'depth' channel
    
    Args:
        sections_numpy (numpy array): 3D Matrix (NWH)Image tensor
    
    Returns:
        [pytorch tensor]: 3D image tensor
    """
    n, w, h = sections_numpy.shape
    image = np.zeros([3, n, w, h])
    image[0] = sections_numpy
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, :, :, row] = const
    image[2] = image[0] * image[1]
    return np.swapaxes(image, 0, 1)


def get_seismic_labels():
    return np.asarray(
        [[69, 117, 180], [145, 191, 219], [224, 243, 248], [254, 224, 144], [252, 141, 89], [215, 48, 39]]
    )


@curry
def decode_segmap(label_mask, n_classes=6, label_colours=get_seismic_labels()):
    """Decode segmentation class labels into a colour image
    Args:
        label_mask (np.ndarray): an (N,H,W) array of integer values denoting
            the class label at each spatial location.
    Returns:
        (np.ndarray): the resulting decoded color image (NCHW).
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], label_mask.shape[2], 3))
    rgb[:, :, :, 0] = r / 255.0
    rgb[:, :, :, 1] = g / 255.0
    rgb[:, :, :, 2] = b / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))
