# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import torchvision.transforms as transforms

try:
    import pyspng
except ImportError:
    pyspng = None

# ----------------------------------------------------------------------------
# Abstract base class for datasets.


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
        cache=False,  # Cache images in CPU memory?
        transform=None,  # Transform to apply to the images.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self._transform = transform

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        if self._transform:
            # Ensure the image array has the correct shape and data type
            if image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze(axis=0)
            elif image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            image = PIL.Image.fromarray(image)
            image = self._transform(image)

        return image, self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        resolution=None,  # Ensure specific resolution, None = highest available.
        use_pyspng=True,  # Use pyspng if available?
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        if self._type == "dir":
            with open(os.path.join(self._path, fname), "rb") as f:
                image = np.array(PIL.Image.open(f))
        elif self._type == "zip":
            with self._get_zipfile().open(fname, "r") as f:
                image = np.array(PIL.Image.open(f))
        else:
            raise IOError("Unknown data source type")

        # Ensure the image has 3 dimensions (CHW)
        if image.ndim == 2:  # Grayscale image
            image = image[np.newaxis, :, :]
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
            image = image.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


# class CustomImageDataset(Dataset):
#     """
#     Custom dataset class to load images and labels.
#     """

#     def __init__(self, root_dir, transform=None):
#         """
#         Args:
#             root_dir (str): Directory containing images and `dataset.json`.
#             transform (callable, optional): A function/transform to apply to the images.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []

#         # Load metadata (image paths and labels)
#         self._load_metadata()

#         self.transforms = (
#             transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(self.resolution),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                 ]
#             )
#             if transform is None
#             else transform
#         )

#     def _load_metadata(self):
#         """Load image paths and labels from the `dataset.json`."""
#         json_path = os.path.join(self.root_dir, "dataset.json")
#         if not os.path.exists(json_path):
#             raise FileNotFoundError(f"`dataset.json` not found in {self.root_dir}")

#         # Parse JSON file
#         with open(json_path, "r") as f:
#             metadata = json.load(f)

#         labels_dict = metadata.get("labels", {})
#         if not labels_dict:
#             raise ValueError("No labels found in `dataset.json`")

#         # Populate image paths and labels
#         for img_name, label in labels_dict.items():
#             img_path = os.path.join(self.root_dir, img_name)
#             if os.path.exists(img_path):
#                 self.image_paths.append(img_path)
#                 self.labels.append(label)
#             else:
#                 print(f"Warning: {img_path} does not exist and will be skipped.")

#         self.num_classes = len(set(self.labels))
#         self.resolution = Image.open(self.image_paths[0]).size[-1]
#         self.dataset_size = len(self.labels)
#         self.num_channels = 3

#     def __len__(self):
#         """Return the total number of samples."""
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """Return a single sample (image and label) at the specified index."""
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_path = self.image_paths[idx]
#         label = self.labels[idx]

#         # Load image
#         image = Image.open(img_path).convert("RGB")

#         # Apply transformations if specified
#         if self.transform:
#             image = self.transform(image)

#         return image, label
