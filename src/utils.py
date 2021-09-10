"""
Helper functions for StereoNet training.

Includes a dataset object for the Scene Flow image and disparity dataset.
"""

from typing import Optional, Tuple, List, Callable, Dict, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as T
import matplotlib.pyplot as plt

import src.utils_io as utils_io


def image_loader(path: Path) -> np.ndarray:  # pylint: disable=missing-function-docstring
    return io.imread(path)


def pfm_loader(path: Path) -> np.ndarray:  # pylint: disable=missing-function-docstring
    return utils_io.readPFM(path)


class SizeRequestedIsLargerThanImage(Exception):
    """
    One (or both) of the requested dimensions is larger than the cropped image.
    """


class SceneflowDataset(Dataset):
    """
    Sceneflow dataset composed of FlyingThings3D, Driving, and Monkaa
    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Download the RGB (cleanpass) PNG image and the Disparity files

    The train set includes FlyingThings3D Train folder and all files in Driving and Monkaa folders
    The test set includes FlyingThings3D Test folder
    """

    def __init__(self,
                 root_path: Path,
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None,
                 transforms: Optional[Union[Callable, List[Callable]]] = None):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include

        if transforms and not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

        self.left_image_path, self.right_image_path, self.left_disp_path, self.right_disp_path = self.get_paths()

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index):
        left = image_loader(self.left_image_path[index])
        right = image_loader(self.right_image_path[index])

        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right, _ = pfm_loader(self.right_disp_path[index])
        disp_right = disp_right[..., np.newaxis]
        disp_right = np.ascontiguousarray(disp_right)

        sample = {'left': left, 'right': right, 'disp_left': disp_left, 'disp_right': disp_right}

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        return sample

    def get_paths(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        string_exclude: If this string appears in the parent path of an image, don't add them to the dataset (ie. 'TEST' will exclude any path with 'TEST' in Path.parts)
        string_include: If this string DOES NOT appear in the parent path of an image, don't add them to the dataset (ie. 'TEST' will require 'TEST' to be in the Path.parts)
        if shuffle is None, don't shuffle, else shuffle.
        """

        left_image_path = []
        right_image_path = []
        left_disp_path = []
        right_disp_path = []

        # For each left image, do some path manipulation to find the corresponding right
        # image and left disparity.
        for path in self.root_path.rglob('*.png'):
            if not 'left' in path.parts:
                continue

            if self.string_exclude and self.string_exclude in path.parts:
                continue
            if self.string_include and self.string_include not in path.parts:
                continue

            r_path = Path("\\".join(['right' if 'left' in part else part for part in path.parts]))
            dl_path = Path("\\".join([f'{part.replace("frames_cleanpass","")}disparity' if 'frames_cleanpass' in part else part for part in path.parts])).with_suffix('.pfm')
            dr_path = Path("\\".join([f'{part.replace("frames_cleanpass","")}disparity' if 'frames_cleanpass' in part else part for part in r_path.parts])).with_suffix('.pfm')
            # assert r_path.exists()
            # assert d_path.exists()

            if not r_path.exists() or not dl_path.exists():
                continue

            left_image_path.append(path)
            right_image_path.append(r_path)
            left_disp_path.append(dl_path)
            right_disp_path.append(dr_path)

        return (left_image_path, right_image_path, left_disp_path, right_disp_path)


class RandomResizedCrop:
    """
    Randomly crops + resizes the left, right images and disparity map.  Applies the same random crop + resize to all 3 tensors.
    Note: Maintains aspect ratio, only weakly maintains that the scale is between the two provided values.
    """

    def __init__(self, output_size: Tuple[int, int], scale: Tuple[float, float], randomizer: np.random.Generator = np.random):
        self.output_size = output_size
        self.scale = scale
        self.randomizer = randomizer

    def __call__(self, sample: Dict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
        random_scale = self.randomizer.random() * (self.scale[1]-self.scale[0]) + self.scale[0]

        h, w = sample['left'].size()[-2:]  # pylint: disable=invalid-name
        scaled_h, scaled_w = int(h*random_scale), int(w*random_scale)

        top = int(self.randomizer.random()*(h - scaled_h))
        left = int(self.randomizer.random()*(w - scaled_w))

        larger_length = self.output_size[0]/scaled_h if (self.output_size[0]*scaled_h) > (self.output_size[1]*scaled_w) else self.output_size[1]/scaled_w
        output_h, output_w = int(larger_length*scaled_h), int(larger_length*scaled_w)

        if np.isclose(output_h, self.output_size[0], rtol=0.01):
            output_h = self.output_size[0]
        if np.isclose(output_w, self.output_size[1], rtol=0.01):
            output_w = self.output_size[1]

        if output_h < self.output_size[0] or output_w < self.output_size[1]:
            raise SizeRequestedIsLargerThanImage()

        resized_top = int(self.randomizer.random()*(output_h - self.output_size[0]))
        resized_left = int(self.randomizer.random()*(output_w - self.output_size[1]))

        for name, x in sample.items():  # pylint: disable=invalid-name
            x = T.functional.crop(x, top=top, left=left, height=scaled_h, width=scaled_w)  # pylint: disable=invalid-name
            x = T.functional.resize(x, size=(output_h, output_w))  # pylint: disable=invalid-name
            x = T.functional.crop(x, top=resized_top, left=resized_left, height=self.output_size[0], width=self.output_size[1])  # pylint: disable=invalid-name
            sample[name] = x
        return sample


class RandomCrop:
    """
    """

    def __init__(self, scale: float, randomizer: np.random.Generator = np.random):
        self.scale = scale
        self.randomizer = randomizer

    def __call__(self, sample: Dict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
        height, width = sample['left'].size()[-2:]  # pylint: disable=invalid-name

        output_height = int(self.scale*height)
        output_width = int(self.scale*width)

        resized_top = int(self.randomizer.random()*(height - output_height))
        resized_left = int(self.randomizer.random()*(width - output_width))

        for name, x in sample.items():  # pylint: disable=invalid-name
            x = T.functional.crop(x, top=resized_top, left=resized_left, height=output_height, width=output_width)  # pylint: disable=invalid-name
            sample[name] = x
        return sample


class ToTensor:
    """
    Converts the left, right, and disparity maps into FloatTensors.
    Left and right uint8 images get rescaled to [0,1] floats.
    Disparities are already floats and just get turned into tensors.
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.FloatTensor]:
        for name, x in sample.items():  # pylint: disable=invalid-name
            sample[name] = T.functional.to_tensor(x)
        return sample


class RandomHorizontalFlip:
    """
    Randomly flip all 3 tensors at the same time.
    """

    def __init__(self, flip_probability: float, randomizer: np.random.Generator = np.random):
        self.prob = flip_probability
        self.randomizer = randomizer

    def __call__(self, sample: Dict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
        if self.randomizer.random() > self.prob:
            for name, x in sample.items():  # pylint: disable=invalid-name
                sample[name] = T.functional.hflip(x)
        return sample


class Rescale:
    """
    Rescales the left and right image tensors (initially ranged between [0, 1]) and rescales them to be between [-1, 1].
    """

    def __call__(self, sample: Dict[str, torch.FloatTensor]) -> Dict[str, torch.FloatTensor]:
        for name in ['left', 'right']:
            sample[name] = (sample[name] - 0.5) * 2
        return sample


def plot_figure(left: torch.Tensor, right: torch.Tensor, disp_gt: torch.Tensor, disp_pred: torch.Tensor) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)  # pylint: disable=invalid-name
    left = (torch.moveaxis(left, 0, 2) + 1) / 2
    right = (torch.moveaxis(right, 0, 2) + 1) / 2
    disp_gt = torch.moveaxis(disp_gt, 0, 2)
    disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left)
    ax[0, 1].imshow(right)
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max())
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max())  # pylint: disable=invalid-name
    ax[0, 0].title.set_text('Left')
    ax[0, 1].title.set_text('Right')
    ax[1, 0].title.set_text('Ground truth disparity')
    ax[1, 1].title.set_text('Predicted disparity')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)
    return fig
