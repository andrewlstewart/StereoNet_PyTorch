"""
Helper functions for StereoNet training.

Includes a dataset object for the Scene Flow image and disparity dataset.
"""

from typing import Optional, Tuple, List, Union, Set, Any
from pathlib import Path
import os

import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402

import stereonet.types as stt  # noqa: E402
import stereonet.utils_io as utils_io  # noqa: E402
# import stereonet.transforms as st_transforms  # noqa: E402


RNG = np.random.default_rng()


def image_loader(path: Union[Path, str]) -> npt.NDArray[np.uint8]:
    """
    Load an image from a path using skimage.io and return a np.uint8 numpy array.
    """
    img: npt.NDArray[np.uint8] = io.imread(path)
    return img


def pfm_loader(path: Union[Path, str]) -> Tuple[npt.NDArray[np.float32], float]:
    """
    Load in a PFM formated file and return a image/disparity using the Freiburg groups method.
    Only used to load in disparity maps and not regular grayscale or colour images.
    Return a tuple of the image/disparity and the scale.
    """
    pfm: Tuple[npt.NDArray[np.float32], float] = utils_io.readPFM(path)
    return pfm


class SizeRequestedIsLargerThanImage(Exception):
    """
    One (or both) of the requested dimensions is larger than the cropped image.
    """


class SceneflowDataset(Dataset[torch.Tensor]):
    """
    Sceneflow dataset composed of FlyingThings3D, Driving, and Monkaa
    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Download the RGB (cleanpass) PNG image and the Disparity files

    The train set includes FlyingThings3D Train folder and all files in Driving and Monkaa folders
    The test set includes FlyingThings3D Test folder
    """

    def __init__(self,
                 root_path: Path,
                 transforms: Optional[List[torch.nn.Module]],
                 string_exclude: Optional[str] = None,
                 string_include: Optional[str] = None
                 ):
        self.root_path = root_path
        self.string_exclude = string_exclude
        self.string_include = string_include

        self.transforms = transforms

        self.left_image_path, self.right_image_path, self.left_disp_path, self.right_disp_path = self.get_paths(self.root_path, self.string_include, self.string_exclude)

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index: int) -> torch.Tensor:
        left = image_loader(self.left_image_path[index])
        right = image_loader(self.right_image_path[index])

        disp_left, _ = pfm_loader(self.left_disp_path[index])
        disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right, _ = pfm_loader(self.right_disp_path[index])
        disp_right = disp_right[..., np.newaxis]
        disp_right = np.ascontiguousarray(disp_right)

        assert left.dtype == np.uint8
        assert right.dtype == np.uint8
        assert disp_left.dtype != np.uint8
        assert disp_right.dtype != np.uint8

        # ToTensor works differently for dtypes, for uint8 it scales to [0,1], for float32 it does not scale
        tensorer = transforms.ToTensor()
        stack = torch.concatenate(list(map(tensorer, (left, right, disp_left, disp_right))), dim=0)  # C, H, W

        if self.transforms is not None:
            for transform in self.transforms:
                stack = transform(stack)

        return stack

    @staticmethod
    def get_paths(root_path: Path, string_include: Optional[str] = None, string_exclude: Optional[str] = None) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
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
        for path in root_path.rglob('*.png'):
            if 'left' not in path.parts:
                continue

            if string_exclude and string_exclude in path.parts:
                continue
            if string_include and string_include not in path.parts:
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


class KeystoneDataset(Dataset[torch.Tensor]):
    """
    https://keystonedepth.cs.washington.edu/download
    """

    def __init__(self,
                 root_path: str,
                 image_paths: str,
                 transforms: Optional[List[torch.nn.Module]],
                 max_size: Optional[List[int]] = None,
                 ):

        self.root_path = root_path

        self.transforms = transforms

        self.left_image_path, self.right_image_path, self.left_disp_path, self.right_disp_path = [], [], [], []
        with open(image_paths, 'r') as f:
            for line in f:
                left, right, disp_left, disp_right = line.rstrip().split(',')
                self.left_image_path.append(left)
                self.right_image_path.append(right)
                self.left_disp_path.append(disp_left)
                self.right_disp_path.append(disp_right)

        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.left_image_path)

    def __getitem__(self, index: int) -> torch.Tensor:
        left = image_loader(os.path.join(self.root_path, self.left_image_path[index]))
        right = image_loader(os.path.join(self.root_path, self.right_image_path[index]))

        disp_left = cv2.imread(os.path.join(self.root_path, self.left_disp_path[index]), cv2.IMREAD_ANYDEPTH)
        # disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right = cv2.imread(os.path.join(self.root_path, self.right_disp_path[index]), cv2.IMREAD_ANYDEPTH)
        # disp_right = disp_right[..., np.newaxis]
        disp_right = np.ascontiguousarray(disp_right)

        assert left.dtype == np.uint8
        assert right.dtype == np.uint8
        assert disp_left.dtype != np.uint8
        assert disp_right.dtype != np.uint8

        min_height = min(left.shape[0], right.shape[0], disp_left.shape[0], disp_right.shape[0])
        min_width = min(left.shape[1], right.shape[1], disp_left.shape[1], disp_right.shape[1])

        # ToTensor works differently for dtypes, for uint8 it scales to [0,1], for float32 it does not scale
        tensorer = transforms.ToTensor()
        tensored = list(map(tensorer, (left, right, disp_left, disp_right)))

        # Not sure if this is the best way to do this...
        # Keystone dataset sizes between left/right/disp_left/disp_right are inconsistent
        cropper = transforms.CenterCrop((min_height, min_width))
        stack = torch.concatenate(list(map(cropper, tensored)), dim=0)  # C, H, W

        if self.transforms is not None:
            for transform in self.transforms:
                stack = transform(stack)

        height, width = stack.size()[-2:]

        # preserve aspect ratio
        if self.max_size and (height > self.max_size[0] or width > self.max_size[1]):
            original_aspect_ratio = width / height
            new_height = int(min(self.max_size[0], self.max_size[0] / original_aspect_ratio))
            new_width = int(min(self.max_size[1], original_aspect_ratio * self.max_size[1]))
            resizer = transforms.Resize(size=(new_height, new_width), antialias=True)
            stack = resizer(stack)

        return stack

    @staticmethod
    def get_paths(root_path: str, image_extensions: Set[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        if shuffle is None, don't shuffle, else shuffle.
        """

        left_image_path = []
        right_image_path = []
        left_disp_path = []
        right_disp_path = []

        # For each left image, do some path manipulation to find the corresponding right
        # image and left/right disparity.
        for root, dirs, files in os.walk(root_path):
            for file_ in files:
                name, ext = os.path.splitext(file_)
                if ext not in image_extensions:
                    continue

                if name[-1] != 'L':
                    continue

                l_path = os.path.join(root, file_)
                r_path = os.path.join(root, name[:-1] + 'R' + ext)
                assert os.path.exists(r_path)

                parent_path = os.path.dirname(os.path.dirname(root))  # idempotent
                parent_path = os.path.join(os.path.join(os.path.join(parent_path, 'processed'), 'rectified'), 'disp_info_LR')
                dl_path = os.path.join(parent_path, name[:-1] + 'L' + '.exr')
                dr_path = os.path.join(parent_path, name[:-1] + 'R' + '.exr')

                assert os.path.exists(dl_path)
                assert os.path.exists(dr_path)

                # if not r_path.exists() or not dl_path.exists():
                #     continue

                left_image_path.append(l_path)
                right_image_path.append(r_path)
                left_disp_path.append(dl_path)
                right_disp_path.append(dr_path)

        return (left_image_path, right_image_path, left_disp_path, right_disp_path)


def plot_figure(left: torch.Tensor, right: torch.Tensor, disp_gt: torch.Tensor, disp_pred: torch.Tensor) -> plt.figure:
    """
    Helper function to plot the left/right image pair from the dataset (ie. normalized between -1/+1 and c,h,w) and the
    ground truth disparity and the predicted disparity.  The disparities colour range between ground truth disparity min and max.
    """
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=2)
    left = (torch.moveaxis(left, 0, 2) + 1) / 2
    right = (torch.moveaxis(right, 0, 2) + 1) / 2
    disp_gt = torch.moveaxis(disp_gt, 0, 2)
    disp_pred = torch.moveaxis(disp_pred, 0, 2)
    ax[0, 0].imshow(left)
    ax[0, 1].imshow(right)
    ax[1, 0].imshow(disp_gt, vmin=disp_gt.min(), vmax=disp_gt.max())
    im = ax[1, 1].imshow(disp_pred, vmin=disp_gt.min(), vmax=disp_gt.max())
    ax[0, 0].title.set_text('Left')
    ax[0, 1].title.set_text('Right')
    ax[1, 0].title.set_text('Ground truth disparity')
    ax[1, 1].title.set_text('Predicted disparity')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.27])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def construct_sceneflow_dataset(cfg: stt.SceneflowData, is_training: bool) -> SceneflowDataset:
    dataset = SceneflowDataset(root_path=Path(cfg.root_path),
                               transforms=cfg.transforms,
                               string_exclude='TEST' if is_training else None,
                               string_include=None if is_training else 'TEST',
                               )
    return dataset


def construct_keystone_dataset(cfg: stt.KeystoneDepthData, is_training: bool) -> KeystoneDataset:
    root_path = Path(HydraConfig.get().runtime.cwd) / 'hydra_conf'

    training_paths = root_path / 'training_paths.txt'
    validation_paths = root_path / 'validation_paths.txt'

    if (is_training and not training_paths.exists()) or (not is_training and not validation_paths.exists()):
        assert not training_paths.exists() and not validation_paths.exists(), "Either both training and validation paths should exist, or neither should exist."
        left_image_path, right_image_path, left_disp_path, right_disp_path = KeystoneDataset.get_paths(cfg.root_path, image_extensions={'.png'})
        train_indices = set(RNG.choice(len(left_image_path), size=int(cfg.split_ratio*len(left_image_path)), replace=False))
        val_indices = set(range(len(left_image_path))) - train_indices

        for path, indices in [(training_paths, train_indices), (validation_paths, val_indices)]:
            with open(path, 'w') as f:
                root = Path(left_image_path[0]).parents[2]
                rows = [f'{Path(left_image_path[i]).relative_to(root)},{Path(right_image_path[i]).relative_to(root)},{Path(left_disp_path[i]).relative_to(root)},{Path(right_disp_path[i]).relative_to(root)}'
                        for i in indices]
                f.write("\n".join(rows))

    dataset = KeystoneDataset(root_path=cfg.root_path,
                              image_paths=str(training_paths) if is_training else str(validation_paths),
                              transforms=cfg.transforms,
                              max_size=cfg.max_size
                              )
    return dataset


def construct_dataloaders(data_config: Union[stt.Training, stt.Validation],
                          is_training: bool,
                          **kwargs: Any
                          ) -> DataLoader[torch.Tensor]:
    for datum_config in data_config.data:
        if isinstance(datum_config, stt.KeystoneDepthData):
            dataset: Dataset[torch.Tensor] = construct_keystone_dataset(datum_config, is_training)
        elif isinstance(datum_config, stt.SceneflowData):
            dataset = construct_sceneflow_dataset(datum_config, is_training)

    return DataLoader(dataset, batch_size=data_config.loader.batch_size, **kwargs)


@hydra.main(version_base=None, config_name="config")
def main(cfg: stt.StereoNetConfig) -> int:
    config: stt.StereoNetConfig = hydra.utils.instantiate(cfg, _convert_="all")['stereonet_config']

    global RNG
    RNG = np.random.default_rng(config.global_settings.random_seed)
    # _ = construct_dataloaders(data_cfg=cfg.validation.data, loader_cfg=cfg.loader, training=False)
    # data_config = [hydra.utils.instantiate(config) for config in cfg.training.data]
    if not config.training:
        raise ValueError("Training config must be specified.")
    loader = construct_dataloaders(data_config=config.training,
                                   is_training=True,
                                   shuffle=True,
                                   num_workers=1,
                                   drop_last=False)
    for _ in loader:
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
