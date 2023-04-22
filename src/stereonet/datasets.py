"""
"""

from typing import Optional, List, Set, Tuple, Union, Any
from pathlib import Path
import os

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402

import stereonet.types as stt  # noqa: E402
import stereonet.utils_io as stu  # noqa: E402


RNG = np.random.default_rng()


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
                 transforms: Optional[List[torch.nn.Module]] = None,
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
        left = stu.image_loader(self.left_image_path[index])
        right = stu.image_loader(self.right_image_path[index])

        disp_left, _ = stu.PFM_loader(self.left_disp_path[index])
        disp_left = disp_left[..., np.newaxis]
        disp_left = np.ascontiguousarray(disp_left)

        disp_right, _ = stu.PFM_loader(self.right_disp_path[index])
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
                 transforms: Optional[List[torch.nn.Module]] = None,
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
        left = stu.image_loader(os.path.join(self.root_path, self.left_image_path[index]))
        right = stu.image_loader(os.path.join(self.root_path, self.right_image_path[index]))

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

        tensored = [torch.from_numpy(array).to(torch.float32) for array in (left, right, disp_left, disp_right)]

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


def get_normalization_values(dataloader: DataLoader[torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Get the min/max values and mean/std values for each channel in the training dataset.
    """
    mins: List[torch.Tensor] = []
    maxs: List[torch.Tensor] = []
    means: List[torch.Tensor] = []
    squared_means: List[torch.Tensor] = []

    for data in dataloader:
        mins.append(data.min(0)[0].min(1)[0].min(1)[0])
        maxs.append(data.max(0)[0].max(1)[0].max(1)[0])
        means.append(data.mean(dim=(0, 2, 3)))
        squared_means.append((data**2).mean(dim=(0, 2, 3)))

    all_mins = torch.vstack(mins).min(dim=0)[0]
    all_maxs = torch.vstack(maxs).max(dim=0)[0]

    all_mean = torch.vstack(means).mean(dim=0)
    all_std = torch.sqrt(torch.vstack(squared_means).mean(dim=0) - all_mean**2)

    return (all_mins, all_maxs), (all_mean, all_std)


@hydra.main(version_base=None, config_name="config")
def main(cfg: stt.StereoNetConfig) -> int:
    config: stt.StereoNetConfig = hydra.utils.instantiate(cfg, _convert_="all")['stereonet_config']

    if config.training is None:
        raise Exception("Need to provide training arguments to get normalization values.")

    # Get training dataset
    train_loader = construct_dataloaders(data_config=config.training,
                                         is_training=True,
                                         shuffle=False, num_workers=8, drop_last=False)
    (mins, maxs), (mean, std) = get_normalization_values(train_loader)

    print(f"{mins=}")
    print(f"{maxs=}")

    print(f"{mean=}")
    print(f"{std=}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
