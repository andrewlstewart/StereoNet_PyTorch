"""
"""

from typing import Optional, Tuple

import torch
import torchvision.transforms as T

import stereonet.types_stereonet as ts

# class RandomResizedCrop(st.Transformer):
#     """
#     Randomly crops + resizes the left, right images and disparity map.  Applies the same random crop + resize to all 3 tensors.
#     Note: Maintains aspect ratio, only weakly maintains that the scale is between the two provided values.
#     """

#     def __init__(self, output_size: Tuple[int, int], scale: Tuple[float, float], randomizer: np.random.Generator = np.random.default_rng()):
#         self.output_size = output_size
#         self.scale = scale
#         self.randomizer = randomizer

#     def __call__(self, sample: st.Sample_Torch) -> st.Sample_Torch:
#         random_scale = self.randomizer.random() * (self.scale[1]-self.scale[0]) + self.scale[0]

#         height, width = sample['left'].size()[-2:]
#         scaled_h, scaled_w = int(height*random_scale), int(width*random_scale)

#         top = int(self.randomizer.random()*(height - scaled_h))
#         left = int(self.randomizer.random()*(width - scaled_w))

#         larger_length = self.output_size[0]/scaled_h if (self.output_size[0]*scaled_h) > (self.output_size[1]*scaled_w) else self.output_size[1]/scaled_w
#         output_h, output_w = int(larger_length*scaled_h), int(larger_length*scaled_w)

#         if np.isclose(output_h, self.output_size[0], rtol=0.01):
#             output_h = self.output_size[0]
#         if np.isclose(output_w, self.output_size[1], rtol=0.01):
#             output_w = self.output_size[1]

#         if output_h < self.output_size[0] or output_w < self.output_size[1]:
#             raise SizeRequestedIsLargerThanImage()

#         resized_top = int(self.randomizer.random()*(output_h - self.output_size[0]))
#         resized_left = int(self.randomizer.random()*(output_w - self.output_size[1]))

#         for name, image in sample.items():
#             image = T.functional.crop(image, top=top, left=left, height=scaled_h, width=scaled_w)
#             image = T.functional.resize(image, size=(output_h, output_w))
#             image = T.functional.crop(image, top=resized_top, left=resized_left, height=self.output_size[0], width=self.output_size[1])
#             sample[name] = image
#         return sample


# class RandomCrop(st.Transformer):
#     """
#     """

#     def __init__(self, scale: float, randomizer: np.random.Generator = np.random.default_rng()):
#         self.scale = scale
#         self.randomizer = randomizer

#     def __call__(self, sample: st.Sample_Torch) -> st.Sample_Torch:
#         height, width = sample['left'].size()[-2:]

#         output_height = int(self.scale*height)
#         output_width = int(self.scale*width)

#         resized_top = int(self.randomizer.random()*(height - output_height))
#         resized_left = int(self.randomizer.random()*(width - output_width))

#         for name, image in sample.items():
#             image = T.functional.crop(image, top=resized_top, left=resized_left, height=output_height, width=output_width)
#             sample[name] = image
#         return sample


class CenterCrop(ts.TorchTransformer):
    """
    """

    def __init__(self, scale: Optional[float] = None, shape: Optional[Tuple[int, int]] = None):
        self.scale = scale
        self.shape = shape

        assert (self.scale is None) != (self.shape is None)

    def __call__(self, sample: ts.Sample_Torch) -> ts.Sample_Torch:
        height, width = sample['left'].size()[-2:]
        if self.scale is not None:
            output_height = int(self.scale*height)
            output_width = int(self.scale*width)
        if self.shape is not None:
            output_height = self.shape[0]
            output_width = self.shape[1]
        cropper = T.CenterCrop((output_height, output_width))
        for name, image in sample.items():
            sample[name] = cropper(image)
        return sample


class ToTensor(ts.NumpyToTorchTransformer):
    """
    Converts the left, right, and disparity maps into FloatTensors.
    Left and right uint8 images get rescaled to [0,1] floats.
    Disparities are already floats and just get turned into tensors.
    """

    @staticmethod
    def __call__(sample: ts.Sample_Numpy) -> ts.Sample_Torch:
        torch_sample: ts.Sample_Torch = {}
        for name, image in sample.items():
            torch_sample[name] = T.functional.to_tensor(image)
        return torch_sample


class PadSampleToBatch(ts.TorchTransformer):
    """
    Unsqueezes the first dimension to be 1 when loading in single image pairs.
    """

    @staticmethod
    def __call__(sample: ts.Sample_Torch) -> ts.Sample_Torch:
        for name, image in sample.items():
            sample[name] = torch.unsqueeze(image, dim=0)
        return sample


class Resize(ts.TorchTransformer):
    """
    Resizes each of the images in a batch to a given height and width
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: ts.Sample_Torch) -> ts.Sample_Torch:
        for name, x in sample.items():
            sample[name] = T.functional.resize(x, self.size)
        return sample


# class RandomHorizontalFlip(st.Transformer):
#     """
#     Randomly flip all 3 tensors at the same time.
#     """

#     def __init__(self, flip_probability: float, randomizer: np.random.Generator = np.random.default_rng()):
#         self.prob = flip_probability
#         self.randomizer = randomizer

#     def __call__(self, sample: st.Sample_Torch) -> st.Sample_Torch:
#         if self.randomizer.random() > self.prob:
#             for name, image in sample.items():
#                 sample[name] = T.functional.hflip(image)
#         return sample


class Rescale(ts.TorchTransformer):
    """
    Rescales the left and right image tensors (initially ranged between [0, 1]) and rescales them to be between [-1, 1].
    """

    @staticmethod
    def __call__(sample: ts.Sample_Torch) -> ts.Sample_Torch:
        for name in ['left', 'right']:
            sample[name] = (sample[name] - 0.5) * 2
        return sample
