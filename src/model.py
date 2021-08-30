"""
"""

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class StereoNet(pl.LightningModule):
    """
    Two repos were relied on heavily to inform the network (along with the actual paper)
    X-StereoLab: https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py
    ZhiXuanLi: https://github.com/zhixuanli/StereoNet/blob/f5576689e66e8370b78d9646c00b7e7772db0394/models/stereonet.py

    I believe ZhiXuanLi's repo follows the paper best up until line 107 (note their CostVolume computation is incorrect)
        https://github.com/zhixuanli/StereoNet/issues/12#issuecomment-508327106
    
    X-StereoLab is good up until line 180.  Further, they return both the up sampled and refined independently.

    I believe the implementation that I have written takes the best of both and follows the paper.

    Noteably, the argmin'd disparity is computed prior to the bilinear interpolation (follow X-Stereo but not ZhiXuanLi, the latter do it reverse order).

    """
    def __init__(self, k_downsampling_layers: int = 4, candidate_disparities: int = 192):
        super().__init__()
        self.k = k_downsampling_layers
        self.max_disps = (candidate_disparities+1) // (2**k_downsampling_layers)

        # Feature network
        self.feature_extractor = FeatureExtractor(in_channels=3, out_channels=32, k_downsampling_layers=self.k)

        # Cost volume
        self.cost_volumizer = CostVolume(in_channels=32, out_channels=32, max_disps=self.max_disps)

        # Hierarchical Refinement: Edge-Aware Upsampling
        self.refiner = Refinement()

    def forward(self, x):
        left, right = x

        left_embedding = self.feature_extractor(left)
        right_embedding = self.feature_extractor(right)

        cost = self.cost_volumizer((left_embedding, right_embedding))

        disparity_low = soft_argmin(cost, self.max_disps)

        disparity_initial = F.interpolate(disparity_low, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)

        disparity_refined = self.refiner(torch.cat((left, disparity_initial), dim=1))

        disparity = F.relu(disparity_initial + disparity_refined)

        return disparity

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


class FeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k_downsampling_layers: int):
        super().__init__()
        self.k = k_downsampling_layers

        net = OrderedDict()

        for block_idx in range(self.k):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
            in_channels = out_channels

        for block_idx in range(6):
            net[f'segment_1_res_{block_idx}'] = ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        net['segment_2_conv_0'] = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x):
        x = self.net(x)
        return x


class CostVolume(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_disps: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_disps = max_disps

        net = OrderedDict()

        for block_idx in range(4):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            net[f'segment_0_bn_{block_idx}'] = nn.BatchNorm3d(num_features=out_channels)
            net[f'segment_0_act_{block_idx}'] = nn.LeakyReLU(negative_slope=0.2)  # Not clear in paper if default or implied to be 0.2 like the rest

            in_channels = out_channels

        net['segment_1_conv_0'] = nn.Conv3d(in_channels=out_channels, out_channels=1, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x):
        # Refer to https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py
        reference_embedding, target_embedding = x

        b, c, h, w = reference_embedding.shape
        cost = torch.Tensor(b, c, self.max_disps, h, w).zero_()
        cost = cost.type_as(reference_embedding)  # PyTorch Lightning handles the devices
        cost[:, :, 0, :, :] = reference_embedding - target_embedding
        for idx in range(1, self.max_disps):
            cost[:, :, idx, :, idx:] = reference_embedding[:, :, :, idx:] - target_embedding[:, :, :, :-idx]
        cost = cost.contiguous()

        cost = self.net(cost)
        cost = torch.squeeze(cost, dim=1)

        return cost


class Refinement(torch.nn.Module):
    def __init__(self):
        super().__init__()

        dilations = [1, 2, 4, 8, 1, 1]

        net = OrderedDict()

        net['segment_0_conv_0'] = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)

        for block_idx, dilation in enumerate(dilations):
            net[f'segment_1_res_{block_idx}'] = ResBlock(in_channels=32, out_channels=32, kernel_size=3, padding=dilation, dilation=dilation)

        net['segment_2_conv_0'] = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.net = nn.Sequential(net)

    def forward(self, x):
        x = self.net(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        # Original Residual Unit: https://arxiv.org/pdf/1603.05027.pdf (Fig 1. Left)

        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activation_1(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)

        out = res + x
        out = self.activation_2(out)

        return out


def soft_argmin(cost: torch.Tensor, max_disps: int) -> torch.Tensor:
    disparity_softmax = F.softmax(-cost, dim=1)

    # Effectively does what disparityregression does on line 119 of the X-StereoLab permalink above
    disparity_grid = torch.range(0,max_disps-1).reshape((max_disps, 1, 1)).repeat(disparity_softmax.size()[0],disparity_softmax.size()[1]//max_disps,disparity_softmax.size()[2],disparity_softmax.size()[3])
    disparity_grid = disparity_grid.type_as(disparity_softmax)

    disp = torch.sum(disparity_softmax * disparity_grid, dim=1, keepdim=True)

    return disp


def main():
    model = StereoNet()
    data = (torch.from_numpy(np.ones((2, 3, 540, 960), dtype=np.float32)), torch.from_numpy(np.ones((2, 3, 540, 960), dtype=np.float32)))
    output = model(data)


if __name__ == "__main__":
    main()
