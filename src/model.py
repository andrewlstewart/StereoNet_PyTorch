"""
Written by Andrew Stewart (andrewlstewart@gmail.com)

Two repos were relied on heavily to inform the network (along with the actual paper)
Original paper: https://arxiv.org/abs/1807.08865
X-StereoLab: https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py
ZhiXuanLi: https://github.com/zhixuanli/StereoNet/blob/f5576689e66e8370b78d9646c00b7e7772db0394/models/stereonet.py

I believe ZhiXuanLi's repo follows the paper best up until line 107 (note their CostVolume computation is incorrect)
    https://github.com/zhixuanli/StereoNet/issues/12#issuecomment-508327106

X-StereoLab is good up until line 180.  X-StereoLabl return both the up sampled and refined independently and don't perform the final ReLU.

I believe the implementation that I have written takes the best of both repos and follows the paper most closely.

Noteably, the argmin'd disparity is computed prior to the bilinear interpolation (follows X-Stereo but not ZhiXuanLi, the latter do it reverse order).

"""

from typing import Tuple, Dict, List
from numbers import Number
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class StereoNet(pl.LightningModule):
    def __init__(self, k_downsampling_layers: int = 4, k_refinement_layers: int = 3, candidate_disparities: int = 192):
        super().__init__()
        self.k_downsampling_layers = k_downsampling_layers
        self.k_refinement_layers = k_refinement_layers
        self.max_disps = (candidate_disparities+1) // (2**k_downsampling_layers)

        # Feature network
        self.feature_extractor = FeatureExtractor(in_channels=3, out_channels=32, k_downsampling_layers=self.k_downsampling_layers)

        # Cost volume
        self.cost_volumizer = CostVolume(in_channels=32, out_channels=32, max_disps=self.max_disps)

        # Hierarchical Refinement: Edge-Aware Upsampling
        self.refiners = nn.ModuleList()
        for _ in range(self.k_refinement_layers):
            self.refiners.append(Refinement())

    def forward_pyramid(self, x: Tuple[torch.Tensor]) -> List[torch.Tensor]:
        left, right = x

        left_embedding = self.feature_extractor(left)
        right_embedding = self.feature_extractor(right)

        cost = self.cost_volumizer((left_embedding, right_embedding))

        disparity_low = soft_argmin(cost, self.max_disps)

        disparities = []
        disparities.append(disparity_low)
        for idx, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.k_refinement_layers) / (2**idx)
            new_h, new_w = int(left.size()[2]//scale), int(left.size()[3]//scale)
            left_rescaled = F.interpolate(left, [new_h, new_w], mode='bilinear', align_corners=True)
            disparity_low_rescaled = F.interpolate(disparities[-1], [new_h, new_w], mode='bilinear', align_corners=True)
            refined_disparity = F.relu(refiner(torch.cat((left_rescaled, disparity_low_rescaled), dim=1)) + disparity_low_rescaled)
            disparities.append(refined_disparity)

        return disparities

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        disparities = self.forward_pyramid(x)
        return disparities[-1]  # Ultimately, only output the last refined disparity

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> torch.Tensor:
        left = batch['left']
        right = batch['right']
        disp_gt = batch['disp']

        disp_pred = self.forward_pyramid((left, right))

        for idx, disparity in enumerate(disp_pred):
            disp_pred[idx] = F.interpolate(disparity, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)

        disp_pred = torch.stack(disp_pred, dim=0)

        loss = torch.mean(robust_loss(disp_gt.tile((disp_pred.size()[0], 1, 1, 1, 1)) - disp_pred, alpha=1, c=2))

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss_epoch", F.l1_loss(F.relu(torch.sum(disp_pred, dim=0)), disp_gt), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> None:
        left = batch['left']
        right = batch['right']
        disp_gt = batch['disp']

        disp_pred = self((left, right))

        loss = F.l1_loss(disp_pred, disp_gt)
        self.log("val_loss_epoch", loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3, weight_decay=0.0001)
        lr_dict = {"scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1),
                   "interval": "epoch",
                   "frequency": 1,
                   "name": "ExponentialDecayLR"}
        config = {"optimizer": optimizer, "lr_scheduler": lr_dict}
        return config


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        # Refer to https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py
        reference_embedding, target_embedding = x

        b, c, h, w = reference_embedding.size()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    disparity_grid = torch.arange(0, max_disps).reshape((max_disps, 1, 1)).repeat(disparity_softmax.size(
    )[0], disparity_softmax.size()[1]//max_disps, disparity_softmax.size()[2], disparity_softmax.size()[3])
    disparity_grid = disparity_grid.type_as(disparity_softmax)

    disp = torch.sum(disparity_softmax * disparity_grid, dim=1, keepdim=True)

    return disp


def robust_loss(x: torch.Tensor, alpha: Number, c: Number) -> torch.Tensor:
    """
    https://arxiv.org/abs/1701.03077
    """
    f = (abs(alpha - 2) / alpha) * (torch.pow(torch.pow(x / c, 2)/abs(alpha - 2) + 1, alpha/2) - 1)
    return f


def main():
    model = StereoNet()
    data = (torch.from_numpy(np.ones((2, 3, 540, 960), dtype=np.float32)), torch.from_numpy(np.ones((2, 3, 540, 960), dtype=np.float32)))
    output = model(data)


if __name__ == "__main__":
    main()
