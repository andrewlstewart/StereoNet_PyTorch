# StereoNet implemented in PyTorch

![Tests](https://github.com/andrewlstewart/StereoNet_PyTorch/actions/workflows/tests.yml/badge.svg)

[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewlstewart/StereoNet_PyTorch/blob/main/StereoNet.ipynb)

Refer to the above Colab notebook for an example of model inference.

Install with:
```
pip install "git+https://github.com/andrewlstewart/StereoNet_PyTorch"
```


Example left image disparity generated from stereo image pair:

![output](https://user-images.githubusercontent.com/7529012/235552448-b96f6023-7349-4f97-899f-fcdc9276e573.png)

How to perform the most basic inference:

```
import numpy as np
import torch

from stereonet.model import StereoNet
import stereonet.utils_io

# Load in the image pair as numpy uint8 arrays, ensure the shapes are the same for both images for concatenation [Height, Width, Channels]
# left = stereonet.utils_io.image_loader(path_to_left_rgb_image_file)  # [Height, Width, Channel] [0, 255] uint8
# right = stereonet.utils_io.image_loader(path_to_left_rgb_image_file)  # [Height, Width, Channel] [0, 255] uint8
# min_height = min(left.shape[0], right.shape[0])
# min_width = min(left.shape[1], right.shape[1])
# tensored = [torch.permute(torch.from_numpy(array).to(torch.float32).to(device), (2, 0, 1)) for array in (left, right)]  [Channel, Height, Width] [0, 255] uint8
# cropper = torchvision.transforms.CenterCrop((min_height, min_width))
# stack = torch.concatenate(list(map(cropper, tensored)), dim=0)  # [Stacked left/right channels, Height, Width] [0, 255] float32

# Here just creating a random image
# stack = torch.randint(0, 256, size=(1, 6, 540, 960), dtype=torch.float32)  # [Batch, Stacked left/right channels, Height, Width]  6 for 3 RGB x 2 images
stack = torch.randint(0, 256, size=(1, 2, 540, 960), dtype=torch.float32)  # [Batch, Stacked left/right channels, Height, Width]  2 for 1 grayscale x 2 images

normalizer = torchvision.transforms.Normalize((111.5684, 113.6528), (61.9625, 62.0313))
normalized = normalizer(stack)

batch = torch.unsqueeze(normalized, dim=0)

# Load in the model from the trained checkpoint
# model = StereoNet.load_from_checkpoint(path_to_checkpoint)  # "C:\\users\\name\\Downloads\\epoch=21-step=696366.ckpt"

# Here just instantiate the model with random weights
model = StereoNet(in_channels=1)  # 3 channels for RGB, 1 channel for grayscale

# Set the model to eval and run the forward method without tracking gradients
model.eval()
with torch.no_grad():
    batched_prediction = model(sample)

# Remove the batch diemnsion and switch back to channels last notation
single_prediction = batched_prediction[0].detach().cpu().numpy()  # [batch, channel, height, width] -> [channel, height, width]
single_prediction = np.moveaxis(single_prediction, 0, 2)  # [channel, height, width] -> [height, width, channel]

assert (single_prediction.shape) == (540, 960, 1)
```

## Weights
KeystoneDepth checkpoint: https://www.dropbox.com/s/ffgeqyzk4kec9cf/epoch%3D21-step%3D696366.ckpt?dl=0

* Trained with this mean/std normalizer for left/right grayscale images: torchvision.transforms.Normalize((111.5684, 113.6528), (61.9625, 62.0313))
* Model was trained on grayscale images and has in_channels=1
* Max disparity parameter during training = 256 with the mask applied
* 3 downsampling (1/8 resolution) and 3 refinement layers
* Validation EPE of 1.543 for all pixels (including >256).

Older model checkpoint trained on Sceneflow corresponding with this [commit](https://github.com/andrewlstewart/StereoNet_PyTorch/tree/9c0260f270547d8001e9d637cf3a94658f805bae): https://www.dropbox.com/s/9gpjfe3r1rfch02/epoch%3D20-step%3D744533.ckpt?dl=0


* Model was trained on RGB images and has in_channels=3
* Max disparity parameter during training = 256 with the mask applied
* 3 downsampling (1/8 resolution) and 3 refinement layers
* Validation EPE of 3.93 for all pixels (including >256).

## Notes

Implementation of the StereoNet network to compute a disparity map using stereo RGB images.

Train and validation loss curves for the KeystoneDepth training run:

<img width="595" alt="KeystoneDepth_train_loss" src="https://user-images.githubusercontent.com/7529012/235552533-1758a358-bfbf-4972-a76a-a3cb3f830f8a.png">
<img width="595" alt="KeystoneDepth_val_loss" src="https://user-images.githubusercontent.com/7529012/235552536-4aa0f242-00db-4da6-b14c-5343337e4014.png">

The above training run used:
* train/val split ratio of 85%
* 1 input channel (grayscale)
* maximum disparity of 256
* 3 downsampling (1/8 resolution) and 3 refinement layers
* input image mean/std normalization using mean=(111.5684, 113.6528) & std=(61.9625, 62.0313)
* batch size of 1
* trained for a maximum of 25 epochs (lowest validation loss was at epoch 21)
* RMSProp with a learning rate of 2.54e-4
* Exponention LR scheduler with gamma=0.9
* Maximum image side length of 625 with aspect ratio preserving resizing

On a GTX 1070, wall clock time was about 5-6 hours per epoch or ~6-7 days of wall clock time to train.

This project was implemented using PyTorch Lightning + Hydra as a learning exercise to learn about stereo networks, PyTorch, PyTorch Lightning, and Hydra.  Feel free to make any comments or recommendations for better coding practice.

Currently implemented

* Downsampling feature network with `k_downsampling_layers`
* Cost volume filtering
    * When training, a left *and* right cost volume is computed with the loss arising from the mean of the losses of left and right disparity delta to ground truth.
* Hierarchical refinement with cascading `k_refinement_layers`
* Robust loss function [A General and Adaptive Robust Loss Function, Barron (2019)](https://arxiv.org/abs/1701.03077)

Two repos were relied on heavily to inform the network (along with the actual paper)

Original paper: https://arxiv.org/abs/1807.08865

X-StereoLab: https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py

ZhiXuanLi: https://github.com/zhixuanli/StereoNet/blob/f5576689e66e8370b78d9646c00b7e7772db0394/models/stereonet.py

I believe ZhiXuanLi's repo follows the paper best up until line 107 (note their CostVolume computation is incorrect)
    https://github.com/zhixuanli/StereoNet/issues/12#issuecomment-508327106

X-StereoLab is good up until line 180.  X-StereoLab return both the up sampled and refined independently and don't perform the final ReLU.

I believe the implementation that I have written takes the best of both repos and follows the paper most closely.

Noteably, the argmin'd disparity is computed prior to the bilinear interpolation (follows X-Stereo but not ZhiXuanLi, the latter do it reverse order).

Further, neither repo had a cascade of refinement networks and neither repo trained on both the left *and* right disparities.  I believe my repo has both of these correctly implemented.

The paper clearly states they use (many) batch norm layers while simultaneously using a batch size of 1.  I find this interesting.  I naively tried training on random 50% crops (same crop applied to left/right/and disparities) so that I could get more samples into a batch but I think I was losing too many features so the EPE was consistently high.  Currently, training using a single sample (left/right images and left/right disparity).  I still needed to crop down to 513x912 images in order to not run into GPU memory issues.
