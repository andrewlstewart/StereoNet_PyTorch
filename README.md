# StereoNet implemented in PyTorch

**Currently training (2021-09-09) (~12hrs per epoch on my 1070)**

Implementation of the StereoNet network to compute a disparity map using stereo RGB images.

Currently training, early results are ok.  Validation EPE <img src="https://render.githubusercontent.com/render/math?math=\approx 12"> pixels.

Epoch 10:

<img src="./readme_images/Epoch_10_Val.JPG" alt="Validation image" style="width:1000px;"/>

Implemented using PyTorch Lightning as a learning exercise to learn about stereo networks, PyTorch, and PyTorch lightning.  Feel free to make any comments or recommendations for better coding practice.

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

Currently unclear

* Do I need to have a max_disps parameter to help the model learn faster/better?
