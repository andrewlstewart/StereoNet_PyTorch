Feature network:

K=3

* First conv layer: 
    * (5x5x3+1)x32
* Second conv layer:
    * (5x5x32+1)x32
* Third conv layer:
    * (5x5x32+1)x32
* ResBlocks are conv2d_0 -> BN_0 -> activation_0 -> conv2d_1 -> BN_1 -> summation -> act_1: 
    * (3x3x32+1)x32 + 32x2 + (3x3x32+1)x32 + 32x2
* Final conv layer:
    * (3x3x32+1)x32

Total trainable parameters in the feature network: 174688

Cost Volume filter network:

* Conv3D layers 1-4:
    * (3x3x3x32+1)x32+32x2
* Final conv layer:
    * (3x3x3x32+1)x1

Totaly trainable parameters in the Cost Volume filter network: 111841

Total number of trainable parameters in the unrefined network 286529.