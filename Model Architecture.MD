# Model Architecture CharGrid
 - Number of input flatten tensors = `height * width `of chargrid image
 - Number of output base channels C = 64
 
![](https://i.ibb.co/FVS46pB/2020-09-03-20-52.png)

## 1 Encoder
 - The encoder consists of five blocks where each block consists of three 3 × 3 convolutions.
 - Each concolution is made of convolution, batch normalization and ReLU activation with a dropout at the end of block.
 - The **first convolution** in a block is a stride-2 convolution to downsample the input to that block.
 - Whenever we downsample, we increase the number of output channels C of each convolution by a factor of two.
 - In block four and five of the encoder, we do not apply any downsampling, and we leave the number of channels at 512 (the first block has C = 64 channels).
 - We use dilated convolutions in block three, four, five with rates d = 2, 4, 8, respectively.

## 2 Decoder
Decoder network has two parts. semantic Segmentation decoder and bounding box regression decoder.  
Difference lies in difference lies between these two decoders only in the last layer.
### 2.1 Common parts for both semantic segmentation decoder and bounding box regression decoder
 - Each block first concatenates features from the encoder via lateral connections followed by 1×1 convolutions.
 - After this upsample `stride 2` and `kernel 3*3` transposed convolutions.
 - At each upsampling channel count is decreased by two.
 - The two decoder branches are identical in architecture up to the last convolutional block.
 - 
 ### 2.2 Semantic segmentation last layer
The decoder for semantic segmentation has an additiona convolutional layer **without** batch normalization and **with** bias and **with** softmax activation.  
Number of output channels of the last convolution corresponds to the number of classes in this decoder branch.

### 2.3 Bounding box regression last layer
 - Last layer of boundig box regression decoder is a linear one.
 - The number of output channels are `2 * number of anchors per pixel (Na)`  for the box mask and `4 * Na` for the 4 box coordinates.

## 3 Weight Initialization
The weights of all layers are initialized following He et al. (2015), except for the last ones, which are initialized with a small constant value `1e − 3` (0.001) for stabilization purposes.

## 4 Loss
In total the network uses 3 equally contributing losses.
 - Segmentation loss is the cross entropy loss
 - Boxmask loss is the binary cross entropy loss for box masks
 - Box coordinate loss is the HUber loss for box coordinate regression
Both types of cross entropy losses are augmented following the focal loss idea.  Aggressive static class weighting is also used in both types of cross entropy losses to counter the strong class imbalance between easy to find and hard to find classes.


