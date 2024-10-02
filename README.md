# Documentation for NN model scripts

## 1) Simple CNN model
### Notebook: transposecnn2d_downscale_high_res_europ.ipynb
i. Input Layer:

Shape: The model takes an input tensor of shape input_shape, which represents the precipitation data to be downscaled. This input typically includes features such as time, latitude, and longitude dimensions.

ii. Transposed Convolutional Layers:
The model consists of several transposed convolutional layers that gradually refine the input features to achieve higher-resolution outputs.

Layer 1:
    Operation: Transposed Convolution (Conv2DTranspose)
    Filters: 2048
    Kernel Size: (3,3)
    Strides: (1,1)
    Activation: Leaky ReLU (with a small slope of 0.2)
    Function: This layer expands the spatial dimensions of the input data while extracting high-level features, contributing to the model's ability to learn complex patterns in precipitation data.

Layer 2:
    Filters: 512
    Kernel Size: (3,3)
    Strides: (1,1)
    Activation: Leaky ReLU
    Function: Further enhances the feature extraction process, refining the learned representations from the previous layer.

Layer 3:
    Filters: 128
    Kernel Size: (3,3)
    Strides: (1,1)
    Activation: Leaky ReLU
    Function: Continues to extract meaningful features and aids in preserving spatial resolution.

Layer 4:
    Filters: 32
    Kernel Size: (3,3)
    Strides: (2,2) (Downsampling)
    Activation: Leaky ReLU
    Function: This layer reduces the spatial dimensions by a factor of 2, while learning lower-level features that are critical for the subsequent upscaling process.

iii. Output Layer:

Operation: Transposed Convolution (Conv2DTranspose)
Filters: 1 (single-channel output)
Kernel Size: (6,6)
Strides: (5,5) (Significant upscaling)
Activation: ReLU
Function: This final layer generates the downscaled precipitation output. The large strides and kernel size facilitate the expansion of the input features to the desired lower-resolution output, effectively downscaling the precipitation data by a factor of 10 in both latitude and longitude.
