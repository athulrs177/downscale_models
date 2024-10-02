# Documentation for NN model scripts

## 1) Simple CNN model
### Notebook: transposecnn2d_downscale_high_res_europ.ipynb
i. Input Layer:

        Shape: Accepts an input tensor of shape input_shape, representing the precipitation data to be downscaled, typically including time, latitude, and longitude dimensions.

ii. Transposed Convolutional Layers:

The model employs several transposed convolutional layers to gradually refine the input features for achieving higher-resolution outputs.

    Layer 1:
        Operation: Transposed Convolution (Conv2DTranspose)
        Filters: 2048
        Kernel Size: (3,3)
        Strides: (1,1)
        Activation: Leaky ReLU (α = 0.2)
        Function: Expands spatial dimensions and extracts high-level features, enabling the model to learn complex patterns in precipitation data.

    Layer 2:
        Filters: 512
        Kernel Size: (3,3)
        Strides: (1,1)
        Activation: Leaky ReLU (α = 0.2)
        Function: Enhances feature extraction, refining learned representations.

    Layer 3:
        Filters: 128
        Kernel Size: (3,3)
        Strides: (1,1)
        Activation: Leaky ReLU (α = 0.2)
        Function: Continues to extract meaningful features while preserving spatial resolution.

    Layer 4:
        Filters: 32
        Kernel Size: (3,3)
        Strides: (2,2) (Downsampling)
        Activation: Leaky ReLU (α = 0.2)
        Function: Reduces spatial dimensions by a factor of 2 while learning lower-level features critical for the upscaling process.

iii. Output Layer:

        Operation: Transposed Convolution (Conv2DTranspose)
        Filters: 1 (single-channel output)
        Kernel Size: (6,6) (Increased kernel size to avoid gaps with strides of 5)
        Strides: (5,5) (Significant upscaling)
        Activation: ReLU
        Function: Generates the downscaled precipitation output, expanding input features to the desired lower-resolution output, effectively downscaling precipitation data by a factor of 10 in both latitude and longitude.
