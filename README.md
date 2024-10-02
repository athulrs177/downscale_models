# Documentation for NN model scripts

## 1) Simple CNN model
### Notebook: transposecnn2d_downscale_high_res_europ.ipynb
#### Model Architecture
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
        Strides: (2,2) 
        Activation: Leaky ReLU (α = 0.2)
        Function: Reduces spatial dimensions by a factor of 2 while learning lower-level features critical for the upscaling process.

iii. Output Layer:

        Operation: Transposed Convolution (Conv2DTranspose)
        Filters: 1 (single-channel output)
        Kernel Size: (6,6) (Increased kernel size to avoid gaps with strides of 5)
        Strides: (5,5) (Significant upscaling)
        Activation: ReLU
        Function: Generates the downscaled precipitation output, expanding input features to the desired lower-resolution output, effectively downscaling precipitation data by a factor of 10 in both latitude and longitude.


## 2) Deterministic Wasserstein GAN with gradient penalalties (det. WGAN-GP)
### Notebooks: 
a) wgan_det_downscale_high_res_europe.ipynb (initial few runs), 

b) wgan_det_train_part2_downscale_high_res_europe.ipynb (continue training intermediate saved models with modified params),

c) wgan_det_model_check.ipynb (generate downscaled data with trained models (intermediate and final))
#### Model Architecture
This model comprises a generator and a discriminator, implementing a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) architecture. Below are the details for each component.
i. Generator:

The generator is designed to transform low-dimensional noise or input features into high-dimensional data, typically used for generating synthetic images or precipitation data.

    Input Layer: Accepts an input tensor of shape input_shape, representing the initial features or noise vector.
    Convolutional Blocks:
        The generator contains six convolutional blocks (conv_block), progressively increasing the number of filters from 16 to 512. Each block consists of:
            Convolution Layer: Applies convolution with specified filters and kernel size.
            Activation Function: Uses Leaky ReLU (α = 0.2) for non-linearity.
            Batch Normalization: Optional layer to stabilize training.
            Dropout: Optional layer to reduce overfitting.
    Upsampling Blocks:
        The generator includes five upsampling blocks (deconv_block) to upscale the feature maps, with the final upsampling layer (Conv2DTranspose) generating the output with a shape suitable for the application (e.g., precipitation data).
    Output Layer: The final layer uses a transposed convolution with 1 filter and a ReLU activation function to produce the output tensor.

ii. Discriminator:

The discriminator assesses the authenticity of generated data, distinguishing between real and synthetic samples.

    Input Layer: Takes an input tensor of shape input_shape, representing the data to be classified (real or generated).
    Convolutional Blocks:
        The discriminator consists of three convolutional blocks, increasing the number of filters from 128 to 512, each with:
            Convolution Layer: Applies convolution with specified filters and kernel size.
            Activation Function: Leaky ReLU (α = 0.2) for non-linearity.
            Batch Normalization: Optional layer to stabilize training.
            Dropout: Optional layer to reduce overfitting.
    Max Pooling and Dropout: After the convolutional layers, a max pooling layer reduces spatial dimensions, followed by a dropout layer to further prevent overfitting.
    Output Layer: A convolution layer with 1 filter and a linear activation function outputs the final score indicating whether the input is real or generated.
