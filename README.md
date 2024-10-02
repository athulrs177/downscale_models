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
        Function: Generates the downscaled precipitation output, expanding input features to the desired lower-resolution output, 
        effectively downscaling precipitation data by a factor of 10 in both latitude and longitude.


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
        The generator includes five upsampling blocks (deconv_block) to upscale the feature maps, with the final upsampling layer (Conv2DTranspose)
        generating the output with a shape suitable for the application (e.g., precipitation data).
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
iii. Alternative Upsampling Block: upsampling_block

The upsampling_block is designed to perform efficient upsampling by combining an upsampling layer with a convolutional layer. \
This method can serve as an alternative to the previously used deconv_block (which employs Conv2DTranspose) for increasing \
spatial dimensions while extracting meaningful features.
Function Signature:

Parameters:

    inputs: Tensor input from the previous layer, representing the feature maps to be upsampled.
    filters: Integer, the number of filters for the convolution layer following the upsampling.
    kernel_size: Tuple specifying the size of the convolution kernel.
    upsample_factor: Integer, the scaling factor for upsampling the input feature maps (e.g., 2 for doubling the dimensions).
    name: String, the name prefix for the layers in this block for easier identification.
    dilation_rate: Tuple specifying the dilation rate for the convolution operation.
    strides: Tuple specifying the stride length for the convolution operation.
    use_batch_norm: Boolean, indicating whether to include a batch normalization layer (default: True).
    use_dropout: Boolean, indicating whether to include a dropout layer (default: True).

Operation:

    Upsampling:
        The block begins with an UpSampling2D layer, which increases the spatial dimensions of the input feature maps by the specified upsample_factor. 
        This operation is performed without learnable parameters, ensuring a straightforward and efficient upsampling method.

    Convolution:
        Following the upsampling, a Conv2D layer applies convolution using the specified number of filters and kernel size. 
        This layer focuses on learning spatial hierarchies and patterns from the upsampled data.

    Activation:
        The output of the convolution layer is passed through a LeakyReLU activation function (with α = 0.2), introducing non-linearity to the model, 
        allowing it to learn complex mappings.

    Batch Normalization:
        If enabled, a BatchNormalization layer is applied, helping to stabilize and accelerate the training process by normalizing the activations.

    Dropout:
        If enabled, a Dropout layer is included to mitigate overfitting by randomly setting a fraction of input units to zero during training.

Key Differences from deconv_block:

    Upsampling Method: Unlike the deconv_block, which relies on Conv2DTranspose to perform both upsampling and convolution in a single layer, 
    the upsampling_block separates these operations. This distinction can lead to improved flexibility and control over the upsampling process.
    
    Parameter Efficiency: The UpSampling2D layer does not have learnable parameters, making it more efficient compared to Conv2DTranspose, 
    which can be parameter-heavy. This can lead to faster training and potentially better generalization.
    
    Feature Learning: The use of a dedicated convolution layer after upsampling allows for more targeted learning from the expanded feature maps, 
    which may enhance the model’s ability to capture complex features.

Advantages:

    Improved Generalization: By reducing the number of parameters and providing more control over the feature extraction process, 
    this architecture can help mitigate overfitting and enhance model performance. Also seen to overcome checker-board artefacts
    generated using transposed convolutions.
    Flexibility in Design: The separation of upsampling and convolution offers more design flexibility for experimenting with 
    different configurations and layer parameters.

Disadvantages:
    
    Longer training times: Since the learned-upsampling (using transposed convolutions) is split into simple upsampling first 
    followed by convolutions which learn the necessary weights, this alternative method is seen to be significanlty slower.  

iv. Loss Functions in WGAN-GP Architecture

The following loss functions are designed to optimize the performance of the generator and discriminator in a 
Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). These loss functions incorporate key metrics such as the Wasserstein distance, Structural Similarity Index (SSIM), and a penalty term for improved training dynamics.

a. SSIM Loss: ssim_loss

This function computes the Structural Similarity Index (SSIM) loss between generated (fake) images and real images.

Parameters:

    fake_images (tf.Tensor): The generated images produced by the generator.
    real_images (tf.Tensor): The real images used for comparison.

Returns:

    tf.Tensor: The SSIM loss, defined as 1−mean(SSIM)1−mean(SSIM), where a higher SSIM value indicates greater similarity between
    the images. Thus, a lower SSIM loss reflects better performance in preserving image quality and structural integrity.

Functionality:

    SSIM is a perceptual metric that evaluates the similarity between two images based on luminance, contrast, and structure. 
    The SSIM loss helps the generator produce images that are not only realistic in pixel values but also visually coherent when compared to real images.

b. Generator Loss: generator_loss

This function calculates the overall loss for the generator, combining the Wasserstein loss, a penalty term, and SSIM loss to ensure 
high-quality image generation.

Parameters:

    fake_output (tf.Tensor): The output from the discriminator for the generated images.
    fake_images (tf.Tensor): The generated images.
    real_images (tf.Tensor): The real images used for comparison.
    penalty_weight (float): The weight applied to the penalty term, controlling its influence on the total loss (default: 15).
    ssim_weight (float): The weight applied to the SSIM loss term, adjusting its impact on the total loss (default: 15).

Returns:

    tf.Tensor: The computed generator loss.

Functionality:

    Wasserstein Loss: The generator aims to minimize the average output of the discriminator for the fake images. The Wasserstein 
    loss is computed as the negative mean of fake_output, which encourages the generator to produce outputs that the discriminator 
    scores positively.
    
    Penalty Term: This term penalizes deviations between generated and real images, encouraging the generator to produce images 
    that closely resemble real data. It is calculated as the mean absolute difference between fake_images and real_images, scaled 
    by the penalty_weight.
    
    SSIM Loss: This additional term assesses the structural similarity between the generated and real images, further guiding the 
    generator to improve the perceptual quality of its outputs.

c. Discriminator Loss: discriminator_loss

This function calculates the loss for the discriminator, measuring its ability to distinguish between real and fake images.

Parameters:

    real_output (tf.Tensor): The output from the discriminator for the real images.
    fake_output (tf.Tensor): The output from the discriminator for the generated images.

Returns:

    tf.Tensor: The computed discriminator loss.

Functionality:

    The discriminator loss is computed as the difference between the average output for fake images and the average output for real 
    images. The goal is to maximize this difference, which indicates that the discriminator is effectively distinguishing real from 
    generated data. This loss is essential for the stability and convergence of the GAN training process.
