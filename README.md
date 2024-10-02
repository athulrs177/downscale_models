# Steps to create the exact conda environment

### 1) Create conda environment
 ```
   conda env create -f nn_env_athul.yaml
  ```
If you want to edit the name of the environment, then open ```nn_env_athul.yaml``` in a text editor or vim and
edit the line ```name: nn_env_athul``` to the name of your liking. Make sure to update the following instructions
to reflect this change.
 
 ```
   conda activate <your_env_name>
   ```
### 2) Install Jupyter Notebook and environment as kernel (Optional)
```
conda activate <your_env_name>
conda install jupyter
```
```
python -m ipykernel install --user --name <your_env_name> --display-name "Python (your_env_name)"
```

# Documentation for NN model scripts

## 1) Simple CNN model
### Notebook: 

transposecnn2d_downscale_high_res_europ.ipynb

#### Model Architecture
i. Input Layer:

        Shape: Accepts an input tensor of shape input_shape, representing the precipitation data to be downscaled, typically 
        including time, latitude, and longitude dimensions.

ii. Transposed Convolutional Layers:

The model employs several transposed convolutional layers to gradually refine the input features for achieving higher-resolution 
outputs.

    Layer 1:
        Operation: Transposed Convolution (Conv2DTranspose)
        Filters: 2048
        Kernel Size: (3,3)
        Strides: (1,1)
        Activation: Leaky ReLU (α = 0.2)
        Function: Expands spatial dimensions and extracts high-level features, enabling the model to learn complex patterns in 
        precipitation data.

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
This model comprises a generator and a discriminator, implementing a Wasserstein Generative Adversarial Network with Gradient Penalty 
(WGAN-GP) architecture. Below are the details for each component.

i. Generator:

The generator is designed to transform low-dimensional noise or input features into high-dimensional data, typically used for generating 
synthetic images or precipitation data.

    Input Layer: Accepts an input tensor of shape input_shape, representing the initial features or noise vector.
    Convolutional Blocks:
        The generator contains six convolutional blocks (conv_block), progressively increasing the number of filters from 16 to 512. Each 
        block consists of:
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
    Max Pooling and Dropout: After the convolutional layers, a max pooling layer reduces spatial dimensions, followed by a dropout layer to 
    further prevent overfitting.
    Output Layer: A convolution layer with 1 filter and a linear activation function outputs the final score indicating whether the input is 
    real or generated.
iii. Alternative Upsampling Block: upsampling_block

The upsampling_block is designed to perform efficient upsampling by combining an upsampling layer with a convolutional layer. This method 
can serve as an alternative to the previously used deconv_block (which employs Conv2DTranspose) for increasing spatial dimensions 
while extracting meaningful features.

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
        The block begins with an UpSampling2D layer, which increases the spatial dimensions of the input feature maps by the specified 
        upsample_factor. This operation is performed without learnable parameters, ensuring a straightforward and efficient upsampling method.

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
Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). These loss functions incorporate key metrics such as the Wasserstein 
distance, Structural Similarity Index (SSIM), and a penalty term for improved training dynamics.

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
    penalty_weight (float): The weight applied to the penalty term, controlling its influence on the total loss (default: 15, 
    do not change here, modify in the train_step function described below).
    ssim_weight (float): The weight applied to the SSIM loss term, adjusting its impact on the total loss (default: 15, 
    do not change here, modify in the train_step function described below).

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

v. Training Functions for WGAN-GP Model

The following functions are designed to facilitate the training of the Wasserstein Generative Adversarial Network with Gradient Penalty 
(WGAN-GP) model. Each function plays a critical role in managing data preprocessing, gradient computation, model updates, and overall 
training logistics.

a. Data Preprocessing: preprocess_data(data)

This function converts raw data into a TensorFlow tensor format and reshapes it for further processing.

Parameters:

    data: A pandas DataFrame/ xarray data-array containing the dataset to be converted.

Returns:

    tf.Tensor: A reshaped tensor containing the data in a format suitable for the model.

Functionality:

    The function utilizes tf.convert_to_tensor to convert the DataFrame values into a TensorFlow tensor of type float32.
    It reshapes the tensor to add an extra dimension at the end (for channels), making it suitable for image data, which typically expects 
    a shape of (height,width,channels)(height,width,channels). This transformation is necessary for convolutional layers in the model.

b. Training Step: train_step(coarse_data_batch, fine_data_batch, dis, clip_value)

This is the most important function responsible for the training. This function performs a single training iteration for both the
generator and discriminator, calculating the respective losses and updating their weights.

Parameters:

    coarse_data_batch (tf.Tensor): A batch of low-resolution input images fed to the generator.
    fine_data_batch (tf.Tensor): A batch of high-resolution images used as the ground truth for the discriminator.
    dis (tf.keras.Model): The discriminator model.
    clip_value (float): The value used to clip the weights of the discriminator for enforcing Lipschitz continuity.

Returns:

    gen_loss (tf.Tensor): The computed loss for the generator.
    dis_loss (tf.Tensor): The computed loss for the discriminator.

Functionality:

    Gradient Tapes: The function utilizes two tf.GradientTape() contexts to record operations for automatic differentiation,
    one for the generator (gen_tape) and one for the discriminator (dis_tape).
    
    Forward Pass: The generator produces a batch of fake images from the coarse data, and both the real and fake images are 
    passed to the discriminator.
    
    Loss Calculation: The generator loss is computed using the generator_loss function, while the discriminator loss is 
    computed with discriminator_loss.
    
    Gradient Calculation: Gradients for both models are calculated using the recorded tapes.
    Weight Updates: The gradients are applied to the respective model's trainable variables using optimizers 
    (gen_optimizer and dis_optimizer).
    
    Weight Clipping: The discriminator’s weights are clipped to maintain the Lipschitz constraint, enhancing training stability.

c. Distributed Training Step: distributed_train_step(coarse_data_batch, fine_data_batch, dis, clip_value)

This function enables the training of the WGAN-GP model across multiple devices or replicas, enhancing computational efficiency.

Parameters:

    coarse_data_batch (tf.Tensor): A batch of low-resolution images.
    fine_data_batch (tf.Tensor): A batch of high-resolution images.
    dis (tf.keras.Model): The discriminator model.
    clip_value (float): The value used to clip the weights of the discriminator.

Returns:

    mean_gen_loss (tf.Tensor): The average generator loss across all replicas.
    mean_dis_loss (tf.Tensor): The average discriminator loss across all replicas.

Functionality:

    The function uses strategy.run() to execute the train_step function in a distributed manner across the available replicas. 
    Each replica computes its generator and discriminator losses.
    The mean losses are aggregated using strategy.reduce(), ensuring that the results are synchronized across all devices.

d. Main Training Loop: train_gan(gen, dis, coarse_data_train, fine_data_train, clip_value, epochs, batch_size, save_intermediate=True)

This function orchestrates the entire training process of the WGAN-GP model, managing the dataset, running multiple training epochs,
and saving model checkpoints.

Parameters:

    gen (tf.keras.Model): The generator model.
    dis (tf.keras.Model): The discriminator model.
    coarse_data_train (pd.DataFrame/ xarray data-array): The training dataset of low-resolution images.
    fine_data_train (pd.DataFrame/ xarray data-array): The training dataset of high-resolution images.
    clip_value (float): The value used to clip the discriminator weights.
    epochs (int): The total number of training epochs.
    batch_size (int): The number of samples per batch.
    save_intermediate (bool): A flag indicating whether to save model checkpoints during training.

Functionality:

    Dataset Preparation: The function creates a TensorFlow dataset from the preprocessed training data, shuffling and batching it 
    appropriately for training.
    
    Training Loop: For each epoch, it initializes cumulative loss variables for the generator and discriminator, iterating over
    the batches in the distributed dataset.
    
    Loss Accumulation: The losses from each batch are accumulated, and averages are calculated for both models after processing 
    all batches.

    
    Logging: At regular intervals (every 10 epochs), it logs the average losses to the console for monitoring training progress.
    
    Model Saving: If enabled (recommended), the function saves intermediate versions of both the generator and discriminator models 
    every 50 epochs for potential future use or analysis.

e. Weight Clipping Function: clip_weights(model, clip_value)

The clip_weights function is responsible for enforcing the weight clipping technique, which is crucial for the stability of the WGAN 
model by ensuring that the weights of the discriminator model lie within a specific range [−clip_value,clip_value][−clip_value,clip_value].

Parameters:

    model (tf.keras.Model): The Keras model (typically the discriminator in WGAN) whose weights will be clipped.
    clip_value (float): The maximum value for the weights. The weights are clipped within the range 
    [−clip_value,clip_value][−clip_value,clip_value].

The function aims to apply weight clipping to all layers of the model that have trainable kernels (weights) to maintain the constraints
imposed by the WGAN framework.

Weight Clipping Process:

    a. Iterating Through Layers:
        The function loops through every layer in the model using for layer in model.layers.

    b. Checking for Trainable Weights (Kernels):
        For each layer, it checks whether the layer has an attribute called kernel using hasattr(layer, 'kernel'). The kernel 
        attribute refers to the trainable weights of the layer, which are typically present in layers like Conv2D, Dense, etc.

    c. Clipping the Weights:
        If the layer has a kernel, the function accesses the kernel (weights) via layer.kernel.
        It then clips the values of the weights using tf.clip_by_value(kernel, -clip_value, clip_value). This ensures that 
        every weight in the kernel remains within the range [−clip_value,clip_value][−clip_value,clip_value]. If a weight exceeds
        this range, it is replaced with either the minimum or maximum value.

    d. Updating the Weights:
        The clipped kernel is assigned back to the layer's weights using layer.kernel.assign(clipped_kernel), effectively updating
        the model with the new, clipped weights.

Why is Weight Clipping Important?

In the WGAN (Wasserstein GAN) framework, weight clipping plays a key role in ensuring that the discriminator satisfies the Lipschitz 
continuity condition. This condition is necessary to compute a meaningful Wasserstein distance between the real and generated data 
distributions.

Lipschitz Continuity: The weight clipping guarantees that the discriminator behaves like a 1-Lipschitz function, i.e., a function 
whose gradients are bounded. This prevents the discriminator from becoming overly confident, which can destabilize the training process.

#### References:
1. Wasserstein GAN; https://arxiv.org/abs/1701.07875.
2. Improved Training of Wasserstein GANs; https://arxiv.org/abs/1704.00028v3


## 3) Probabilistic Wasserstein GAN with gradient penalties (prob. WGAN-GP)

### Notebooks: 
a) wgan_prob_downscale_high_res_europe.ipynb (initial few runs), 

b) wgan_prob_train_part2_downscale_high_res_europe.ipynb (continue training intermediate saved models with modified params),

c) wgan_prob_model_check.ipynb (generate downscaled data with trained models (intermediate and final))

#### Model Architecture

The model architecture of probabilistic WGAN model is very similar to the deterministic version but with some key differences to enable 
the generation of an ensemble of downscaled outputs. These differences are explained here.

i. Generator and Discriminator architecture
These new versions of conv_block and deconv_block differ from the previous ones primarily in the use of randomized dropout rates.

Randomized Dropout Rate:

    In both blocks, the dropout rate is no longer fixed. Instead, a random dropout rate between 0.25 and 0.35 is generated for each 
    application of the block.
    
    Advantage:
        This introduces more stochasticity and regularization into the model during training, helping the model generalize better. 
        By preventing neurons from co-adapting too strongly, randomized dropout improves the robustness of the network and prevents 
        overfitting. The random variation in dropout ensures that no specific neuron or connection becomes overly dependent on others,
        as different parts of the network are randomly silenced during different iterations of training.
    
##### Note: 
Although not being utilised in the deterministic model, the dropout layers are designed in both versions of the WGAN-GP model to retain
the dropouts also during inference (prediction). This is especially important in the probabilistic version because it allows slight 
variations in the predictions everytime. This variation is what enables the creation of an ensemble of downscaled outputs.

ii. Ensemble generation

The generate_downscaled_data function is used to create an ensemble of downscaled precipitation data from coarse-resolution input data, 
leveraging a trained generator model. This ensemble represents multiple plausible fine-resolution outputs, capturing the uncertainty 
inherent in fine resolution precipitation outputs that can create the same coarse output. 

Detailed Explanation:

    Initialization of the Ensemble:
    The function starts by creating an empty list to store the downscaled precipitation outputs for each ensemble member. The ensemble size 
    is defined by the num_members parameter (default: 10; 50 used in the notebook; change only during the application), representing the 
    number of distinct precipitation realizations the function will generate. This is particularly important in probabilistic modeling, 
    where multiple realizations are needed to account for variability and uncertainty.

    Loop for Generating Ensemble Members:
    The function loops over the number of ensemble members (num_members). For each iteration, a new downscaled precipitation dataset is 
    generated by the model, simulating different possible high-resolution precipitation forecasts based on the same coarse-resolution input. 
    The loop is essential for building the ensemble, allowing each iteration to capture potential variability in precipitation patterns.

    (Optional) Add Noise to Coarse Data:
    Though not activated in the current implementation, the function includes an option to add Gaussian noise to the coarse precipitation 
    data. This would introduce variability in the input data, helping to generate different precipitation outputs by making the input 
    slightly different each time. The noise simulates the random nature of precipitation and can be useful in probabilistic forecasting 
    to increase the diversity of the ensemble.

    Generate Downscaled Predictions:
    For each ensemble member, the coarse precipitation data is reshaped and passed through the trained generator model. The generator 
    then produces a downscaled, high-resolution precipitation forecast based on the low-resolution input. This process uses machine 
    learning to enhance the resolution of the precipitation data, turning coarse input into finer, more detailed output.

    Organizing the Downscaled Data:
    After generating each downscaled prediction, the function converts it into an xarray DataArray. This step adds metadata such as time, 
    latitude, and longitude coordinates, ensuring that the downscaled precipitation data is well-structured and carries over the 
    attributes of the original fine-resolution data. This is useful for analysis and compatibility with climate data formats.

    Applying a NaN Mask:
    The function applies a NaN mask to each ensemble member’s prediction. This mask ensures that regions with missing data (NaN values)
    in the original fine-resolution precipitation data remain missing in the downscaled output. This is crucial for maintaining data integrity, 
    as the training is performed using E-OBS dataset which is available only over land regions.

    Storing Ensemble Members:
    Each downscaled precipitation realization (an ensemble member) is stored in a list. This list accumulates all ensemble members, providing 
    a collection of diverse precipitation forecasts that can be analyzed together.

    Combining the Ensemble Members:
    Once all ensemble members are generated, they are concatenated along a new 'number' dimension. This creates a single dataset containing 
    all ensemble members. The 'number' dimension allows users to distinguish between the different precipitation realizations, making it 
    easy to perform ensemble analysis.

    Assigning Ensemble Identifiers:
    The function assigns unique identifiers to each ensemble member, labeling them along the 'number' dimension (e.g., 0, 1, 2,...). 
    This is important for tracking and analyzing each member of the ensemble independently, enabling statistical comparisons and probabilistic 
    interpretation of the downscaled precipitation forecasts.

    Reorganizing Data Dimensions:
    The resulting dataset is reorganized to follow a standard format: time, latitude, longitude, and number (ensemble index). This format is 
    commonly used in climate and weather data, ensuring compatibility with tools and workflows for spatiotemporal analysis.

    Returning the Final Dataset:
    Finally, the function returns the full ensemble of downscaled precipitation predictions. This output is a multi-dimensional dataset, where 
    each ensemble member represents a possible realization of the downscaled precipitation. The ensemble format allows users to quantify 
    uncertainty and explore the range of possible outcomes.

##### Note: 
Unlike the deterministic models, the outputs generated using the probabilistic model will not be exactly reproducible due to the stochasticity
inherent in the model's architecture.

Aside from these, the core structure remains the same with the det. WGAN-GP in terms of training and prediction.
