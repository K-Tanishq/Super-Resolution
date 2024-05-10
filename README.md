# Image Super Resolution Using SR-GANs and SR-ResNet

## Problem Statement
The problem of image super-resolution (SR) involves the process of transforming a low-resolution image into a high-resolution counterpart. This task is inherently ill-posed, as the same low-resolution image can correspond to multiple high-resolution images. The challenge is to generate a high-resolution image that is as close as possible to the original, high-resolution image, based on the limited information present in the low-resolution input.

## Methodology
We have used two different approaches to tackle this problem. First of them being SR-ResNet (Super Resolution ResNet) and second being SR-GANs (Super Resoltion GANs). Since it is a task of generating the unknown pixels in the lower resolution image to generate a higher resolution image therefore we have used GANs since we know it's ability to give excellent results in generative tasks.

## **SR Resnet Architecture**

### **Input**

- LR (Low-Resolution) image: *W*×*H*×*C* (Width, Height, Channels)
    
    𝑊×𝐻×𝐶
    

### **Layers and Blocks**

1. **Initial Convolution Block (conv1):**
    - Conv2D with large_kernel_size, channels, and PReLU activation.
    
    ![PReLU.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/a8496903-64ee-4a05-b6a1-3c5ffd3ff54b/PReLU.png)
    
    - Batch normalization after convolution.
2. **Residual Blocks (residual_blocks):**
    - Consists of multiple ResidualBlock instances.
    - Each ResidualBlock:
        - Contains two Convolutional layers with kernel_size, channels, and PReLU activation.
        - Applies Batch Normalization after each Convolutional layer.
        - Utilizes skip connections and adds the residual to the output.
3. **Intermediate Convolution Block (conv2):**
    - Conv2D with small_kernel_size, channels, and Batch Normalization.
    - No activation function after this block.
4. **Sub-Pixel Convolution Blocks (subPix_blocks):**
    - Multiple SubPixelConvBlock instances for upscaling.
    - Each SubPixelConvBlock:
        - Performs Conv2D to increase the number of channels (channels * scaling_factor^2).
        - Utilizes PixelShuffle to achieve upscaling (scaling_factor).
        - image(b, c, h, w) --> pixel_shuffle(image)(b, c/r^2, h*r, w*r), where r is scaling factor
        - Applies PReLU activation after PixelShuffle.
5. **Final Convolution Block (conv3):**
    - Conv2D with large_kernel_size and Tanh activation.
    - Output layer for generating the HR (High-Resolution) image.

### **Loss Function**

- Mean Squared Error (MSE) loss is used for the purpose of loss calculation.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/89df832f-5b76-4346-b853-7a142dd18b67/Untitled.png)

### **SRGAN Architecture**

### **Generator (SR Resnet-based)**

The generator in SRGAN is similar to the SR Resnet as described above:

1. **Input:** LR image: 𝑊×𝐻×𝐶*W*×*H*×*C* (Width, Height, Channels)
2. **Layers and Blocks:**
    - Initial Convolution Block (conv1):
        - Conv2D with large_kernel_size, channels, and PReLU activation.
        - Batch normalization after convolution.
    - Residual Blocks (residual_blocks):
        - Multiple ResidualBlock instances.
        - Each ResidualBlock:
            - Two Convolutional layers with kernel_size, channels, and PReLU activation.
            - Batch Normalization after each Convolutional layer.
            - Skip connections and addition of residuals.
    - Intermediate Convolution Block (conv2):
        - Conv2D with small_kernel_size, channels, and Batch Normalization.
        - No activation function after this block.
    - Sub-Pixel Convolution Blocks (subPix_blocks):
        - Multiple SubPixelConvBlock instances for upscaling.
        - Each SubPixelConvBlock:
            - Conv2D to increase channels (channels * scaling_factor^2).
            - PixelShuffle for upscaling (scaling_factor).
            - PReLU activation after PixelShuffle.
    - Final Convolution Block (conv3):
        - Conv2D with large_kernel_size and Tanh activation.
        - Output layer for generating the HR (High-Resolution) image.

### **Discriminator**

The discriminator aims to distinguish between real HR images and generated HR images from the generator. Here's a simplified version of the discriminator architecture:

1. **Input:** HR image: 𝑊×𝐻×𝐶*W*×*H*×*C* (Width, Height, Channels)
2. **Layers and Blocks:**
    - Convolution Blocks:
        - Conv2D layers with increasing channels and kernel sizes, followed by LeakyReLU activation.
        - Batch Normalization after each Conv2D layer.
    - Dense Layers:
        - Flatten the output of Convolution Blocks.
        - Fully connected layers (Dense) with LeakyReLU activation.
    - Output Layer:
        - Final Dense layer with sigmoid activation (binary classification).

### **Training Process**

- The generator (SR Resnet-based) aims to generate realistic HR images from LR inputs.
- The discriminator learns to distinguish between real HR images and generated HR images.
- Adversarial training involves alternating between training the generator to fool the discriminator and training the discriminator to distinguish real from fake images.

### **Loss Functions**

- **Generator Loss:**
    - Adversarial Loss: Minimize the discriminator's ability to differentiate between real and fake HR images.
    - Perceptual Loss: Difference between features extracted from real and generated images using a pre-trained network (e.g., VGG).
- **Discriminator Loss:**
    - Binary Cross-Entropy Loss: Discriminator's loss in classifying real and generated images.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/bbd3d071-b48e-4e2e-92fd-bffa25569ca5/Untitled.png)
    

# Results

## Low Resolution Image

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/83f35db7-d241-4e9f-a60d-d0a334b0db5e/Untitled.png)

## Super Resolution Image using SR ResNet

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/202b66b1-1a59-4344-826d-ff00bf07534c/Untitled.png)

## Super Resolution Image using SR GANs

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/64450356-9b5d-4020-acf1-65e905d3319c/a1eaef31-d991-423f-af3d-59607cda4b62/Untitled.png)

## Original High Resolution Image
### Results of SR-ResNets

### Results of SR-GANs

### References
- [Research Paper](https://arxiv.org/pdf/1609.04802)
