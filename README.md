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
    ![PReLU](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/e5fce5b1-11f4-4af1-a739-02fc90a39912)
    
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

### **SRGAN Architecture**

![Untitled (7)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/e231c4a3-47ee-411d-b6be-f4d0a907e7eb)

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
        - 
![Untitled (8)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/ef0f02d8-5b86-4651-95d5-16972c779011)

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
    
    

# Results

## Low Resolution Image
![Untitled (9)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/8d02b3cd-b1a7-4c8c-8a1e-4575ff8899cb)

## Super Resolved Images using SR-ResNet
![Untitled (10)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/a2875c9f-25cb-4474-890c-5505b155d11b)

## Super Resolved Images using SR-ResGAN
![Untitled (11)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/e96019f9-0625-4ca8-b1fc-807145bea604)

## Original High Resolution Image
![Untitled (12)](https://github.com/K-Tanishq/Super-Resolution/assets/169484818/1b4440cd-37a4-43b8-b4a9-acec3be7b38d)


### References
- [Research Paper](https://arxiv.org/pdf/1609.04802)
