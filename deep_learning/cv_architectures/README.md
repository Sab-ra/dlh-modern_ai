# Computer Vision Architectures

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### Core CNN Concepts

- What is a 2D convolution and how does it process images?
- What are kernels/filters, strides, padding, and dilation?
- What is a receptive field and why does it grow with depth?
- Feature maps, channels, and pooling (max, average, global)
- How do activation functions (ReLU, LeakyReLU, GELU, Swish) affect training and feature extraction?
- How does stacking layers increase feature hierarchy?
- What does each level of CNN blocks learn?

### Deep CNN Concepts

- Residual (skip) connections and identity mapping
- Residual blocks for deeper networks and vanishing gradients
- What are Bottleneck blocks and 1×1 convolutions?
- ResNet variants (ResNet-50, 101, 152) and their block structures
- Depthwise separable convolutions and pointwise convolutions
- What is a depthwise separable convolution, and why is it computationally efficient?
- What is an inverted residual block with linear bottleneck (MobileNetV2)?
- How does MobileNetV2 use expansion factors, pointwise convolutions, and residual connections?
- EfficientNet’s compound scaling strategy (depth, width, resolution)
- How do computational cost (FLOPs) and parameter count influence architecture choice?

## 0-create_cnn_model.py

### Create Convolutional architecture

Write a function `create_cnn_model(input_shape, filters, kernel_sizes, activations, pooling_type='max')` that creates a Convolutional Neural Network (CNN) model.

Arguments:
- input_shape: tuple, the shape of the input data (excluding the batch size).
- filters: list, the number of filters in each convolutional layer.
- kernel_sizes: list, the size of the kernels for each convolutional layer.
- activations: list, the activation functions for each convolutional layer.
- pooling_type: str, the type of pooling ('max' or 'avg', default is 'max').

Returns: A compiled CNN model.

## 1-train_cnn.py

### Train CNN

Write a function `compile_and_train_cnn(model, epochs, batch_size, optimizer_name='adam', optimizer_params=None)` that trains a CNN model.

Arguments:
- model: The CNN model to be trained.
- epochs: int, the number of training epochs.
- batch_size: int, the size of the batches for training.
- optimizer_name: str, the name of the optimizer to use (default is adam).
- optimizer_params: dict, additional parameters for the optimizer (default is None).

Returns the trained CNN model, raining history object.

## 2-bottleneck_block.py

### Bottleneck Block

Write a function `def bottleneck_block(x, filters, stride=1, downsample=False, name=None)` that implements a ResNet bottleneck residual block.

Arguments:
- x: input tensor.
- filters: number of filters for the 3×3 convolution.
- stride: stride for the first convolution (used for spatial downsampling).
- downsample: boolean indicating whether to apply a projection shortcut.
- name: optional string to name the block layers.

The block should consist of:
- A 1×1 convolution that reduces the number of channels.
- A 3×3 convolution applied to the reduced representation.
- A 1×1 convolution that expands the channels by a factor of 4.
- Batch Normalization after each convolution.
- ReLU activation after the first and second convolutions.
- A residual (skip) connection:
- Identity shortcut if downsample=False.
- Projection shortcut (1×1 convolution + BatchNorm) if downsample=True.
- A final ReLU activation after adding the shortcut.

Returns the output tensor of the bottleneck residual block.

## 3-resnet_101.py

### ResNet-101

Write a function `def build_resnet101(input_shape=(224, 224, 3), num_classes=1000)` that builds the ResNet‑101 architecture that builds the ResNet‑101 architecture as described in Deep Residual Learning for Image Recognition” (2015).

Arguments:
- input_shape: tuple representing the input image shape.
- num_classes: number of output classes.

The architecture should:
- Begin with an initial convolutional layer and max pooling.
- Stack bottleneck residual blocks using the standard ResNet‑101 configuration:
- 3 blocks in conv2_x
- 4 blocks in conv3_x
- 23 blocks in conv4_x
- 3 blocks in conv5_x
- Downsample spatial dimensions at the start of each stage (except the first).
- End with global average pooling and a fully connected classification layer. _You may use the following helper function to build each stage:_
```python
def make_layer(x, blocks, filters, stride=1, name=None):
    x = bottleneck_block(x, filters, stride=stride, downsample=True,
                         name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(x, filters, stride=1, downsample=False,
                             name=f'{name}_block{i+1}')
    return x
```
- Returns the Keras model implementing the ResNet‑101 architecture.

## 4-depthwise_separable_conv.py

### Depthwise Separable Convolution Block

Write a function `def depthwise_separable_conv(X, filters, stride=1)` that implements a depthwise separable convolution block which is core building block of MobileNetV1.

Arguments:
- X: input tensor.
- filters: number of output channels for the pointwise convolution.
- stride: stride applied to the depthwise convolution.

The block should consist of:
- A depthwise convolution (DepthwiseConv2D) with a 3×3 kernel.
- Batch Normalization and ReLU activation.
- A pointwise convolution (1×1 Conv2D).
- Batch Normalization and ReLU activation.

Returns the output tensor of the depthwise separable convolution block.

## 5-mobilenet_backbone.py

### MobileNet Backbone

Write a function `def mobilenet_backbone(inputs)` that builds the feature extraction backbone of MobileNetV1.

Arguments:
- inputs: input tensor to the network.

The backbone should:
- Begin with a standard 3×3 convolution with stride 2.
- Stack multiple depthwise separable convolution blocks.
- Perform spatial downsampling by increasing stride at specific stages.
- Follow the original MobileNetV1 architectural pattern.

Returns the output tensor of the MobileNet backbone (before classification).

## 6-mobilenetv1.py

### MobileNetV1

Write a function `def mobilenet(input_shape=(224, 224, 3), num_classes=1000)` that builds the MobileNetV1 architecture as described in MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017).

Arguments:
- input_shape: tuple representing the input image shape.
- num_classes: number of output classes.

The model should include:
- An input layer.
- The MobileNet backbone.
- A global average pooling layer.
- A final Dense layer with softmax activation.

Returns a Keras Model instance representing MobileNetV1.
