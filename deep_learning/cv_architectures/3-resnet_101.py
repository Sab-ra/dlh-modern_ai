#!/usr/bin/env python3
"""
Build a ResNet-101 model.
"""
from tensorflow import keras


def conv_bn_relu(
        x,
        filters,
        kernel_size,
        strides=1,
        name=None
):
    """
    Apply convolution, batch normalization, and ReLU.
    """
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=f'{name}_conv'
    )(x)
    x = keras.layers.BatchNormalization(
        name=f'{name}_bn'
    )(x)
    return keras.layers.ReLU(
        name=f'{name}_relu'
    )(x)


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """
    Build a bottleneck residual block.
    """
    shortcut = x

    x = conv_bn_relu(
        x, filters, (1, 1), stride, f'{name}_conv1'
    )
    x = conv_bn_relu(
        x, filters, (3, 3), 1, f'{name}_conv2'
    )

    x = keras.layers.Conv2D(
        filters * 4,
        (1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_conv3'
    )(x)
    x = keras.layers.BatchNormalization(
        name=f'{name}_bn3'
    )(x)

    if downsample:
        shortcut = keras.layers.Conv2D(
            filters * 4,
            (1, 1),
            strides=stride,
            padding='same',
            use_bias=False,
            name=f'{name}_shortcut_conv'
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            name=f'{name}_shortcut_bn'
        )(shortcut)

    x = keras.layers.Add(
        name=f'{name}_add'
    )([x, shortcut])

    return keras.layers.ReLU(
        name=f'{name}_out'
    )(x)


def make_layer(x, blocks, filters, stride=1, name=None):
    """
    Build one ResNet stage.
    """
    x = bottleneck_block(
        x,
        filters,
        stride=stride,
        downsample=True,
        name=f'{name}_block1'
    )

    for block_number in range(2, blocks + 1):
        x = bottleneck_block(
            x,
            filters,
            name=f'{name}_block{block_number}'
        )

    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """
    Build and return a ResNet-101 model.
    """
    inputs = keras.Input(shape=input_shape)

    x = conv_bn_relu(
        inputs,
        64,
        (7, 7),
        strides=2,
        name='conv1'
    )
    x = keras.layers.MaxPooling2D(
        (3, 3),
        strides=2,
        padding='same',
        name='pool1'
    )(x)

    x = make_layer(x, 3, 64, name='conv2')
    x = make_layer(x, 4, 128, stride=2, name='conv3')
    x = make_layer(x, 23, 256, stride=2, name='conv4')
    x = make_layer(x, 3, 512, stride=2, name='conv5')

    x = keras.layers.GlobalAveragePooling2D(
        name='avg_pool'
    )(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(x)

    return keras.Model(
        inputs,
        outputs,
        name='resnet101'
    )