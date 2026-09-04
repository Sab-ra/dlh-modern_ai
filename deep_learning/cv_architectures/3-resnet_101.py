#!/usr/bin/env python3
"""
Build a ResNet-101 model
"""
from tensorflow import keras as K

def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """
    Build a bottleneck residual block locally.
    """
    prefix = '' if name is None else f'{name}_'
    shortcut = x

    x = K.layers.Conv2D(
        filters,
        (1, 1),
        strides=stride,
        padding='same',
        use_bias=False,
        name=f'{prefix}conv1'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn1'
    )(x)
    x = K.layers.ReLU(
        name=f'{prefix}relu1'
    )(x)

    x = K.layers.Conv2D(
        filters,
        (3, 3),
        padding='same',
        use_bias=False,
        name=f'{prefix}conv2'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn2'
    )(x)
    x = K.layers.ReLU(
        name=f'{prefix}relu2'
    )(x)

    x = K.layers.Conv2D(
        filters * 4,
        (1, 1),
        padding='same',
        use_bias=False,
        name=f'{prefix}conv3'
    )(x)
    x = K.layers.BatchNormalization(
        name=f'{prefix}bn3'
    )(x)

    if downsample:
        shortcut = K.layers.Conv2D(
            filters * 4,
            (1, 1),
            strides=stride,
            padding='same',
            use_bias=False,
            name=f'{prefix}shortcut_conv'
        )(shortcut)
        shortcut = K.layers.BatchNormalization(
            name=f'{prefix}shortcut_bn'
        )(shortcut)

    x = K.layers.Add(
        name=f'{prefix}add'
    )([x, shortcut])
    x = K.layers.ReLU(
        name=f'{prefix}out'
    )(x)

    return x


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

    for block_number in range(1, blocks):
        x = bottleneck_block(
            x,
            filters,
            name=f'{name}_block{block_number + 1}'
        )

    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """
    Build and return a ResNet-101 model.
    """
    inputs = K.Input(shape=input_shape)

    x = K.layers.Conv2D(
        64,
        (7, 7),
        strides=2,
        padding='same',
        use_bias=False,
        name='conv1'
    )(inputs)
    x = K.layers.BatchNormalization(name='conv1_bn')(x)
    x = K.layers.ReLU(name='conv1_relu')(x)
    x = K.layers.MaxPooling2D(
        (3, 3),
        strides=2,
        padding='same',
        name='pool1'
    )(x)

    x = make_layer(x, 3, 64, name='conv2')
    x = make_layer(x, 4, 128, stride=2, name='conv3')
    x = make_layer(x, 23, 256, stride=2, name='conv4')
    x = make_layer(x, 3, 512, stride=2, name='conv5')

    x = K.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = K.layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(x)

    return K.Model(inputs, outputs, name='resnet101')
