# Introduction to Deep Learning with Keras

### Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
- What is Keras?
- What is TensorFlow?
- What is a model?
- What is a shallow neural network?
- What defines a deep neural network?
- What is the Sequential model in Keras?
- What is the functional API in Keras?
- When should you use a Sequential model vs. a functional model?
- What does compiling a model in Keras do?
- How do you train a model in Keras?
- How do you choose the right loss function and optimizer?
- How can you monitor performance during training?
- How do you assess the performance of a trained model?
- How do you make predictions on new data using a trained model?
- How do you save an entire Keras model?
- How do you save only the weights of a model?
- What is TensorBoard and what is it used for?

## 0-sequential.py

### Build Model using Sequential

Write a function `build_model(input_dim)` to create a shallow neural network with a single hidden layer to perform multi-class classification using the Sequential class .

- Sigmoid as an activation function for the hidden layer
- Softmax as an activation function for the output layer

Arguments:
- input_dim: Number of input features.
- neurons_h: Number of neurons for the hidden layer

Returns:
- model: keras model.

## 1-functional.py

### Build Model using Functional API alternative

Write a function `build_model(input_dim, neurons_h)` to create a shallow neural network with a single hidden layer to perform multi-class classification, without using the Sequential class.

- Sigmoid as an activation function for the hidden layer
- Softmax as an activation function for the output layer

Arguments:
- input_dim: Number of input features.
- neurons_h: Number of neurons for the hidden layer

Returns:
- model: keras model.
- HINT: use keras.Model()

## 2-compile.py

### Compile Model

Write a function `compile_model(model, learning_rate=0.01)` to configure the keras model for training having:

- Stochastic gradient descent as the optimizer
- Binary cross-entropy loss as the loss function.
- Include accuracy as a metric to monitor classification performance.

Arguments:
- model: keras model.
- learning_rate: Learning rate for gradient descent (default is 0.01).

Returns: `None`

## 3-train.py

### Train Model

Write a function `train_model(model, X, Y, epochs, verbose=1)` that trains a Keras model.

Arguments:
- model: Keras model.
- X: Input data, shape (number of examples, input features).
- Y: labels, shape (number of examples, 1).
- epochs: Number of training epochs.
- verbose: Verbosity mode (0 = silent, 1 = progress bar).

Returns: `None`

## 4-evaluate.py

### Evaluate Model

Write a function `evaluate_model(model, X, Y, verbose=0)` to assess a trained Keras model's performance on a given data.

Arguments:
- model: A trained Keras model.
- X: Input data with a shape of (number of examples, input features).
- Y: True labels corresponding to the input data with a shape of (number of examples, 1).
- verbose: Verbosity mode (0 = silent, 1 = progress bar).

Returns:
- loss: The calculated loss on the provided data.
- accuracy: The accuracy of the model on the provided data.

## 5-save_load_model.py

### Save and Load Model

Write two functions, `save_model(model, filepath)` and `load_model(filepath)`, to save and reload a Keras model, including its architecture, weights, and optimizer state.

Save Function: `save_model(model, filepath)`

Arguments:
- model: A trained Keras model to be saved.
- filepath: A string representing the file path (including the file name) where the model will be saved.

Returns: None. The function saves the model to the specified location.

Load Function: `load_model(filepath)`

Arguments:
- filepath: A string representing the file path (including the file name) from where the model will be loaded.

Returns:
- model: The reloaded Keras model.

## 6-save_load_weights.py

### Save and Load Model Weights

Write two functions, `save_model_weights(model, filepath)` and `load_model_weights(model, filepath)`, to save and reload only the weights of a trained Keras model.

Save Function: `save_model_weights(model, filepath)`

Arguments:
- model: A trained Keras model whose weights need to be saved.
- filepath: A string representing the file path (including the file name) where the weights will be saved.

Returns: None. The function saves the model weights to the specified location.

Load Function: `load_model_weights(model, filepath)`

Arguments:
- model: A compatible Keras model instance where the weights will be loaded.
- filepath: A string representing the file path (including the file name) from where the weights will be loaded.

Returns: None. The function loads the weights into the provided model.

## 7-predict.py

### Generate Predictions

Write a function `predict(model, X, verbose=0)` to make predictions on a given dataset using a trained Keras model.

Not allowed to import any module except `import tensorflow as tf`

Arguments:
- model: A trained Keras model.
- X: Input data with a shape of (number of examples, input features).
- verbose: (Optional) Verbosity level during predictions:
    - 0: Silent (default).
    - 1: Displays a progress bar.
    - 2: Displays one line per batch.

Returns:
- predictions: A list of predicted class labels for the input data.

## 8-deep_nn_model.py

### Build a Deep Neural Network

Write a function `build_deep_model(input_dim, hidden_layers)` to create a deep neural network to perform multi-class classification.

Use the Sequential class

The hidden layers must have:
- ReLu as an activation function

Arguments:
- input_dim: Number of input features.
- hidden_layers: List of integers representing the number of neurons in each hidden layer e.g., [16, 8, 4] for three hidden layers.

Returns:
- model: Keras model

## 9-tensorboard.py

### Getting started with TensorBoard

Write a function `log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1)` to log a Keras model's training metrics to TensorBoard.

You must configure a TensorBoard callback that:
- Logs training metrics (e.g., loss and accuracy) after each epoch.
- Logs weight histograms and activation histograms using histogram_freq=1 to help visualize how weights evolve over time.
- Saves logs in a subdirectory named with a unique timestamp in the format YYYYMMDD-HHMMSS (e.g. 20250616-153245) to prevent overwriting logs from previous runs.

Not allowed to import any module except `from tensorflow import keras` and `import datetime`

Arguments:
- log_dir: (str) Base directory where logs should be saved.
- model: Keras model.
- X: Input data, shape (number of examples, input features).
- Y: labels, shape (number of examples, 1).
- epochs: Number of training epochs.
- verbose: Verbosity mode (0 = silent, 1 = progress bar).

Returns: `None`
