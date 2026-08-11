# Enhancing Deep Learning Models

## 0-gradient_descent_variants.py

### (OPTIMIZATION) Compare Gradient Descent Variants

Write the function `train_with_gradient_descent_variant(variant, learning_rate, x_train, batch_size)` to return a configured gradient descent optimizer and the appropriate batch size based on the selected gradient descent variant.

Arguments:
- variant: (str) The training variants: 'batch', 'stochastic', or 'mini_batch':
- batch: Perform updates based on the entire dataset (batch gradient descent).
- stochastic Perform updates on a single training example at a time (stochastic gradient descent).
- mini_batch Perform updates on a custom-sized batch of training examples (mini-batch gradient descent).
- learning_rate: (float) The learning rate for the optimizer.
- x_train: The training dataset (input data).
- batch_size: (int) The batch size to use when 'mini_batch' is selected.

Returns:
- optimizer: A Gradient Descent optimizer configured with the specified learning rate.
- bs: The correct batch size based on the selected variant.

## 1-momentum_sgd_variants.py

### (OPTIMIZATION) Momentum-Based SGD Variants

Write the function `get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False)` that returns a configured SGD-based optimizer based on the specified variant, including options for momentum and Nesterov acceleration.

Arguments:
- name: (str) the optimizer variant: 'SGD', 'SGD+Momentum', or 'SGD+Momentum+Nesterov'.
- SGD: Standard stochastic gradient descent.
- SGD+Momentum: SGD with classical momentum.
- SGD+Momentum+Nesterov: SGD with momentum and Nesterov acceleration.
- lr: (float) The learning rate.
- momentum: (float) The momentum factor.
- nesterov: (boolean) Indicating whether to apply Nesterov acceleration (default is False).

Returns:
- optimizer: A Keras SGD optimizer instance configured with the provided settings.

## 2-adaptive_optimizers.py

### (OPTIMIZATION) Adaptive optimizers vs. SGD

Write the function `get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho)` that returns a Keras optimizer configured based on the specified optimizer name and its corresponding parameters.

Arguments:
- name: (str) The name of the optimizer to use : 'sgd', 'adam', or 'rmsprop'.
- sgd: Stochastic Gradient Descent, with optional momentum.
- adam: Adaptive Moment Estimation: combines the benefits of RMSprop and momentum-based optimization techniques.
- rmsprop: Root Mean Square Propagation: adapts the learning rate for each parameter based on its historical gradients.
- learning_rate: (float) The learning rate for the optimizer (e.g., 0.01).
- momentum: (float) The momentum factor (only used for SGD).
- beta_1: (float) The exponential decay rate for the first moment estimate (only used for Adam).
- beta_2: (float) The exponential decay rate for the second moment estimate (only used for Adam).
- rho: (float) The decay factor for RMSprop (only used for RMSprop).

Returns:
- optimizer: A Keras optimizer instance (SGD, Adam, or RMSprop) configured with the provided settings.

## 3-learning_rate_schedule.py

### (OPTIMIZATION) SGD with Learning Rate Schedules

Write the function `get_optimizer_SGD_with_schedule(schedule_type, initial_lr, decay_steps, decay_rate, momentum)` to return a Keras SGD optimizer with momentum and a specified learning rate schedule.

Arguments:
- schedule_type: (str) The schedule type: 'exponential' or 'inverse_time'.
- 'exponential': Applies exponential decay to the learning rate.
- 'inverse_time': Applies inverse time decay to the learning rate.
- initial_lr: (float) The initial learning rate.
- decay_steps: (int) The number of steps before applying decay.
- decay_rate: (float) the decay rate factor.
- momentum: (float) The momentum factor.

The learning rate decay should occur in a stepwise fashion.

Returns:
- optimizer: A tf.keras.optimizers.SGD optimizer configured with the selected schedule and momentum.
- lr_schedule: A tf.keras.optimizers.schedules.LearningRateSchedule object applied to the optimizer

## 4-weight_initialization.py

### (REGULARIZATION) Weight Initialization

Write the function `build_model_initializer_by_activation(input_dim, hidden_units, activation)` to return a compiled Keras model with:

- One hidden layer that uses an appropriate weight initializer based on the activation function.
- Followed by a softmax output layer.

Arguments:
- input_dim: (int) The number of input features.
- hidden_units: (int) The number of neurons in the hidden layer.
- activation: (string) the activation function to use in the hidden layer: 'sigmoid', 'tanh', 'relu', or 'leaky_relu'.
- sigmoid and tanh: Use Glorot Uniform initializer.
- relu and leaky_relu: Use He Normal initializer.

Returns:
- model: A Keras model with the described architecture.

## 5-l2_reg.py

### (REGULARIZATION) L2 Regularization

Write the function `build_model_with_L2_regularization(input_dim, hidden_units, n_layers, lambda_l2)` to create a Keras model with L2 regularization:

Multiple hidden layers, each consisting of:

- A dense layer.
- ReLU activation.
- L2 regularization applied to the kernel weights.

Followed by a softmax output layer.

Arguments:
- input_dim: (int) The number of input features.
- hidden_units: (int) The number of neurons in each hidden layer.
- n_layers: (int) specifying the number of hidden layers to include.
- lambda_l2: (float) The strength of L2 regularization.

Returns:
- model: A Keras model with the described architecture and L2 regularization.

## 6-dropout.py

### (REGULARIZATION) Dropout

Write the function `build_model_with_dropout(input_dim, hidden_units, n_layers, dropout_rate_input, dropout_rate_hidden)` to create a Keras model with dropout regularization:

The architecture should include:
- An input layer followed by a dropout layer.
- Multiple hidden layers, each consisting of:
- A dense layer.
- ReLU activation.
- A dropout layer applied after each hidden layer.
- A final output layer with softmax activation for classification.

Arguments:
- input_dim: (int) Number of input features.
- hidden_units: (int) Number of neurons in each hidden layer.
- n_layers: (int) Number of hidden layers to include.
- dropout_rate_input: (float) Dropout rate to apply after the input layer.
- dropout_rate_hidden: (float) Dropout rate to apply after each hidden layer.

Returns:
- model: A Keras model instance with the described architecture.

## 7-early_stopping.py

### (REGULARIZATION) Early Stopping

Write the function `get_early_stopping_callback(patience, monitor='val_loss', verbose=1)` to create a customizable early stopping callback for Keras training.

This callback should:

- Monitor a specific metric during training (e.g., validation loss or accuracy).
- Stop training if no improvement is seen after a defined number of epochs.
- Must restore the best model weights once training stops.

Arguments
- patience: (int) Number of epochs to wait without improvement before stopping training.
- monitor: (str) Metric to monitor, such as val_loss or val_accuracy.
- verbose: (int) Verbosity mode to display messages when the callback takes an action.

Returns
- keras.callbacks.EarlyStopping: A configured Keras EarlyStopping callback.

## 8-build_model_to_be_tuned.py

### (HYPERPARAMETER TUNING) Build a Model to be Tuned

Write the function `build_model(hp)` to create a Keras model for multi-class classification, where the model architecture and training parameters are tuned via Keras tuner.

The model should include the following tunable aspects:

Input Layer:

The model will take in input vectors of shape (784,).

Hidden Layers:

The number of hidden layers and their configurations should be tunable:

- `num_layers`: (int) The number of hidden layers in the network (between 1 and 2).
- `units`: (int) The number of neurons in each hidden layer (between 4 and 12, with a step size of 4).
- `activation`: (str) The activation function for each hidden layer. Choose from `relu` or `sigmoid`.
Output Layer:

The model should have a Dense output layer with 10 units, using the softmax activation function, for multi-class classification.

Optimizer and Learning Rate:

Use the Adam optimizer.

learning_rate: (float) The learning rate for the Adam optimizer, selected from one of the fixed values: 1e-2 or 1e-3.

Arguments:
- hp: An instance of HyperParameters provided by Keras Tuner that defines the search space for the hyperparameters.

Returns:
- A compiled Keras Sequential model based on the hyperparameters defined in the hp object.

## 9-initiate_tuner.py

### (HYPERPARAMETER TUNING) Initiate the Tuner

Write the function `initiate_tuner(tuner_type, build_model, x_train, y_train, seed, hyperband_iterations, max_trials)` to initialize a Keras Tuner for hyperparameter tuning.

Not allowed to import any module except `import keras_tuner`

Arguments:
- tuner_type: (str) Type of tuner. Must be one of 'Hyperband', 'RandomSearch', or 'BayesianOptimization'.
- build_model: (function) A function that returns a compiled Keras model.
- x_train: (ndarray) Training features.
- y_train: (ndarray) Training labels.
- seed: (int) The random seed.
- hyperband_iterations: (int) Number of iterations for Hyperband tuning.
- max_trials: (int) Maximum number of trials for RandomSearch and BayesianOptimization.
- objective: (str) Metric to optimize during tuning (e.g., 'valaccuracy', 'valloss').
- overwrite: (bool) Whether to overwrite the previous tuning project. Default is True.

Returns:
- A Keras Tuner object (Hyperband, RandomSearch, or BayesianOptimization), ready for use in hyperparameter optimization.

## 10-search.py

### (HYPERPARAMETER TUNING) Search the Best Model and Hyperparameters

Write the function `search_and_return_best_model(tuner, x_train, y_train, epochs, validation_split, verbose=0)` to perform hyperparameter tuning and retrieve the best hyperparameters.

Arguments:
- tuner: A Keras Tuner object (Hyperband, RandomSearch, or BayesianOptimization) that wraps the hyperparameter search process.
- x_train: (ndarray) Training input data.
- y_train: (ndarray) Training target data.
- epochs: (int) Number of training epochs for each trial during the search.
- validation_split: (float) Fraction of training data to use as validation during tuning.
- verbose: Verbosity mode(0 = silent, 1 = search bar)

Returns:
- best_hyperparameters: The hyperparameter configuration that led to the best model, as a kerastuner.engine.hyperparameters.HyperParameters object.
