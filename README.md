# an Artificial Neural Network Framework in C

This project implements a neural network framework in C, designed to facilitate the construction, training, and evaluation of neural network models for both regression and classification tasks. It provides a modular structure that enables users to define various components of a neural network, including layers, activation functions, loss functions, and optimization algorithms. The framework is built to be flexible and extensible, making it suitable for educational purposes, prototyping, and experimenting with neural network architectures and training techniques.

The project draws inspiration from Keras' API for a user-friendly and intuitive experience.

### Important Note
This framework requires the `lin_alg_c` project as a dependency for linear algebra operations.

### Features

1. **Data Handling**:
   - **Data Points Management**: Functions to construct, destruct, append, shuffle, and validate collections of data points.
   - **File I/O**: Support for saving and loading datasets from files.

2. **Neural Network Layers**:
   - **Layer Management**: Functions to initialize and manage neural network layers.
   - **Layer Types**: Support for different types of layers, including input, hidden, and output layers.

3. **Activation Functions**:
   - **Activation Functions**: Implementations of various activation functions such as Identity, Sigmoid, Tanh, and ReLU.
   - **Derivatives**: Functions to compute the derivatives of activation functions for backpropagation.

4. **Loss Functions**:
   - **Loss Function Types**: Implementations of loss functions such as Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE).
   - **Derivatives**: Functions to compute the derivatives of loss functions for backpropagation.

5. **Optimization Algorithms**:
   - **SGD**: Implementation of the Stochastic Gradient Descent optimizer.
   - **ADAM**: Implementation of the Adaptive Moment Estimation (ADAM) optimizer.
   - **Optimizer Management**: Functions to initialize and manage optimizers.

6. **Model Management**:
   - **Model Construction**: Functions to construct, destruct, and manage neural network models.
   - **Model Training**: Functions to train models using specified datasets, optimizers, and loss functions.
   - **Model Evaluation**: Functions to evaluate models on datasets and compute performance metrics.
   - **Model Saving/Loading**: Support for saving and loading models from files for persistence.

7. **Example Implementation**:
   - **ann_test.c**: Contains example code for generating sample data, creating neural network models, training, and evaluating them.

## Detailed Project Structure

### Header Files
- **data_points.h**: Defines structures and functions for managing data points.
- **nn.h**: Includes core neural network structures and functions.
- **nn_activ.h**: Defines activation functions and their derivatives.
- **nn_config.h**: Contains configuration settings for the neural network framework.
- **nn_layer.h**: Defines structures and functions for managing neural network layers.
- **nn_loss.h**: Defines loss functions and their derivatives.
- **nn_model.h**: Defines structures and functions for managing neural network models.
- **nn_model_intern.h**: Contains internal model data structures.
- **nn_optim.h**: Defines optimization algorithms and their management.
- **nn_optim_cls_ADAM.h**: Implements the ADAM optimization algorithm.
- **nn_optim_cls_SGD.h**: Implements the SGD optimization algorithm.
- **rnd.h**: Provides random number generation utilities.

### Source Files
- **ann_test.c**: Contains test functions and sample data generation for regression tasks.
- **data_points.c**: Implements functions for managing collections of data points.
- **nn_activ.c**: Implements activation functions and their derivatives.
- **nn_layer.c**: Implements functions for managing neural network layers.
- **nn_loss.c**: Implements loss functions and their derivatives.
- **nn_model.c**: Implements the overall neural network model structure.
- **nn_optim_cls_ADAM.c**: Implements the ADAM optimization algorithm.
- **nn_optim_cls_SGD.c**: Implements the SGD optimization algorithm.
- **rnd.c**: Implements random number generation utilities.
    
