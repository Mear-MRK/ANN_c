# C Artificial Neural Network Framework

This project implements a neural network framework in C, designed to facilitate the construction, training, and evaluation of neural network models for both regression and classification tasks. It provides a modular structure that enables users to define various components of a neural network, including layers, activation functions, loss functions, and optimization algorithms. The framework is built to be flexible and extensible, making it suitable for prototyping, and experimenting with neural network architectures and training techniques.

The project draws inspiration from Keras API for a user-friendly and intuitive experience.

### Important Note
- This framework requires the [lin_alg_c](https://github.com/Mear-MRK/lin_alg_c) project as a dependency for linear algebra operations.
- A C11 compliant compiler is sufficient for compilation.

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


## Detailed Project Structure

### Header Files
- **nn.h**: (main header) Includes all needed headers.
- **data_points.h**: Defines structures and functions for managing data points.
- **nn_activ.h**: Defines activation functions and their derivatives.
- **nn_config.h**: Contains configuration settings for the neural network framework.
- **nn_layer.h**: Defines structures and functions for managing neural network layers.
- **nn_loss.h**: Defines loss functions and their derivatives.
- **nn_model.h**: Defines structures and functions for managing neural network models.
- **nn_model_intern.h**: Contains internal model data structures.
- **nn_optim.h**: Defines optimization algorithms and their management.
- **nn_optim_cls_ADAM.h**: Defines the ADAM optimizer.
- **nn_optim_cls_SGD.h**: Defines the SGD optimizer.
- **rnd.h**: Provides random number generation utilities.

### Source Files
- **ann_test.c**: Contains test functions and sample data generation.
- **data_points.c**: Implements functions for managing collections of data points.
- **nn_activ.c**: Implements activation functions and their derivatives.
- **nn_layer.c**: Implements functions for managing neural network layers.
- **nn_loss.c**: Implements loss functions and their derivatives.
- **nn_model.c**: Implements the overall neural network model structure.
- **nn_optim_cls_ADAM.c**: Implements the ADAM optimization algorithm.
- **nn_optim_cls_SGD.c**: Implements the SGD optimization algorithm.
- **rnd.c**: Implements random number generation utilities.

## Example Usage
```c
#include "nn.h"

//...

// Define hidden layer configuration
nn_layer lay_hid;
lay_hid.out_sz = 64;           // Set the number of neurons in the hidden layer to 64
lay_hid.dropout = 0.1;         // Apply a dropout of 10% to prevent overfitting
lay_hid.activ = nn_activ_RELU; // Use ReLU as the activation function for the hidden layer

// Define output layer configuration
nn_layer lay_out;
lay_out.out_sz = nbr_labels;   // Set the number of neurons in the output layer to the number of labels
lay_out.dropout = 0;           // No dropout for the output layer
lay_out.activ = nn_activ_ID;   // Use Identity function as the activation function for the output layer

// Initialize the neural network model
nn_model model = nn_model_NULL;  // Create a null model to start with
nn_model_construct(&model, capacity, nbr_feat); // Construct the model with the given capacity (max nbr. of layers) and number of features

// Add layers to the model
nn_model_append(&model, &lay_hid); // Append the hidden layer to the model
nn_model_append(&model, &lay_out); // Append the output layer to the model

// Initialize model weights and biases uniformly with mean `mu` and half width `w`
nn_model_init_uniform_rnd(&model, w, mu);

// Initialize the optimizer
nn_optim opt;
nn_optim_construct(&opt, &nn_optim_cls_ADAM, &model); // Construct the optimizer using the ADAM optimization algorithm

// Train the neural network model
nn_model_train(
    &model,                  // Model to be trained
    &x_data, slice_NONE,     // Training data (inputs)
    &y_data, slice_NONE,     // Training data (labels)
    NULL, slice_NONE,        // No weights for train data
    batch_sz,                // Batch size for training
    nbr_ep,                  // Number of epochs for training
    true,                    // Shuffle data during training
    &opt,                    // Optimizer
    nn_loss_CrossEnt         // Loss function (Cross-Entropy for classification)
);

// Evaluate the neural network model
avg_err = nn_model_eval(
    &model, 
    &x_test_data, slice_NONE, // Test data (inputs)
    &y_test_data, slice_NONE, // Test data (labels)
    NULL, slice_NONE,         // No weights for test data
    nn_loss_CrossEnt,         // Loss function (Cross-Entropy for classification)
    true                      // Eval for classification
);

//...
```
    
