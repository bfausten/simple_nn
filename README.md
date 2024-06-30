# simple_nn
A simple neural network implementation in Python using only numpy.

## Architecture
The neural network is a VERY simple feedforward neural network with one hidden layer. The input layer has 784 nodes, the hidden layer has 128 nodes, and the output layer has 10 nodes. The activation function used for the hidden layer is ReLU. The output layer uses the softmax function. The neural network is trained online using the delta-rule.

## Results
The neural network is trained on the MNIST dataset. The accuracy of the neural network on the test set is around 64%. The neural network is trained for 20 epochs with a learning rate of 0.09.
