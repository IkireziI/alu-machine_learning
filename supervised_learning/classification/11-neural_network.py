#!/usr/bin/env python3
""" Neural Network
"""

import numpy as np


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer performing
        binary classification.
    """

    def __init__(self, nx, nodes):
        """ Instantiation function

        Args:
            nx (int): size of the input layer
            nodes (int): number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter functions
    @property
    def W1(self):
        """Return weights vector for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Return bias for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Return activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Return weights vector for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Return bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Return activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)

        Returns:
            tuple: The activated outputs for the hidden layer and output layer
        """
        z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z))  # Sigmoid activation for the hidden layer
        z = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z))  # Sigmoid activation for the output layer
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): True labels
            A (numpy.ndarray): Predictions

        Returns:
            float: The cost function
        """
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))  # Logistic loss
        cost = np.mean(loss)  # Mean loss for cost
        return cost
