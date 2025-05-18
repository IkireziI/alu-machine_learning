#!/usr/bin/env python3
"""
This module contains the RNNCell class.
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Initialize the RNNCell with random weights and zero biases.

        Parameters:
        i -- dimensionality of the data
        h -- dimensionality of the hidden state
        o -- dimensionality of the outputs
        """
        # Weights for the concatenated input and hidden state
        self.Wh = np.random.normal(size=(i + h, h))
        # Weights for the output
        self.Wy = np.random.normal(size=(h, o))
        # Biases for the hidden state and output
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
        h_prev -- numpy.ndarray of shape (m, h), previous hidden state
        x_t -- numpy.ndarray of shape (m, i), input data for the cell

        Returns:
        h_next -- the next hidden state
        y -- the output of the cell
        """
        # Concatenate h_prev and x_t along axis 1 (features axis)
        concatenated = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state with tanh activation
        h_next = np.tanh(np.dot(concatenated, self.Wh) + self.bh)

        # Compute the output of the cell
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    def softmax(self, x):
        """
        Compute the softmax of the input array.

        Parameters:
        x -- numpy.ndarray, input to softmax

        Returns:
        Softmax output
        """
        # Stability improvement
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)