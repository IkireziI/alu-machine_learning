#!/usr/bin/env python3
"""
This module contains the BiRNN class.
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a cell of a bidirectional RNN.
    """
    def __init__(self, i, h, o):
        """
        Constructor for the BidirectionalCell class.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        # Weights and biases for the forward direction
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        # Weights and biases for the backward direction
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))  # Bias for backward hidden state

        # Weights and biases for the output
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step in
        the forward direction.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Data input for the current time
            step of shape (m, i).

        Returns:
            h_next (numpy.ndarray): The next hidden state.
        """
        m, _ = x_t.shape  # Batch size
        h = h_prev.shape[1]  # Hidden state dimension

        # Concatenate the previous hidden state and current input
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state using tanh activation
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)

        return h_next
