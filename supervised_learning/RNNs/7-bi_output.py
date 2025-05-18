#!/usr/bin/env python3
"""
This module contains the BiRNN class.
"""

import numpy as np


class BidirectionalCell:
    """
    This class represents a bidirectional cell of an RNN.
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
        self.bhf = np.zeros((1, h))  # Bias for forward hidden state

        # Weights and biases for the backward direction
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))  # Bias for backward hidden state

        # Weights and biases for the output
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))  # Bias for the output

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step in
        the forward direction.

        Args:
            h_prev (numpy.ndarray): Previous hidden state
            of shape (m, h).
            x_t (numpy.ndarray): Data input for the current
            time step of shape (m, i).

        Returns:
            h_next (numpy.ndarray): The next hidden state.
        """
        # Concatenate the previous hidden state and current input
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state using tanh activation
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Performs backward propagation for one time step in the
        backward direction.

        Args:
            h_next (numpy.ndarray): Next hidden state of shape (m, h).
            x_t (numpy.ndarray): Data input for the current time step
            of shape (m, i).

        Returns:
            h_prev (numpy.ndarray): The previous hidden state.
        """
        # Concatenate the next hidden state and current input
        concat_input = np.concatenate((h_next, x_t), axis=1)

        # Compute the previous hidden state using tanh activation
        h_prev = np.tanh(np.dot(concat_input, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """
        Applies softmax to a 2D numpy array.

        Args:
            x (numpy.ndarray): Array to apply softmax to.

        Returns:
            numpy.ndarray: Softmax applied array.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Args:
            H (numpy.ndarray): Concatenated hidden states
            from both directions.
                              Shape (t, m, 2 * h), where:
                                t: number of time steps
                                m: batch size
                                h: hidden state dimensionality

        Returns:
            Y (numpy.ndarray): The outputs of the RNN.
                               Shape (t, m, o), where o is the
                               output dimensionality.
        """

        # Calculate the outputs Y
        Y = np.dot(H, self.Wy) + self.by

        # Apply softmax to get output probabilities
        Y = self.softmax(Y)

        return Y