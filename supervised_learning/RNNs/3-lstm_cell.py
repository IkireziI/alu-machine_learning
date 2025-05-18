#!/usr/bin/env python3
"""
This module contains the LSTMCell class.
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM (Long Short-Term Memory) cell."""

    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell.

        Parameters:
        i -- Dimensionality of the input data
        h -- Dimensionality of the hidden state
        o -- Dimensionality of the output
        """
        # Weights for the forget gate
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # Weights for the update gate
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # Weights for the cell candidate
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # Weights for the output gate
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # Weights for the output
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        """Applies the tanh activation function."""
        return np.tanh(z)

    def softmax(self, z):
        """Applies the softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step of the LSTM cell.

        Parameters:
        h_prev -- numpy.ndarray of shape (m, h), containing the previous
        hidden state
        c_prev -- numpy.ndarray of shape (m, h), containing the previous
        cell state
        x_t -- numpy.ndarray of shape (m, i), containing the data input at
        time step t
        m -- Batch size
        i -- Dimensionality of the input
        h -- Dimensionality of the hidden state

        Returns:
        h_next -- The next hidden state
        c_next -- The next cell state
        y -- The output of the cell
        """
        m, _ = x_t.shape
        h = h_prev.shape[1]

        # Concatenate previous hidden state and input
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.dot(h_x_concat, self.Wf) + self.bf)

        # Update gate
        u_t = self.sigmoid(np.dot(h_x_concat, self.Wu) + self.bu)

        # Cell candidate
        c_hat_t = self.tanh(np.dot(h_x_concat, self.Wc) + self.bc)

        # Output gate
        o_t = self.sigmoid(np.dot(h_x_concat, self.Wo) + self.bo)

        # Compute the next cell state
        c_next = f_t * c_prev + u_t * c_hat_t

        # Compute the next hidden state
        h_next = o_t * self.tanh(c_next)

        # Compute the output of the cell
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
