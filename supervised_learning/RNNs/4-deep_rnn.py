#!/usr/bin/env python3
"""
This module contains the DeepRNN class.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances for each layer.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h).

    Returns:
        H (numpy.ndarray): Hidden states for all time steps and layers.
        Y (numpy.ndarray): Output for all time steps.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # Initialize hidden states H and outputs Y
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0  # Set the initial hidden states

    Y = []  # Store outputs for all timesteps

    # Perform forward propagation through time steps
    for step in range(t):
        x_t = X[step]  # Input at time step t
        h_prev = H[step]  # Previous hidden states for all layers

        h_current = []
        for layer in range(l):
            # Forward propagate through each layer
            rnn_cell = rnn_cells[layer]
            h_prev_layer = h_prev[layer]
            h_next, y_next = rnn_cell.forward(h_prev_layer, x_t)

            x_t = h_next
            h_current.append(h_next)

        # Store current hidden states
        H[step + 1] = np.array(h_current)
        Y.append(y_next)

    # Convert Y to a numpy array with shape (t, m, o)
    #  where o is output dimension
    Y = np.array(Y)

    return H, Y