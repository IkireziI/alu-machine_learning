#!/usr/bin/env python3
"""
This module contains the function rnn.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN over
    multiple time steps.

    Parameters:
    rnn_cell -- an instance of RNNCell that will be used
    for forward propagation
    X -- numpy.ndarray of shape (t, m, i), the data input for
    the RNN
         t is the maximum number of time steps
         m is the batch size
         i is the dimensionality of the data
    h_0 -- numpy.ndarray of shape (m, h), the initial hidden state
           h is the dimensionality of the hidden state

    Returns:
    H -- numpy.ndarray containing all hidden states
    for every time step, shape (t + 1, m, h)
    Y -- numpy.ndarray containing all outputs for every time step,
    shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize H to store hidden states for all time steps
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize an empty list to store the outputs
    Y = []

    # Loop through each time step
    for step in range(t):
        h_prev = H[step]  # Previous hidden state
        x_t = X[step]     # Input at current time step
        # Perform forward propagation through the RNNCell
        h_next, y = rnn_cell.forward(h_prev, x_t)
        # Store the next hidden state in H
        H[step + 1] = h_next
        # Append the output to Y
        Y.append(y)

    # Convert the list of outputs Y into a numpy array
    Y = np.array(Y)

    return H, Y