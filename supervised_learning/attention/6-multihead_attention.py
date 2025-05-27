#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to perform multi head attention
"""

import tensorflow as tf
from sdp_attention import sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi-head attention
    """
    def __init__(self, dm, h):
        """
        Class constructor

        Parameters:
            dm (int): Dimensionality of the model
            h (int): Number of heads
        """
        if type(dm) is not int:
            raise TypeError("dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError("h must be int representing number of heads")
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension of tensor into (h, depth)
        and transposes the result to shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Performs multi-head attention

        Parameters:
            Q: Query tensor of shape (batch, seq_len_q, dk)
            K: Key tensor of shape (batch, seq_len_v, dk)
            V: Value tensor of shape (batch, seq_len_v, dv)
            mask: Optional mask (can be None)

        Returns:
            output: Attention output tensor (batch, seq_len_q, dm)
            weights: Attention weights tensor (batch, h, seq_len_q, seq_len_v)
        """
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch, -1, self.dm))
        output = self.linear(concat_attention)

        return output, weights
