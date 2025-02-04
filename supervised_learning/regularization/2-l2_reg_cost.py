#!/usr/bin/env python3
""" L2 Regularization Cost """

import tensorflow as tf

def l2_reg_cost(cost):
    """ 
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): Tensor containing the cost of the network without L2 regularization.

    Returns:
        tf.Tensor: Regularized cost including L2 losses.
    """
    return cost + tf.reduce_sum(tf.compat.v1.losses.get_regularization_losses())
