#!/usr/bin/env python3
""" Training with RMSProp Optimization """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow
    using the RMSProp optimization algorithm.

    Args:
        loss: The loss tensor of the network.
        alpha: Learning rate.
        beta2: Decay factor for RMSProp (equivalent to rho in TF2).
        epsilon: Small value to prevent division by zero.

    Returns:
        A TensorFlow training operation.
    """

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)

    # Compute gradients and apply updates
    trainable_vars = tf.compat.v1.trainable_variables()
    gradients = tf.gradients(loss, trainable_vars)
    train_op = optimizer.apply_gradients(zip(gradients, trainable_vars))

    return train_op
