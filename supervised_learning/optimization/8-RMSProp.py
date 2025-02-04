#!/usr/bin/env python3
""" Training with RMSProp optimization in TensorFlow 2.x
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Creates the training operation for a neural network using RMSProp
        Args:
            loss: Tensor, the loss of the network
            alpha: float, learning rate
            beta2: float, RMSProp decay factor (equivalent to rho in TF2)
            epsilon: float, small value to avoid division by zero
        Returns:
            The RMSProp optimization operation
    """

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
    
    # Compute gradients and apply updates
    train_op = optimizer.minimize(loss, var_list=tf.compat.v1.trainable_variables())

    return train_op
