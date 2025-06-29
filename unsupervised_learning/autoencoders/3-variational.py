#!/usr/bin/env python3
""" Variational Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in the encoder
        latent_dims: integer, dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """

    # Encoder
    X_input = keras.Input(shape=(input_dims,))
    Y_prev = X_input
    for nodes in hidden_layers:
        Y_prev = keras.layers.Dense(units=nodes, activation='relu')(Y_prev)

    z_mean = keras.layers.Dense(units=latent_dims)(Y_prev)
    z_log_sigma = keras.layers.Dense(units=latent_dims)(Y_prev)

    def sampling(args):
        """Sampling from latent space"""
        z_m, z_stand_des = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_stand_des / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_sigma])
    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma])

    # Decoder
    X_decode = keras.Input(shape=(latent_dims,))
    Y_prev = X_decode
    for nodes in reversed(hidden_layers):
        Y_prev = keras.layers.Dense(units=nodes, activation='relu')(Y_prev)

    output = keras.layers.Dense(units=input_dims, activation='sigmoid')(Y_prev)
    decoder = keras.Model(X_decode, output)

    # Autoencoder
    z, z_mean, z_log_sigma = encoder(X_input)
    d_output = decoder(z)
    auto = keras.Model(X_input, d_output)

    def vae_loss(x, x_decoded):
        x_loss = keras.backend.binary_crossentropy(x, x_decoded)
        x_loss = keras.backend.sum(x_loss, axis=1)

        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma),
            axis=1
        )
        return x_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
