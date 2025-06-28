#!/usr/bin/env python3
""" Variational Autoencoder"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    function that creates a variational autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                     representation
    Returns: encoder, decoder, auto
    """

    # Encoder
    X_input = keras.Input(shape=(input_dims,))
    Y_prev = X_input # Start with input for the first hidden layer

    # Encoder hidden layers
    for i in range(len(hidden_layers)):
        Y_prev = keras.layers.Dense(units=hidden_layers[i],
                                    activation='relu')(Y_prev)

    # Latent space: z_mean and z_log_sigma
    # IMPORTANT: These must be separate Dense layers to match the expected output structure.
    z_mean_dense = keras.layers.Dense(units=latent_dims, activation=None)
    z_log_sigma_dense = keras.layers.Dense(units=latent_dims, activation=None)

    z_mean = z_mean_dense(Y_prev)
    z_log_sigma = z_log_sigma_dense(Y_prev)

    # Sampling function
    def sampling(args):
        """Sampling similar points in latent space"""
        z_m, z_ls = args # z_m for mean, z_ls for log_sigma
        batch = K.shape(z_m)[0]
        dim = K.int_shape(z_m)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        # This calculates sigma and then multiplies by epsilon
        # std_dev = K.exp(0.5 * z_ls)
        # return z_m + std_dev * epsilon
        # The original code's formula was effectively: mean + exp(log_sigma / 2) * epsilon
        # which is correct for reparameterization trick
        return z_m + K.exp(z_ls / 2) * epsilon


    # Lambda layer for sampling from the latent distribution
    # The output_shape argument is crucial for Keras to correctly infer shapes
    z = keras.layers.Lambda(sampling,
                             output_shape=(latent_dims,))([z_mean,
                                                          z_log_sigma])
    # Encoder model
    # The encoder returns z, z_mean, and z_log_sigma
    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma], name='encoder')

    # Decoder
    X_decode = keras.Input(shape=(latent_dims,), name='decoder_input')
    Y_prev_decoder = X_decode # Start with decoder input for the first hidden layer

    # Decoder hidden layers (reversed order of encoder's hidden_layers)
    for j in range(len(hidden_layers) - 1, -1, -1):
        Y_prev_decoder = keras.layers.Dense(units=hidden_layers[j],
                                            activation='relu')(Y_prev_decoder)

    # Decoder output layer (reconstruction)
    last_ly = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_ly(Y_prev_decoder)
    decoder = keras.Model(X_decode, output, name='decoder')

    # Full Autoencoder (VAE)
    # The autoencoder takes X_input, passes it through the encoder,
    # then takes the 'z' (sampled latent vector) from the encoder's output
    # and passes it through the decoder to get the reconstruction.
    # The encoder outputs [z, z_mean, z_log_sigma], so we need encoder(X_input)[0] for 'z'
    e_output_z = encoder(X_input)[0] # Get 'z' from encoder output
    d_output = decoder(e_output_z)
    auto = keras.Model(X_input, d_output, name='vae')

    # VAE Loss function
    def vae_loss(x_true, x_reconstructed):
        # Reconstruction loss (Binary Cross-Entropy for binary input_dims like images)
        reconstruction_loss = K.binary_crossentropy(x_true, x_reconstructed)
        reconstruction_loss = K.sum(reconstruction_loss, axis=-1) # Sum over feature dimension

        # KL Divergence loss (Regularization term)
        # KL_loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # where z_log_sigma is log(sigma^2)
        kl_loss = -0.5 * K.sum(1 + z_log_sigma -
                               K.square(z_mean) -
                               K.exp(z_log_sigma), axis=-1) # Sum over latent dimension

        # Total VAE loss
        return K.mean(reconstruction_loss + kl_loss) # Mean over batch dimension

    # Compile the VAE model
    auto.compile(loss=vae_loss, optimizer='adam')

    return encoder, decoder, auto