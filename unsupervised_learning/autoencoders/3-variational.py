#!/usr/bin/env python3
""" Variational Autoencoder"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K  # Explicitly import K for backend operations
import tensorflow as tf # Import tensorflow to set seed


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
    # Set a fixed random seed for reproducibility of random operations
    # like tf.random_normal in the sampling layer, which can affect loss
    tf.random.set_seed(0) # Using 0 as a common default seed

    # Encoder
    X_input = keras.Input(shape=(input_dims,), name='encoder_input') # Added name for clarity
    Y_prev = X_input  # Start with input for the first hidden layer

    # Encoder hidden layers
    # All hidden layers use ReLU activation as per requirements
    for i, units in enumerate(hidden_layers):
        Y_prev = keras.layers.Dense(units=units,
                                    activation='relu',
                                    name=f'encoder_dense_{i+1}')(Y_prev)

    # Latent space: z_mean and z_log_sigma
    # These layers directly follow the last hidden layer (Y_prev).
    # They must be *separate instances* of Dense layers.
    # Activation is None (linear) as these are unbounded parameters.
    z_mean = keras.layers.Dense(units=latent_dims, activation=None,
                                 name='z_mean')(Y_prev) # Removed explicit layer name, using simple 'z_mean'
    z_log_sigma = keras.layers.Dense(units=latent_dims, activation=None,
                                      name='z_log_sigma')(Y_prev) # Removed explicit layer name, using simple 'z_log_sigma'


    # Sampling function (Reparameterization Trick)
    # This function must be defined within the scope where z_mean and z_log_sigma are available.
    def sampling(args):
        """
        Samples similar points from the latent distribution using the
        reparameterization trick.
        Args:
            args (tuple): A tuple containing (z_mean, z_log_sigma).
        Returns:
            tf.Tensor: Sampled latent vector.
        """
        z_m, z_ls = args  # z_m for mean, z_ls for log_sigma (log variance)
        batch = K.shape(z_m)[0]
        dim = K.int_shape(z_m)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        # z = mean + std_dev * epsilon, where std_dev = exp(0.5 * log_variance)
        return z_m + K.exp(z_ls / 2) * epsilon

    # Lambda layer to apply the sampling function
    # It takes z_mean and z_log_sigma as inputs and outputs the sampled z.
    # The output_shape is crucial for Keras to correctly infer shapes in some cases.
    z = keras.layers.Lambda(sampling,
                             output_shape=(latent_dims,),
                             name='z_sampling_layer')([z_mean, z_log_sigma]) # Added name for clarity

    # Encoder model
    # The encoder outputs the sampled latent vector 'z' along with
    # its mean and log-variance parameters.
    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma], name='encoder')

    # Decoder
    X_decode = keras.Input(shape=(latent_dims,), name='decoder_input')
    Y_prev_decoder = X_decode  # Start with decoder input for the first hidden layer

    # Decoder hidden layers (reversed order of encoder's hidden_layers)
    for j in range(len(hidden_layers) - 1, -1, -1):
        Y_prev_decoder = keras.layers.Dense(units=hidden_layers[j],
                                            activation='relu',
                                            name=f'decoder_dense_{len(hidden_layers)-j}')(Y_prev_decoder)

    # Decoder output layer (reconstruction)
    # Activation is sigmoid for outputting probabilities (e.g., for images 0-1)
    last_ly = keras.layers.Dense(units=input_dims, activation='sigmoid',
                                 name='decoder_output')
    output = last_ly(Y_prev_decoder)
    decoder = keras.Model(X_decode, output, name='decoder')

    # Full Autoencoder (VAE)
    # The autoencoder takes the original input, encodes it to get the
    # sampled latent vector 'z', and then decodes 'z' back to the input space.
    # encoder(X_input) returns [z, z_mean, z_log_sigma], so we use [0] for 'z'.
    e_output_z = encoder(X_input)[0]  # Get the sampled 'z' from encoder output (first output of encoder)
    d_output = decoder(e_output_z)
    auto = keras.Model(X_input, d_output, name='vae')


    # VAE Custom Loss Function
    # The loss for VAEs combines reconstruction loss and KL divergence loss.
    # x_true: original input
    # x_reconstructed: output from the decoder
    def vae_loss(x_true, x_reconstructed):
        # 1. Reconstruction Loss (Binary Cross-Entropy for pixel values 0-1)
        # Sum over the feature dimension (e.g., pixel dimensions for an image).
        reconstruction_loss = K.sum(K.binary_crossentropy(x_true, x_reconstructed), axis=-1)

        # 2. KL Divergence Loss (Regularization term)
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Here, z_log_sigma represents log(sigma^2).
        # Sum over the latent dimension (per sample).
        kl_loss = -0.5 * K.sum(1 + z_log_sigma -
                               K.square(z_mean) -
                               K.exp(z_log_sigma), axis=-1)

        # Total VAE loss: Mean of (reconstruction_loss + kl_loss) over the batch.
        return K.mean(reconstruction_loss + kl_loss)

    # Compile the VAE model
    auto.compile(loss=vae_loss, optimizer='adam')

    return encoder, decoder, auto
