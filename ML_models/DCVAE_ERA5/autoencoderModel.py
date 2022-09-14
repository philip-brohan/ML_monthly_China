# Specify a Deep Convolutional Variational AutoEncoder
#  for the HadUK-Grid monthly fields.

import os
import sys
import tensorflow as tf

# Make SST mask tensor
sys.path.append("%s/../../get_data/" % os.path.dirname(__file__))
from ERA5_monthly_load import lm_ERA5

sst_mask = tf.constant(lm_ERA5.data.mask)


class DCVAE(tf.keras.Model):

    # Initialiser - set up instance and define the models
    def __init__(self):
        super(DCVAE, self).__init__()

        # Hyperparameters
        # Latent space dimension
        self.latent_dim = 20
        # Ratio of RMSE to KLD in error
        self.RMSE_scale = tf.constant(100.0, dtype=tf.float32)
        # Relative importances of each variable in error
        self.PRMSL_scale = tf.constant(1.0, dtype=tf.float32)
        self.SST_scale = tf.constant(1.0, dtype=tf.float32)
        self.T2M_scale = tf.constant(1.0, dtype=tf.float32)
        self.PRATE_scale = tf.constant(1.0, dtype=tf.float32)

        # Model to encode input to latent space distribution
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(200, 320, 4)),
                tf.keras.layers.Conv2D(
                    filters=10,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2D(
                    filters=40,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        # Model to generate output from latent space
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=25 * 40 * 40, activation=tf.nn.elu),
                tf.keras.layers.Reshape(target_shape=(25, 40, 40)),
                tf.keras.layers.Conv2DTranspose(
                    filters=20,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=10,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=3, strides=2, padding="same"
                ),
            ]
        )

    # Call the encoder model with a batch of input examples and return a batch of
    #  means and a batch of variances of the encoded latent space PDFs.
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Sample a batch of points in latent space from the encoded means and variances
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # Call the generator model with a batch of points in latent space and return a
    #  batch of outputs
    def generate(self, z):
        generated = self.generator(z)
        return generated

    # Run the full VAE - convert a batch of inputs to one of outputs
    # Note, does not do training - there's another function for that.
    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)
        generated = self.generate(latent)
        return generated

    # Utility function to calculte fit of sample to N(mean,logvar)
    # Used in loss calculation
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * 3.141592653589793)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    # Calculate the losses from autoencoding a batch of inputs
    # We are calculating a seperate loss for each variable, and for for the
    #  two components of the latent space KLD regularizer. This is useful
    #  for monitoring and debugging, but the weight update only depends
    #  on a single value (their sum).
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x[0])
        latent = self.reparameterize(mean, logvar)
        generated = self.generate(latent)

        rmse_PRMSL = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 0], x[0][:, :, :, 0])
                )
            )
            * self.RMSE_scale
            * self.PRMSL_scale
        )
        mask = tf.broadcast_to(sst_mask, generated[:, :, :, 1].shape)  # Add batch dim
        rmse_SST = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.boolean_mask(generated[:, :, :, 1], mask),
                        tf.boolean_mask(x[0][:, :, :, 1], mask),
                    )
                )
            )
            * self.RMSE_scale
            * self.SST_scale
        )
        rmse_T2M = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 2], x[0][:, :, :, 2])
                )
            )
            * self.RMSE_scale
            * self.T2M_scale
        )
        rmse_PRATE = (
            tf.math.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(generated[:, :, :, 3], x[0][:, :, :, 3])
                )
            )
            * self.RMSE_scale
            * self.PRATE_scale
        )
        logpz = tf.reduce_mean(self.log_normal_pdf(latent, 0.0, 0.0) * -1)
        logqz_x = tf.reduce_mean(self.log_normal_pdf(latent, mean, logvar))
        return tf.stack([rmse_PRMSL, rmse_SST, rmse_T2M, rmse_PRATE, logpz, logqz_x])

    # Run the autoencoder for one batch, calculate the errors, calculate the
    #  gradients and update the layer weights.
    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_values = self.compute_loss(x)
            overall_loss = tf.math.reduce_sum(loss_values, axis=0)
        gradients = tape.gradient(overall_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))