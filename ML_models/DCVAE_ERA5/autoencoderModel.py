# Specify a Deep Convolutional Variational AutoEncoder
#  for the China-region monthly fields.

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
        self.latent_dim = 10
        # Ratio of RMSE to KLD in error
        self.RMSE_scale = tf.constant(1000.0, dtype=tf.float32)
        # Relative importances of each variable in error
        self.PRMSL_scale = tf.constant(1.0, dtype=tf.float32)
        self.SST_scale = tf.constant(1.0, dtype=tf.float32)
        self.T2M_scale = tf.constant(1.0, dtype=tf.float32)
        self.PRATE_scale = tf.constant(1.0, dtype=tf.float32)
        # Max gradient to apply in optimizer
        self.max_gradient=2.0

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
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Conv2D(
                    filters=20,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Conv2D(
                    filters=40,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(
                    self.latent_dim + self.latent_dim,
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
            ]
        )

        # Model to generate output from latent space
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                #tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    units=25 * 40 * 40,
                    activation=tf.nn.elu,
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Reshape(target_shape=(25, 40, 40)),
                tf.keras.layers.Conv2DTranspose(
                    filters=20,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=10,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=4,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01),
                ),
            ]
        )

    # Call the encoder model with a batch of input examples and return a batch of
    #  means and a batch of variances of the encoded latent space PDFs.
    def encode(self, x, training=False):
        mean, logvar = tf.split(
            self.encoder(x, training=training), num_or_size_splits=2, axis=1
        )
        return mean, logvar

    # Sample a batch of points in latent space from the encoded means and variances
    def reparameterize(self, mean, logvar,training=False):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # Call the generator model with a batch of points in latent space and return a
    #  batch of outputs
    def generate(self, z,training=False):
        generated = self.generator(z, training=training)
        return generated

    # Run the full VAE - convert a batch of inputs to one of outputs
    def call(self, x,training=True):
        mean, logvar = self.encode(x,training=training)
        latent = self.reparameterize(mean, logvar,training=training)
        generated = self.generate(latent,training=training)
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
    def compute_loss(self, x, training):
        mean, logvar = self.encode(x[0],training=training)
        latent = self.reparameterize(mean, logvar,training=training)
        generated = self.generate(latent,training=training)

        # Metric is fractional variance reduction compared to climatology (0.5 everywhere)
        skill = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 0], x[0][:, :, :, 0]))
        guess = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 0]*0.0+0.5,x[0][:, :, :, 0]))
        rmse_PRMSL = (skill/guess) * self.RMSE_scale * self.PRMSL_scale

        mask = tf.broadcast_to(
            tf.logical_not(sst_mask), generated[:, :, :, 1].shape
        )  # Add batch dim
        skill = tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.boolean_mask(generated[:, :, :, 1], mask),
                        tf.boolean_mask(x[0][:, :, :, 1], mask),
                    )
                )
        guess = tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.boolean_mask(generated[:, :, :, 1]*0.0+0.5, mask), 
                        tf.boolean_mask(x[0][:, :, :, 1], mask),
                    )
                )
        rmse_SST = (skill/guess) * self.RMSE_scale * self.SST_scale

        skill = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 2], x[0][:, :, :, 2]))
        # T2M clim is 0.75 - normalisation
        guess = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 2]*0.0+0.75,x[0][:, :, :, 2]))
        rmse_T2M = (skill/guess) * self.RMSE_scale * self.T2M_scale

        skill = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 3], x[0][:, :, :, 3]))
        guess = tf.reduce_mean(tf.math.squared_difference(generated[:, :, :, 3]*0.0+0.5,x[0][:, :, :, 3]))
        rmse_PRATE = (skill/guess) * self.RMSE_scale * self.PRATE_scale

        logpz = tf.reduce_mean(self.log_normal_pdf(latent, 0.0, 0.0) * -1)
        logqz_x = tf.reduce_mean(self.log_normal_pdf(latent, mean, logvar))
        return tf.stack([rmse_PRMSL, rmse_SST, rmse_T2M, rmse_PRATE, logpz, logqz_x])

    # Run the autoencoder for one batch, calculate the errors, calculate the
    #  gradients and update the layer weights.
    @tf.function
    def train_on_batch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss_values = self.compute_loss(x,training=True)
            overall_loss = tf.math.reduce_sum(loss_values, axis=0)
        gradients = tape.gradient(overall_loss, self.trainable_variables)
        # Clip the gradients - helps against sudden numerical problems
        gradients = [tf.clip_by_norm(g, self.max_gradient) for g in gradients]
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
