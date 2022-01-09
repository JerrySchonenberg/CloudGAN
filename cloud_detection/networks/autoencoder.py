import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers


class AE(tf.keras.Model):
   def __init__(self, 
                input_res = 256, 
                activation = "relu", 
                activation_out = "sigmoid", 
                seed = 42):
      super(AE, self).__init__()
      
      kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

      self.encoder = tf.keras.Sequential([
         layers.Input(shape=(input_res,input_res,3), name="input"),
         layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         ),
         layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         ),
         layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         )
      ])
      self.decoder = tf.keras.Sequential([
         layers.Conv2DTranspose(
            filters=32,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         ),
         layers.Conv2DTranspose(
            filters=64,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         ),
         layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3,3),
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding="same",
            strides=2,
         ),
         layers.Conv2D(
            filters=1,
            kernel_size=(3,3),
            padding="same",
            activation=activation_out,
            kernel_initializer=kernel_initializer,
         )
      ])

   def call(self, x):
      x_encoded = self.encoder(x)
      x_decoded = self.decoder(x_encoded)
      return x_decoded

   def encode(self, x):
      return self.encoder(x)
   
   def decode(self, x):
      return self.decoder(x)
   
   def to_hardmask(self, x):
      return np.around(x)