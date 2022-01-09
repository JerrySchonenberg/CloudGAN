import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class Unet(tf.keras.Model):
   def __init__(self, 
                input_res = 256, 
                activation = "relu", 
                activation_out = "sigmoid", 
                seed = 42):
      super(Unet, self).__init__()

      kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

      # ======== Define U-NET architecture ========
      merge = layers.Concatenate(axis=3)

      inputs = layers.Input(shape=(input_res,input_res,3))

      conv1 = layers.Conv2D(filters=16, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(inputs)
      conv1 = layers.Conv2D(filters=16, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv1)
      pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)

      conv2 = layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(pool1)
      conv2 = layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv2)
      pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

      conv3 = layers.Conv2D(filters=64, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(pool2)
      conv3 = layers.Conv2D(filters=64, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv3)
      pool3 = layers.MaxPooling2D(pool_size=(2,2))(conv3)

      conv4 = layers.Conv2D(filters=128, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(pool3)
      conv4 = layers.Conv2D(filters=128, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv4)
      drop4 = layers.Dropout(0.5)(conv4)
      pool4 = layers.MaxPooling2D(pool_size=(2,2))(drop4)

      conv5 = layers.Conv2D(filters=256, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(pool4)
      conv5 = layers.Conv2D(filters=256, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv5)
      drop5 = layers.Dropout(0.5)(conv5)
      up6 = layers.Conv2D(filters=128, kernel_size=2, activation=activation, padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2,2))(drop5))

      conv6 = layers.Conv2D(filters=128, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(merge([drop4,up6]))
      conv6 = layers.Conv2D(filters=128, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv6)
      up7 = layers.Conv2D(filters=64, kernel_size=(2,2), activation=activation, padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2,2))(conv6))

      conv7 = layers.Conv2D(filters=64, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(merge([conv3,up7]))
      conv7 = layers.Conv2D(filters=64, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv7)
      up8 = layers.Conv2D(filters=32, kernel_size=(2,2), activation=activation, padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2,2))(conv7))

      conv8 = layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(merge([conv2,up8]))
      conv8 = layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv8)
      up9 = layers.Conv2D(filters=16, kernel_size=(2,2), activation=activation, padding='same', kernel_initializer=kernel_initializer)(layers.UpSampling2D(size=(2,2))(conv8))

      conv9 = layers.Conv2D(filters=16, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(merge([conv1,up9]))
      conv9 = layers.Conv2D(filters=16, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv9)
      conv9 = layers.Conv2D(filters=2, kernel_size=(3,3), activation=activation, padding='same', kernel_initializer=kernel_initializer)(conv9)

      outputs = layers.Conv2D(filters=1, kernel_size=(1,1), activation=activation_out)(conv9)
      # ===========================================

      self.model = tf.keras.Model(inputs, outputs)

   def call(self, x):
      return self.model(x)