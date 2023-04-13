from enum import Enum
import tensorflow as tf

import Settings

class Architectures(Enum):
  CONV_80x80 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same', input_shape=(Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
    tf.keras.layers.Conv2DTranspose(filters=Settings.RENDER_PIXEL_DEPTH, kernel_size=3, strides=2, padding='same', activation='sigmoid')
  ])

  REWARD_80x80 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', input_shape=(Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  