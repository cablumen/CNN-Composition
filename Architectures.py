from enum import Enum
import tensorflow as tf

import Settings

class Architectures(Enum):

  """
  CONV_4_16 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='relu'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='relu'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2DTranspose(filters=Settings.RENDER_PIXEL_DEPTH, kernel_size=3, activation='sigmoid')
  ])

  REWARD_4_16 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=(Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(528, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  """
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
  
  """
  REWARD_80x80 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=(Settings.RENDER_PIXEL_HEIGHT, Settings.RENDER_PIXEL_WIDTH, Settings.RENDER_PIXEL_DEPTH)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  """