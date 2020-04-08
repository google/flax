import jax
import flax

import jax.numpy as jnp

OUTPUT_CHANNELS = 3


class DownSample(flax.nn.Module):
  def apply(self, x, features, size, apply_batchnorm=True):
    x = flax.nn.Conv(x, features=features, kernel_size=(size, size),
                     strides=(2, 2), padding='SAME', bias=False)
    if apply_batchnorm:
      x = flax.nn.BatchNorm(x)
    x = flax.nn.leaky_relu(x)
    return x


class UpSample(flax.nn.Module):
  def apply(self, x, features, size, apply_dropout=True):
    x = flax.nn.ConvTranspose(x, features=features,
                              kernel_size=(size, size), strides=(2, 2),
                              padding='SAME', bias=False)
    x = flax.nn.BatchNorm(x)
    if apply_dropout:
      x = flax.nn.dropout(x, 0.5)
    x = flax.nn.relu(x)
    return x


down_list = [[64, 4, False],
             [128, 4],
             [256, 4],
             [512, 4],
             [512, 4],
             [512, 4],
             [512, 4],
             [512, 4]]

up_list = [[512, 4, True],
           [512, 4, True],
           [512, 4, True],
           [512, 4],
           [256, 4],
           [128, 4],
           [64, 4]]


class Generator(flax.nn.Module):
  def apply(self, x):
    skips = []
    for down in down_list:
      x = DownSample(x, *down)
      skips.append(x)

    skips = list(reversed(skips[:-1]))
    for up, skip in zip(up_list, skips):
      x = UpSample(x, *up)
      x = jnp.concatenate((x, skip))

    x = flax.nn.ConvTranspose(x, features=OUTPUT_CHANNELS,
                              kernel_size=(4, 4), strides=(2, 2),
                              padding='SAME')
    x = flax.nn.tanh(x)
    return x


class Discriminator(flax.nn.Module):
  def apply(self, x):
    x = DownSample(x, 64, 4, False)
    x = DownSample(x, 128, 4)
    x = DownSample(x, 256, 4)

    x = jnp.pad(x, 1)  # padding with zeros

    x = flax.nn.Conv(x, 512, kernel_size=(4, 4), strides=(1, 1), bias=False)
    x = flax.nn.BatchNorm(x)
    x = flax.nn.leaky_relu(x)

    x = jnp.pad(x, 1)

    x = flax.nn.Conv(x, 1, kernel_size=(4, 4), strides=(1, 1))

    return x
