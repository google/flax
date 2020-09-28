"""Failing scenario for new linen API."""

from absl import app

import jax
from jax.config import config
config.enable_omnistaging()

from flax import linen as nn


class MyModule(nn.Module):
  @nn.compact
  def __call__(self):
    return self.param('foo', lambda: 0)


def main(_):
  # The parameter "foo" is not present in this empty variable dict --
  # raise an error clarifying that that's the problem (missing "foo" parameter)
  # not something cryptic about RNGs.
  MyModule().apply({})


if __name__ == '__main__':
  app.run(main)
