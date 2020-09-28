"""Failing scenario for new linen API."""

from absl import app

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
  MyModule().apply({'bar': lambda: 1})


if __name__ == '__main__':
  app.run(main)
