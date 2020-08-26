"""Main file for running the example.

This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.

Usage:
  python main.py 2> /dev/null
"""

import importlib
import sys
import time

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags
import tensorflow as tf

import train

# Config TPU.
try:
  import tpu_magic
except ImportError:
  print("Did not configure TPU.")
  sys.exit()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "workdir", f"/tmp/workdir_{int(time.time())}", "Work unit directory.")
config_flags.DEFINE_config_file(
    "config", "configs/default.py", "Training configuration.", lock_config=True)
flags.DEFINE_string(
    "jax_backend_target", None,
    "JAX backend target to use. Set this to grpc://<TPU_IP_ADDRESS>:8740 to use your TPU (2VM mode)."
)


def main(argv):
    del argv

    # Use stdout for Python logging.
    # The C++ logging goes to stderr but can be very verbose. Users might want
    # to redirect it to /dev/null.
    logging.get_absl_handler().python_handler.stream = sys.stdout

    # Turn on omnistaging since it fixes some bugs but is not yet the default.
    jax.config.enable_omnistaging()

    if FLAGS.jax_backend_target:
        # Configure JAX to run in 2VM mode with a remote TPU node.
        logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
        jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", FLAGS.jax_backend_target)

    # Make sure TF does not allocate memory on the GPU.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
    logging.info("JAX devices: %r", jax.devices())

    logging.info("Config: %s", FLAGS.config)
    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
