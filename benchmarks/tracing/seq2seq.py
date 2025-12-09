# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Seq2Seq helper functions."""

from typing import Any

from absl import flags
from flax.examples.seq2seq.input_pipeline import CharacterTable
import jax
import ml_collections


def get_fake_batch(batch_size: int, ctable: CharacterTable) -> dict[str, Any]:
  """Returns fake data for the given batch size.

  Args:
    batch_size: The global batch size to generate.
    ctable: CharacterTable for encoding.

  Returns:
    A fake batch dictionary with query and answer one-hot encoded.
  """
  return ctable.get_batch(batch_size)


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, kwargs, and any metadata.
  """
  # Import train module - it registers its flags at module level
  # Only delete conflicting flags if the module hasn't been imported yet
  import sys

  if "flax.examples.seq2seq.train" not in sys.modules:
    # Undefine flags that conflict between nlp_seq and seq2seq
    # Both modules define: learning_rate, batch_size, num_train_steps
    for flag_name in ["learning_rate", "batch_size", "num_train_steps"]:
      if flag_name in flags.FLAGS:
        delattr(flags.FLAGS, flag_name)

  from flax.examples.seq2seq import train  # pylint: disable=g-import-not-at-top

  # Set flag values from config (flags are now registered by train module)
  flags.FLAGS.learning_rate = config.learning_rate
  flags.FLAGS.batch_size = config.batch_size
  flags.FLAGS.hidden_size = config.hidden_size
  flags.FLAGS.max_len_query_digit = config.max_len_query_digit

  rng = jax.random.key(0)
  ctable = CharacterTable("0123456789+= ", config.max_len_query_digit)
  state = train.get_train_state(rng, ctable)
  batch = get_fake_batch(config.batch_size, ctable)
  return train.train_step, (state, batch, rng, ctable.eos_id), {}
