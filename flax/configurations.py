# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global configuration options for Flax.

Now a wrapper over jax.config, in which all config vars have a 'flax_' prefix.

To modify a config value on run time, call:
`flax.config.update('flax_<config_name>', <value>)`

"""

import os
from jax import config as jax_config

# Keep a wrapper at the flax namespace, in case we make our implementation
# in the future.
config = jax_config

# Config parsing utils

def define_bool_state(name, default, help):
  """Set up a boolean flag using JAX's config system.

  The flag will actually be stored as an environment variable of
  'FLAX_<UPPERCASE_NAME>'. JAX config ensures that the flag can be overwritten
  on runtime with `flax.config.update('flax_<config_name>', <value>)`.
  """
  return jax_config.define_bool_state('flax_' + name, default, help)


def static_bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  This is deprecated. Please use define_bool_state() unless your flag
  will be used in a static method and does not require runtime updates.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.
  Args:
    varname: the name of the variable
    default: the default boolean value
  Returns:
    boolean return value derived from defaults and environment.
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(
        'invalid truth value {!r} for environment {!r}'.format(val, varname))


# Flax Global Configuration Variables:

# Whether to use the lazy rng implementation.
flax_lazy_rng = static_bool_env('FLAX_LAZY_RNG', True)

flax_filter_frames = define_bool_state(
    name='filter_frames',
    default=True,
    help=('Whether to hide flax-internal stack frames from tracebacks.'))

flax_profile = define_bool_state(
    name='profile',
    default=True,
    help=('Whether to run Module methods under jax.named_scope for profiles.'))

flax_use_orbax_checkpointing = define_bool_state(
    name='use_orbax_checkpointing',
    default=False,
    help=('Whether to use Orbax to save checkpoints.'))

flax_relaxed_naming = define_bool_state(
    name='relaxed_naming',
    default=False,
    help=('Whether to relax naming constraints.'))
