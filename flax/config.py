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

.. data:: flax_filter_frames

    Whether to hide flax-internal stack frames from tracebacks.  Set by the
    FLAX_FILTER_FRAMES environment variable.  Defaults to True.

.. data:: flax_profile

    Whether to automatically wrap Module methods with jax.named_scope for
    profiles.  Set by the FLAX_PROFILE environment variable.  Defaults to True.
"""

import os

# Config parsing utils


def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

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

# Whether to hide flax-internal stack frames from tracebacks.
flax_filter_frames = bool_env('FLAX_FILTER_FRAMES', True)

# Whether to run Module methods under jax.named_scope for profiles.
flax_profile = bool_env('FLAX_PROFILE', True)

# Whether to use the lazy rng implementation
flax_lazy_rng = bool_env('FLAX_LAZY_RNG', True)

# Whether to use Orbax to save checkpoints
flax_use_orbax_checkpointing = bool_env('FLAX_USE_ORBAX_CHECKPOINTING', False)
