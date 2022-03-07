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

"""Tools for working with dtypes."""

from typing import Any, Callable, Optional, Tuple, Type

import jax.numpy as jnp
import numpy as np


Array = Any    # pylint: disable=invalid-name
PRNGKey = Any    # pylint: disable=invalid-name
Shape = Tuple[int, ...]
FloatingDType = Type[jnp.floating]
GenericDType = Type[np.generic]
InexactDType = Type[jnp.inexact]
NumericDType = Type[jnp.number]
Initializer = Callable[[PRNGKey, Shape, InexactDType], Array]


def canonicalize_inexact_dtypes(
    input_dtype: InexactDType,
    param_dtype: Optional[InexactDType],
    computation_dtype: Optional[InexactDType]) -> Tuple[InexactDType,
                                                        InexactDType]:
  returned_param_dtype = input_dtype if param_dtype is None else param_dtype
  dtype = (jnp.result_type(input_dtype, returned_param_dtype)
           if computation_dtype is None else computation_dtype)

  assert jnp.issubdtype(input_dtype, jnp.inexact)
  return returned_param_dtype, dtype


def canonicalize_numeric_dtypes(
    input_dtype: NumericDType,
    param_dtype: Optional[NumericDType],
    computation_dtype: Optional[NumericDType]) -> Tuple[NumericDType,
                                                        NumericDType]:
  returned_param_dtype = input_dtype if param_dtype is None else param_dtype
  dtype = (jnp.result_type(input_dtype, returned_param_dtype)
           if computation_dtype is None else computation_dtype)

  assert jnp.issubdtype(input_dtype, jnp.number)
  return returned_param_dtype, dtype

