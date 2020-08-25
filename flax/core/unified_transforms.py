# Copyright 2020 The Flax Authors.
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

from dataclasses import dataclass

import jax
from jax import lax

from typing import Union, Optional, Callable, Any

@dataclass(frozen=True)
class Scan:
  axis: int

ScanAxis = Optional[int]


def scan(
    fn: Callable[..., Any],
    scan_in_axis: Any,
    scan_out_axis: Any):

  def body_fn(c, x):
    jax.tree_multimap()
    c, y = fn(c, x)
    return c, y



  def scan_fn(init, *args,
              length: Optional[int] = None, reverse: bool = False):

    return lax.scan(body_fn, init, args, length=length, reverse=reverse)

  return scan_fn
