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

from .axes_scan import broadcast as broadcast
from .frozen_dict import (
  FrozenDict as FrozenDict,
  freeze as freeze,
  unfreeze as unfreeze
)

from .tracers import (
  current_trace as current_trace,
  trace_level as trace_level,
  check_trace_level as check_trace_level
)

from .scope import (
  Scope as Scope,
  Array as Array,
  DenyList as DenyList,
  apply as apply,
  init as init,
  lazy_init as lazy_init,
  bind as bind)

from .lift import (
  scan as scan,
  vmap as vmap,
  jit as jit,
  remat as remat,
  remat_scan as remat_scan,
  while_loop as while_loop,
  custom_vjp as custom_vjp,
  vjp as vjp,
  jvp as jvp
)

from .meta import (
  AxisMetadata as AxisMetadata,
  unbox as unbox,
  map_axis_meta as map_axis_meta,
)
