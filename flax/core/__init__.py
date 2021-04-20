# Copyright 2021 The Flax Authors.
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

from .axes_scan import broadcast
from .frozen_dict import FrozenDict, freeze, unfreeze
from .tracers import current_trace, trace_level, check_trace_level
from .scope import Scope, Array, DenyList, apply, init, bind
from .lift import scan, vmap, jit
