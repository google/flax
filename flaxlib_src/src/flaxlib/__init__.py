# Copyright 2024 The Flax Authors.
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

from .flaxlib_cpp import IndexMapping as IndexMapping
from .flaxlib_cpp import RefIndexMapping as RefIndexMapping
from .flaxlib_cpp import IndexRefMapping as IndexRefMapping
from .flaxlib_cpp import create_index_ref as create_index_ref
from .flaxlib_cpp import _graph_flatten as _graph_flatten
from .flaxlib_cpp import _graph_flatten_top as _graph_flatten_top

# -----------------------------
# Register pytrees types
# -----------------------------
import jax

jax.tree_util.register_static(IndexMapping)

del jax