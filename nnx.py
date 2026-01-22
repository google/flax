# Copyright 2026 The Flax Authors.
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

"""Standalone nnx module shim.

This module allows importing nnx directly as a standalone module:

    import nnx

instead of:

    from flax import nnx

Both imports refer to the exact same module, ensuring that
`nnx.Module` and `flax.nnx.Module` are the same class in memory.
"""

from flax.nnx import *
from flax.version import __version__ as __version__
from flax import nnx as _nnx

# Re-export the module's metadata
__all__: list[str] = _nnx.__all__ if hasattr(_nnx, '__all__') else []