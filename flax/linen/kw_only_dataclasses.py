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

"""This module is kept for backward compatibility.

Previous code targeting Python versions <3.10 is removed and wired to
built-in dataclasses module.
"""

import dataclasses
from typing import Any, TypeVar

import flax

M = TypeVar('M', bound='flax.linen.Module')
FieldName = str
Annotation = Any
Default = Any
KW_ONLY = dataclasses.KW_ONLY
field = dataclasses.field
dataclass = dataclasses.dataclass
