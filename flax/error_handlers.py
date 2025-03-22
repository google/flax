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

"""Error Handlers."""

from dataclasses import dataclass

import typing as tp

@dataclass(frozen=True)
class ErrorHandlers:
  """Simple dataclass to hold error handlers."""

  missing_key: tp.Optional[tp.Callable[[str], Exception]] = None
  set_item: tp.Optional[tp.Callable[[str, tp.Any], Exception]] = None