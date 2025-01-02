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

import contextlib
import dataclasses
import threading
import time


@dataclasses.dataclass
class DebuggingContext(threading.local):
  time_stack: list[float] = dataclasses.field(default_factory=list)


DEBUGGING_CONTEXT = DebuggingContext()


@contextlib.contextmanager
def time_context():
  DEBUGGING_CONTEXT.time_stack.append(time.time())
  try:
    yield
  finally:
    DEBUGGING_CONTEXT.time_stack.pop()


def time_to(location: str):
  total_time = time.time() - DEBUGGING_CONTEXT.time_stack[-1]
  print(f'{location}: {total_time:.6f}s')
