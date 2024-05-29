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

import importlib.util

penzai_installed = importlib.util.find_spec('penzai') is not None
try:
  from IPython import get_ipython

  in_ipython = get_ipython() is not None
except ImportError:
  in_ipython = False


def display(*args):
  """Display the given objects using a Penzai visualizer.

  If Penzai is not installed or the code is not running in IPython, ``display``
  will print the objects instead.
  """
  if not penzai_installed or not in_ipython:
    for x in args:
      print(x)
    return

  from penzai import pz  # type: ignore[import-not-found,import-untyped]

  with pz.ts.active_autovisualizer.set_scoped(pz.ts.ArrayAutovisualizer()):
    for x in args:
      pz.ts.display(x, ignore_exceptions=True)
