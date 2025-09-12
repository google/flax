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
from __future__ import annotations

import contextlib
import dataclasses
import threading
import typing as tp

from jax._src.hijax import Box

from flax.nnx.module import Module, CAPTURE_BOX_NAME
from flax import traverse_util


@dataclasses.dataclass
class CaptureContext(threading.local):
  model: Module

  def extract_boxes(self, remove: bool = True):
    if self not in CAPTURE_CONTEXTS:
      raise ValueError('`extract_boxes` only works inside the capture context.')
    return extract_boxes(self.model, remove=remove)


CAPTURE_CONTEXTS = []


@contextlib.contextmanager
def capture_intms(model: Module):
  context = CaptureContext(model)
  add_boxes(context.model)
  CAPTURE_CONTEXTS.append(context)
  boxes: dict[str, tp.Any] = {}
  try:
    yield boxes
  finally:
    boxes.update(context.extract_boxes(remove=True))
    CAPTURE_CONTEXTS.pop()
    return boxes


def add_boxes(model: Module):
  for path, m in model.iter_modules():
    setattr(m, CAPTURE_BOX_NAME, Box({}))


def extract_boxes(model: Module, remove: bool = True) -> dict[str, tp.Any]:
  boxes = {}
  for path, m in model.iter_modules():
    if hasattr(m, CAPTURE_BOX_NAME):
      assert isinstance((box := getattr(m, CAPTURE_BOX_NAME)), Box)
      captures = box.get()
      for k, v in captures.items():
        boxes[(*path, k)] = v
      if remove:
        delattr(m, CAPTURE_BOX_NAME)
  return traverse_util.unflatten_dict(boxes)


def remove_boxes(model: Module):
  for path, m in model.iter_modules():
    if hasattr(m, CAPTURE_BOX_NAME):
      delattr(m, CAPTURE_BOX_NAME)

