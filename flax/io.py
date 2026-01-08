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

"""IO Abstraction Layer.
The sole purpose of this abstraction layer is to avoid requiring tensorflow
as an open-source dependency solely for its tensorflow.io.gfile functions.
"""
import contextlib
import glob as glob_module
import importlib
import os
import shutil
from typing import Callable, ParamSpec, TypeVar
from enum import Enum

from absl import logging

from . import errors


_T = TypeVar('_T')
_P = ParamSpec('_P')


def _check_gs_path(func: Callable[_P, _T]) -> Callable[_P, _T]:
  """Decorator to check for gcs paths and warn if tf is not available."""

  _gs_warning = (
    'Tensorflow library not found, tensorflow.io.gfile '
    'operations will use native shim calls. '
    "GCS paths (i.e. 'gs://...') cannot be accessed."
  )

  def wrapper(*args: _P.args, **kwargs: _P.kwargs):
    assert io_mode is not None
    if io_mode == BackendMode.DEFAULT:
      values = list(args) + list(kwargs.values())
      str_values = [a for a in values if isinstance(a, str)]

      # we still check against "mode" param of GFile function, but it's fine.
      if any(v.startswith('gs://') for v in str_values):
        logging.warning(_gs_warning)

    return func(*args, **kwargs)

  return wrapper


# Global Modes and selective import of tensorflow.io gfile.


class BackendMode(Enum):
  DEFAULT = 0
  TF = 1


io_mode = None
gfile = None

if importlib.util.find_spec('tensorflow'):
  from tensorflow.io import gfile  # type: ignore

  io_mode = BackendMode.TF
else:
  io_mode = BackendMode.DEFAULT


# Constants and Exceptions


if io_mode == BackendMode.TF:
  from tensorflow import errors as tf_errors  # type: ignore

  NotFoundError = tf_errors.NotFoundError
else:
  NotFoundError = FileNotFoundError


# Overrides for testing.


@contextlib.contextmanager
def override_mode(override: BackendMode):
  # pylint: disable=g-doc-return-or-yield
  """Returns a context manager that changes backend IO mode.
  Args:
    override: BackendMode enum value to set IO mode inside context.
  """
  # pylint: enable=g-doc-return-or-yield
  global io_mode
  io_mode_prev = io_mode
  io_mode = override
  try:
    yield
  finally:
    io_mode = io_mode_prev


def set_mode(override: BackendMode):
  """Sets global io mode.
  Args:
    override: BackendMode enum value to set for IO mode.
  """
  global io_mode
  io_mode = override


# tensorflow.io.gfile API shim functions.


@_check_gs_path
def GFile(name, mode):  # pylint: disable=invalid-name
  if io_mode == BackendMode.DEFAULT:
    if 'b' in mode:
      return open(name, mode)  # pylint: disable=unspecified-encoding
    else:
      return open(name, mode, encoding='utf-8')
  elif io_mode == BackendMode.TF:
    return gfile.GFile(name, mode)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def listdir(path):
  if io_mode == BackendMode.DEFAULT:
    return os.listdir(path=path)
  elif io_mode == BackendMode.TF:
    return gfile.listdir(path=path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def isdir(path):
  if io_mode == BackendMode.DEFAULT:
    return os.path.isdir(path)
  elif io_mode == BackendMode.TF:
    return gfile.isdir(path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def copy(src, dst, overwrite=False):
  if io_mode == BackendMode.DEFAULT:
    if os.path.exists(dst) and not overwrite:
      raise errors.AlreadyExistsError(dst)
    shutil.copy(src, dst)
    return
  elif io_mode == BackendMode.TF:
    return gfile.copy(src, dst, overwrite=overwrite)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def rename(src, dst, overwrite=False):
  if io_mode == BackendMode.DEFAULT:
    if os.path.exists(dst) and not overwrite:
      raise errors.AlreadyExistsError(dst)
    return os.rename(src, dst)
  elif io_mode == BackendMode.TF:
    return gfile.rename(src, dst, overwrite=overwrite)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def exists(path):
  if io_mode == BackendMode.DEFAULT:
    return os.path.exists(path)
  elif io_mode == BackendMode.TF:
    return gfile.exists(path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def makedirs(path):
  if io_mode == BackendMode.DEFAULT:
    return os.makedirs(path, exist_ok=True)
  elif io_mode == BackendMode.TF:
    return gfile.makedirs(path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def glob(pattern):
  if io_mode == BackendMode.DEFAULT:
    return [
      path.rstrip('/') for path in glob_module.glob(pattern, recursive=False)
    ]
  elif io_mode == BackendMode.TF:
    return gfile.glob(pattern)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def remove(path):
  """Remove the file at path. Might fail if used on a directory path."""
  if io_mode == BackendMode.DEFAULT:
    return os.remove(path)
  elif io_mode == BackendMode.TF:
    return gfile.remove(path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


@_check_gs_path
def rmtree(path):
  """Remove a directory and recursively all contents inside. Might fail if used on a file path."""
  if io_mode == BackendMode.DEFAULT:
    return shutil.rmtree(path)
  elif io_mode == BackendMode.TF:
    return gfile.rmtree(path)
  else:
    raise ValueError('Unknown IO Backend Mode.')


def getsize(path):
  """Return the size, in bytes, of path."""
  if io_mode == BackendMode.DEFAULT:
    return os.path.getsize(path)
  elif io_mode == BackendMode.TF:
    return gfile.stat(path).length
  else:
    raise ValueError('Unknown IO Backend Mode.')
