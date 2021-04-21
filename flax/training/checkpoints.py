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

"""Checkpointing helper functions.

Handles saving and restoring optimizer checkpoints based on step-number or
other numerical metric in filename.  Cleans up older / worse-performing
checkpoint files.
"""

from concurrent.futures import thread
import os
import re
from typing import Union

from absl import logging
from flax import core
from flax import errors
from flax import serialization
from tensorflow.io import gfile


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# Module name folowed by number.
MODULE_NUM_RE = re.compile(r'(.*)_\d+$')


def _checkpoint_path(ckpt_dir, step, prefix='checkpoint_'):
  return os.path.join(ckpt_dir, f'{prefix}{step}')


def natural_sort(file_list, signed=True):
  """Natural sort for filenames with numerical substrings.

  Args:
    file_list: List[str]: list of paths to sort containing numerical
      substrings.
    signed: bool: if leading '-' (or '+') signs should be included in
      numerical substrings as a sign or treated as a separator.
  Returns:
    List of filenames sorted 'naturally', not lexicographically: any
    integer substrings are used to subsort numerically. e.g.
    file_1, file_10, file_2  -->  file_1, file_2, file_10
    file_0.1, file_-0.2, file_2.0  -->  file_-0.2, file_0.1, file_2.0
  """
  float_re = SIGNED_FLOAT_RE if signed else UNSIGNED_FLOAT_RE
  def maybe_num(s):
    if float_re.match(s):
      return float(s)
    else:
      return s
  def split_keys(s):
    return [maybe_num(c) for c in float_re.split(s)]
  return sorted(file_list, key=split_keys)


def save_checkpoint(ckpt_dir: Union[str, os.PathLike],
                    target,
                    step,
                    prefix='checkpoint_',
                    keep=1,
                    overwrite=False):
  """Save a checkpoint of the model.

  Attempts to be pre-emption safe by writing to temporary before
  a final rename and cleanup of past files.

  Args:
    ckpt_dir: str or pathlib-like path to store checkpoint files in.
    target: serializable flax object, usually a flax optimizer.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.
    overwrite: overwrite existing checkpoint files if a checkpoint
      at the current or a later step already exits (default: False).
  Returns:
    Filename of saved checkpoint.
  """
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  # Write temporary checkpoint file.
  logging.info('Saving checkpoint at step: %s', step)
  if ckpt_dir.startswith('./'):
    ckpt_dir = ckpt_dir[2:]  # gfile.glob() can remove leading './'
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
  gfile.makedirs(os.path.dirname(ckpt_path))
  base_path = os.path.join(ckpt_dir, prefix)
  checkpoint_files = gfile.glob(base_path + '*')

  if ckpt_path in checkpoint_files:
    if not overwrite:
      raise errors.InvalidCheckpointError(ckpt_path, step)
  else:
    checkpoint_files.append(ckpt_path)

  checkpoint_files = natural_sort(checkpoint_files)
  # Handle the case if the job was preempted after the temporary checkpoint was
  # written, but before it was renamed to the final checkpoint name
  if checkpoint_files[-1] == ckpt_tmp_path:
    checkpoint_files.pop(-1)
  if ckpt_path != checkpoint_files[-1]:
    if not overwrite:
      raise errors.InvalidCheckpointError(ckpt_path, step)

  with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
    fp.write(serialization.to_bytes(target))

  # Rename once serialization and writing finished.
  gfile.rename(ckpt_tmp_path, ckpt_path, overwrite=overwrite)
  logging.info('Saved checkpoint at %s', ckpt_path)

  # Remove newer checkpoints
  if overwrite:
    ind = checkpoint_files.index(ckpt_path) + 1
    newer_ckpts = checkpoint_files[ind:]
    checkpoint_files = checkpoint_files[:ind]
    for path in newer_ckpts:
      logging.info('Removing checkpoint at %s', path)
      gfile.remove(path)

  # Remove old checkpoint files.
  if len(checkpoint_files) > keep:
    old_ckpts = checkpoint_files[:-keep]
    for path in old_ckpts:
      logging.info('Removing checkpoint at %s', path)
      gfile.remove(path)

  return ckpt_path


def latest_checkpoint(ckpt_dir, prefix='checkpoint_'):
  """Retrieve the path of the latest checkpoint in a directory.

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.

  Returns:
    The latest checkpoint path or None if no checkpoints were found.
  """
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  checkpoint_files = natural_sort(gfile.glob(glob_path))
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
  if checkpoint_files:
    return checkpoint_files[-1]
  else:
    return None


def restore_checkpoint(ckpt_dir,
                       target,
                       step=None,
                       prefix='checkpoint_',
                       parallel=True):
  """Restore last/best checkpoint from checkpoints in path.

  Sorts the checkpoint files naturally, returning the highest-valued
  file, e.g.:

  *  ``ckpt_1, ckpt_2, ckpt_3 --> ckpt_3``

  *  ``ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1``

  *  ``ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5``

  Args:
    ckpt_dir: str: checkpoint file or directory of checkpoints to restore from.
    target: matching object to rebuild via deserialized state-dict. If None,
      the deserialized state-dict is returned as-is.
    step: int: step number to load or None to load latest. If specified,
      ckpt_dir must be a directory.
    prefix: str: name prefix of checkpoint files.
    parallel: bool: whether to load seekable checkpoints in parallel, for speed.

  Returns:
    Restored `target` updated from checkpoint file, or if no step specified and
    no checkpoint files present, returns the passed-in `target` unchanged.
    If a file path is specified and is not found, the passed-in `target` will be
    returned. This is to match the behavior of the case where a directory path
    is specified but the directory has not yet been created.
  """
  if step is not None:
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    if not gfile.exists(ckpt_path):
      raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
  else:
    if gfile.isdir(ckpt_dir):
      ckpt_path = latest_checkpoint(ckpt_dir, prefix)
      if not ckpt_path:
        logging.info(f'Found no checkpoint files in {ckpt_dir}')
        return target
    else:
      ckpt_path = ckpt_dir
      if not gfile.exists(ckpt_path):
        logging.info(f'Found no checkpoint file at {ckpt_path}')
        return target

  logging.info('Restoring checkpoint from %s', ckpt_path)
  with gfile.GFile(ckpt_path, 'rb') as fp:
    if parallel and fp.seekable():
      buf_size = 128 << 20  # 128M buffer.
      num_bufs = fp.size() / buf_size
      logging.debug('num_bufs: %d', num_bufs)
      checkpoint_contents = bytearray(fp.size())

      def read_chunk(i):
        # NOTE: We have to re-open the file to read each chunk, otherwise the
        # parallelism has no effect. But we could reuse the file pointers
        # within each thread.
        with gfile.GFile(ckpt_path, 'rb') as f:
          f.seek(i * buf_size)
          buf = f.read(buf_size)
          if buf:
            checkpoint_contents[i * buf_size:i * buf_size + len(buf)] = buf
          return len(buf) / buf_size

      pool_size = 32
      pool = thread.ThreadPoolExecutor(pool_size)
      results = pool.map(read_chunk, range(int(num_bufs) + 1))
      results = list(results)
      pool.shutdown(wait=False)
      logging.debug('results: %s', results)
    else:
      checkpoint_contents = fp.read()

    if target is None:
      return serialization.msgpack_restore(checkpoint_contents)
    else:
      return serialization.from_bytes(target, checkpoint_contents)


def convert_pre_linen(params):
  """Converts a pre-Linen parameter pytree.

  In pre-Linen API submodules were numbered incrementally, independent of the
  submodule class. With Linen this behavior has changed to keep separate
  submodule counts per module class.

  Consider the following module::

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Conv(1, 1)(x)
        x = nn.Dense(1)(x)
        return x

  In pre-Linen the resulting params would have had the structure:

    ``{'Conv_0': { ... }, 'Dense_1': { ... } }``

  With Linen the resulting params would instead have had the structure:

    ``{'Conv_0': { ... }, 'Dense_0': { ... } }``

  To convert from pre-Linen format to Linen simply call::

    params = convert_pre_linen(pre_linen_params)

  Note that you can also use this utility to convert pre-Linen collections
  because they're following the same module naming. Note though that collections
  were "flat" in pre-Linen and first need to be unflattened before they can be
  used with this function::

    batch_stats = convert_pre_linen(flax.traverse_util.unflatten_dict({
        tuple(k.split('/')[1:]): v
        for k, v in pre_linen_model_state.as_dict().items()
    }))

  Then Linen variables can be defined from these converted collections::

    variables = {'params': params, 'batch_stats': batch_stats}

  Args:
    params: Parameter pytree in pre-Linen format. If the pytree is already in
      Linen format, then the returned pytree is unchanged (i.e. this function
      can safely be called on any loaded checkpoint for use with Linen).

  Returns:
    Parameter pytree with Linen submodule naming.
  """
  if not isinstance(params, (dict, core.FrozenDict)):
    return params
  params_renamed = {}
  counts = {}
  names = natural_sort(params.keys())
  for name in names:
    value = params[name]
    match = MODULE_NUM_RE.match(name)
    if match:
      module = match.group(1)
      num = counts.get(module, 0)
      name = f'{module}_{num}'
      counts[module] = num + 1
    params_renamed[name] = convert_pre_linen(value)

  return core.freeze(params_renamed)
