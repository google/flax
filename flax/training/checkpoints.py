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

from absl import logging

from flax import serialization
from tensorflow.io import gfile


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


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


def save_checkpoint(ckpt_dir,
                    target,
                    step,
                    prefix='checkpoint_',
                    keep=1):
  """Save a checkpoint of the model.

  Attempts to be pre-emption safe by writing to temporary before
  a final rename and cleanup of past files.

  Args:
    ckpt_dir: str: path to store checkpoint files in.
    target: serializable flax object, usually a flax optimizer.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.

  Returns:
    Filename of saved checkpoint.
  """
  # Write temporary checkpoint file.
  logging.info('Saving checkpoint at step: %s', step)
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
  gfile.makedirs(os.path.dirname(ckpt_path))
  with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
    fp.write(serialization.to_bytes(target))

  # Rename once serialization and writing finished.
  gfile.rename(ckpt_tmp_path, ckpt_path)
  logging.info('Saved checkpoint at %s', ckpt_path)

  # Remove old checkpoint files.
  base_path = os.path.join(ckpt_dir, f'{prefix}')
  checkpoint_files = natural_sort(gfile.glob(base_path + '*'))
  if len(checkpoint_files) > keep:
    old_ckpts = checkpoint_files[:-keep]
    for path in old_ckpts:
      logging.info('Removing checkpoint at %s', path)
      gfile.remove(path)

  return ckpt_path


def restore_checkpoint(ckpt_dir,
                       target,
                       step=None,
                       prefix='checkpoint_',
                       parallel=True):
  """Restore last/best checkpoint from checkpoints in path.

  Sorts the checkpoint files naturally, returning the highest-valued
  file, e.g.:
    ckpt_1, ckpt_2, ckpt_3 --> ckpt_3
    ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1
    ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    target: matching object to rebuild via deserialized state-dict. If None,
      the deserialized state-dict is returned as-is.
    step: int: step number to load or None to load latest.
    prefix: str: name prefix of checkpoint files.
    parallel: bool: whether to load seekable checkpoints in parallel, for speed.

  Returns:
    Restored `target` updated from checkpoint file, or if no step specified and
    no checkpoint files present, returns the passed-in `target` unchanged.
  """
  if step:
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    if not gfile.exists(ckpt_path):
      raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
  else:
    glob_path = os.path.join(ckpt_dir, f'{prefix}*')
    checkpoint_files = natural_sort(gfile.glob(glob_path))
    ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
    checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
    if not checkpoint_files:
      return target
    ckpt_path = checkpoint_files[-1]

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
