# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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

import os
import re

from absl import logging

from flax import serialization
from tensorflow.compat.v2.io import gfile


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')


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
  ckpt_tmp_path = os.path.join(ckpt_dir, 'checkpoint_tmp')
  ckpt_path = os.path.join(ckpt_dir, f'{prefix}{step}')
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


def restore_checkpoint(ckpt_dir, target, prefix='checkpoint_'):
  """Restore last/best checkpoint from checkpoints in path.

  Sorts the checkpoint files naturally, returning the highest-valued
  file, e.g.:
    ckpt_1, ckpt_2, ckpt_3 --> ckpt_3
    ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1
    ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    target: matching object to rebuild via deserialized state-dict.
    prefix: str: name prefix of checkpoint files.

  Returns:
    Restored `target` updated from checkpoint file or if no checkpoint
    files present returns the passed-in `target` unchanged.
  """
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  checkpoint_files = natural_sort(gfile.glob(glob_path))
  if not checkpoint_files:
    return target
  last_checkpoint = checkpoint_files[-1]
  logging.info('Restoring checkpoint from %s', last_checkpoint)
  with gfile.GFile(last_checkpoint, 'rb') as fp:
    return serialization.from_bytes(target, fp.read())
