# Copyright 2022 The Flax Authors.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from absl import logging
from flax import core
from flax import errors
from flax import serialization
from flax import traverse_util
from jax import process_index
from jax.experimental.gda_serialization.serialization import get_tensorstore_spec
from jax.experimental.gda_serialization.serialization import GlobalAsyncCheckpointManager
from jax.experimental.global_device_array import GlobalDeviceArray
from tensorflow.io import gfile  # pytype: disable=import-error


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# Module name folowed by number.
MODULE_NUM_RE = re.compile(r'(.*)_\d+$')
# Alternative schemes handled by `gfile`, e.g. on Google Cloud Storage (GCS).
SCHEME_RE = re.compile('^(?P<scheme>[a-z][a-z0-9.+-]+://)?(?P<path>.*)', re.I)

# GlobalDeviceArrays is across devices and would not be stored in the same file
# with non-GDA data. So their occurrences in the target pytree will be replaced
# by this string placeholder.
GDA_PH = '//GDAPlaceholder:'

PyTree = Any


def _checkpoint_path(ckpt_dir: str,
                     step: Union[int, str],
                     prefix: str = 'checkpoint_') -> str:
  return os.path.join(ckpt_dir, f'{prefix}{step}')


def _checkpoint_path_step(path: str) -> Optional[float]:
  """Returns the step number of a checkpoint path."""
  for s in SIGNED_FLOAT_RE.split(path)[::-1]:
    if SIGNED_FLOAT_RE.match(s):
      return float(s)
  return None


def _split_gdas(
    target: Dict[str, Any]) -> Tuple[Dict[str, Any], List[GlobalDeviceArray]]:
  # When target is a single leaf instead of a pytree dict.
  if not isinstance(target, (core.FrozenDict, dict)):
    if isinstance(target, GlobalDeviceArray):
      return GDA_PH, [target]
    return target, []
  # Traverse the target and handle GlobalDeviceArrays.
  flattened = traverse_util.flatten_dict(target, keep_empty_nodes=True)
  gda_targets = []
  for key, value in flattened.items():
    if isinstance(value, GlobalDeviceArray):
      subpath = '/'.join(key)
      gda_targets.append((value, subpath))
      flattened[key] = GDA_PH + subpath
  target = traverse_util.unflatten_dict(flattened)
  return target, gda_targets


def _save_gdas(gda_manager: GlobalAsyncCheckpointManager,
               gda_targets: List[Tuple[GlobalDeviceArray, str]],
               tmp_path: str, final_path: str):
  gda_list, gda_subpaths = zip(*gda_targets)
  ts_specs = [
      get_tensorstore_spec(os.path.join(tmp_path, x)) for x in gda_subpaths
  ]
  gda_manager.serialize(
      list(gda_list),
      ts_specs,
      temp_checkpoint_dir=tmp_path,
      final_checkpoint_dir=final_path)


def _restore_gdas(state_dict,
                  target: Optional[Any],
                  ckpt_path: str,
                  step: Optional[int] = None,
                  gda_manager: Optional[GlobalAsyncCheckpointManager] = None):

  # When target is a single leaf instead of a pytree dict.
  if not isinstance(state_dict, (core.FrozenDict, dict)):
    if isinstance(target, GlobalDeviceArray) and isinstance(
        state_dict, GlobalDeviceArray):
      if not gda_manager:
        raise errors.GDACheckpointingRequiredError(ckpt_path, step)
      if not target:
        raise errors.GDARestoreTargetRequiredError(ckpt_path, step)
      gda_list = gda_manager.deserialize(
          [target.mesh], [target.mesh_axes],
          [get_tensorstore_spec(ckpt_path + '_gda')])
      return gda_list[0]
    return state_dict

  # Check if a GDA is present in the restored pytree
  flattened = traverse_util.flatten_dict(state_dict, keep_empty_nodes=True)
  gda_paths = []
  for key, value in flattened.items():
    if isinstance(value, str) and value.startswith(GDA_PH):
      subpath = value[len(GDA_PH):]
      gda_paths.append((key, os.path.join(ckpt_path+'_gda', subpath)))

  if gda_paths:
    if not gda_manager:
      raise errors.GDACheckpointingRequiredError(ckpt_path, step)
    if not target:
      raise errors.GDARestoreTargetRequiredError(ckpt_path, step)
    target_flattened = traverse_util.flatten_dict(
        serialization.to_state_dict(target), keep_empty_nodes=True)
    target_gdas = [target_flattened[x[0]] for x in gda_paths]
    if not all(isinstance(x, GlobalDeviceArray) for x in target_gdas):
      raise errors.GDARestoreTargetRequiredError(ckpt_path, step)
    meshes = [x.mesh for x in target_gdas]
    partition_specs = [x.mesh_axes for x in target_gdas]
    ts_specs = [get_tensorstore_spec(x[1]) for x in gda_paths]
    gda_list = gda_manager.deserialize(meshes, partition_specs, ts_specs)
    for gda, (key, _) in zip(gda_list, gda_paths):
      flattened[key] = gda
  state_dict = traverse_util.unflatten_dict(flattened)
  return state_dict


def natural_sort(file_list: Iterable[str], signed: bool = True) -> List[str]:
  """Natural sort for filenames with numerical substrings.

  Args:
    file_list: list of paths to sort containing numerical substrings.
    signed: bool: if leading '-' (or '+') signs should be included in numerical
      substrings as a sign or treated as a separator.

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


def safe_normpath(path: str) -> str:
  """Normalizes path safely to get around `gfile.glob()` limitations."""
  d = SCHEME_RE.match(path).groupdict()
  return (d['scheme'] or '') + os.path.normpath(d['path'])


class AsyncManager():
  """A simple object to track async checkpointing.

  How to use: create an instance and pass to save_checkpoint() calls:
    am = AsyncManager()
    save_checkpoint(..., async_manager=am)
  """

  def __init__(self, max_workers: int = 1):
    self.executor = thread.ThreadPoolExecutor(max_workers=max_workers)
    self.save_future = None

  def wait_previous_save(self):
    """Block until the previous save finishes, to keep files' consistency."""
    if self.save_future and not self.save_future.done():
      logging.warning(
          'The previous async save_checkpoint has not finished yet. Waiting '
          'for it to complete before the next save.'
      )
      self.save_future.result()

  def save_async(self, task: Callable[[], Any]):
    """Run a task async. The future will be tracked as self.save_future.

    Args:
      task: The callable to be executed asynchrously.
    """
    self.wait_previous_save()
    self.save_future = self.executor.submit(task)


def save_checkpoint(ckpt_dir: Union[str, os.PathLike],
                    target: PyTree,
                    step: int,
                    prefix: str = 'checkpoint_',
                    keep: int = 1,
                    overwrite: bool = False,
                    keep_every_n_steps: Optional[int] = None,
                    async_manager: Optional[AsyncManager] = None,
                    gda_manager: Optional[
                        GlobalAsyncCheckpointManager] = None) -> str:
  """Save a checkpoint of the model.

  Attempts to be pre-emption safe by writing to temporary before
  a final rename and cleanup of past files.

  Args:
    ckpt_dir: str or pathlib-like path to store checkpoint files in.
    target: serializable flax object, usually a flax optimizer.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.
    overwrite: overwrite existing checkpoint files if a checkpoint at the
      current or a later step already exits (default: False).
    keep_every_n_steps: if defined, keep every checkpoints every n steps (in
      addition to keeping the last 'keep' checkpoints).
    async_manager: if defined, the save will run without blocking the main
      thread. Only works for single host. Note that an ongoing save will still
      block subsequent saves, to make sure overwrite/keep logic works correctly.
    gda_manager: required if target contains a JAX GlobalDeviceArray. Will save
      the GDAs to a separate subdirectory with postfix "_gda" asynchronously. 
      Same as async_manager, this will block subsequent saves.
  Returns:
    Filename of saved checkpoint.
  """

  def _save_checkpoint_files(target: bytes, paths: Tuple[str, str],
                             checkpoint_files: List[Any], keep: int,
                             overwrite: bool,
                             keep_every_n_steps: Optional[int]):
    """Save the checkpoint bytes via file system."""
    ckpt_tmp_path, ckpt_path = paths
    gfile.makedirs(os.path.dirname(ckpt_path))
    if ckpt_path in checkpoint_files:
      if not overwrite:
        raise errors.InvalidCheckpointError(ckpt_path, step)
    else:
      checkpoint_files.append(ckpt_path)

    checkpoint_files = natural_sort(checkpoint_files)
    # Handle the case if the job was preempted after the temporary checkpoint
    # was written, but before it was renamed to the final checkpoint name
    if checkpoint_files[-1] == ckpt_tmp_path:
      checkpoint_files.pop()
    if ckpt_path != checkpoint_files[-1]:
      if not overwrite:
        raise errors.InvalidCheckpointError(ckpt_path, step)

    with gfile.GFile(ckpt_tmp_path, 'wb') as fp:
      fp.write(target)

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
        gfile.rmtree(path)

    # Remove old checkpoint files.
    last_kept = -float('inf')
    if len(checkpoint_files) > keep:
      exclude_gda = [f for f in checkpoint_files if f[:-4] != '_gda']
      ind = checkpoint_files.index(exclude_gda[-keep])
      old_ckpts = checkpoint_files[:ind]
      # Note: old_ckpts is sorted from oldest to newest.
      for path in old_ckpts:
        if keep_every_n_steps:
          step_number = _checkpoint_path_step(path)
          if step_number and (step_number - last_kept) >= keep_every_n_steps:
            logging.debug('Not deleting %s, because last_kept=%f and keeping '
                          'every %d steps.',
                          path, last_kept, keep_every_n_steps)
            last_kept = step_number
            continue
        logging.info('Removing checkpoint at %s', path)
        gfile.remove(path)

  # Make sure all saves are finished before the logic of checking and removing
  # outdated checkpoints happens.
  if async_manager:
    async_manager.wait_previous_save()
  if gda_manager:
    gda_manager.wait_until_finished()

  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  # Write temporary checkpoint file.
  logging.info('Saving checkpoint at step: %s', step)
  # normalize path because gfile.glob() can modify path './', '//' ...
  ckpt_dir = safe_normpath(ckpt_dir)
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
  base_path = os.path.join(ckpt_dir, prefix)
  checkpoint_files = gfile.glob(base_path + '*')

  target = serialization.to_state_dict(target)
  target, gda_targets = _split_gdas(target)
  target = serialization.msgpack_serialize(target)

  # Save the files via I/O sync or async.
  def save_task():
    return _save_checkpoint_files(target, (ckpt_tmp_path, ckpt_path),
                                  checkpoint_files, keep, overwrite,
                                  keep_every_n_steps)
  if process_index() == 0:
    if async_manager:
      async_manager.save_async(save_task)
    else:
      save_task()

  if gda_targets:
    if not gda_manager:
      raise errors.GDACheckpointingRequiredError(ckpt_path, step)
    gda_tmp_path, gda_final_path = ckpt_tmp_path + '_gda', ckpt_path + '_gda'
    _save_gdas(gda_manager, gda_targets, gda_tmp_path, gda_final_path)

  return ckpt_path


def latest_checkpoint(ckpt_dir: Union[str, os.PathLike],
                      prefix: str = 'checkpoint_') -> Optional[str]:
  """Retrieve the path of the latest checkpoint in a directory.

  Args:
    ckpt_dir: str: directory of checkpoints to restore from.
    prefix: str: name prefix of checkpoint files.

  Returns:
    The latest checkpoint path or None if no checkpoints were found.
  """
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  checkpoint_files = natural_sort(gfile.glob(glob_path))
  ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
  checkpoint_files = [
      f for f in checkpoint_files
      if f != ckpt_tmp_path and not f.endswith('_gda')
  ]
  if checkpoint_files:
    return checkpoint_files[-1]
  else:
    return None


def restore_checkpoint(
    ckpt_dir: Union[str, os.PathLike],
    target: Optional[Any],
    step: Optional[int] = None,
    prefix: str = 'checkpoint_',
    parallel: bool = True,
    gda_manager: Optional[GlobalAsyncCheckpointManager] = None) -> PyTree:
  """Restore last/best checkpoint from checkpoints in path.

  Sorts the checkpoint files naturally, returning the highest-valued
  file, e.g.:

  *  ``ckpt_1, ckpt_2, ckpt_3 --> ckpt_3``

  *  ``ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1``

  *  ``ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5``

  Args:
    ckpt_dir: str: checkpoint file or directory of checkpoints to restore from.
    target: matching object to rebuild via deserialized state-dict. If None, the
      deserialized state-dict is returned as-is.
    step: int: step number to load or None to load latest. If specified,
      ckpt_dir must be a directory.
    prefix: str: name prefix of checkpoint files.
    parallel: bool: whether to load seekable checkpoints in parallel, for speed.
    gda_manager: required if checkpoint contains a JAX GlobalDeviceArray. Will
      read the GDAs from the separate subdirectory with postfix "_gda".

  Returns:
    Restored `target` updated from checkpoint file, or if no step specified and
    no checkpoint files present, returns the passed-in `target` unchanged.
    If a file path is specified and is not found, the passed-in `target` will be
    returned. This is to match the behavior of the case where a directory path
    is specified but the directory has not yet been created.
  """
  ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
  ckpt_dir = safe_normpath(ckpt_dir)
  if step is not None:
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    if not gfile.exists(ckpt_path):
      raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
  else:
    if not gfile.exists(ckpt_dir):
      logging.info('Found no checkpoint directory at %s', ckpt_dir)
      return target
    if not gfile.isdir(ckpt_dir):
      ckpt_path = ckpt_dir
    else:
      ckpt_path = latest_checkpoint(ckpt_dir, prefix)
      if not ckpt_path:
        logging.info('Found no checkpoint files in %s with prefix %s',
                     ckpt_dir, prefix)
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
      pool.shutdown(wait=False)
      logging.debug(f'results: {list(results)}')
    else:
      checkpoint_contents = fp.read()

  state_dict = serialization.msgpack_restore(checkpoint_contents)
  state_dict = _restore_gdas(state_dict, target, ckpt_path, step, gda_manager)

  if target is None:
    return state_dict
  return serialization.from_state_dict(target, state_dict)


def convert_pre_linen(params: PyTree) -> PyTree:
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

  if isinstance(params, core.FrozenDict):
    params_renamed = core.freeze(params_renamed)
  return params_renamed
