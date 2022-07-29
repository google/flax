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

"""Tests for flax.traceback_util."""

import contextlib
import traceback
import sys
from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import random
from jax._src import traceback_util as jax_traceback_util
from flax import linen as nn
from flax import traceback_util


# pylint: disable=arguments-differ,protected-access, g-wrong-blank-lines

# __tracebackhide__ is a python >=3.7 feature.
TRACEBACKHIDE_SUPPORTED = tuple(sys.version_info)[:3] >= (3, 7, 0)

EXPECTED_FILES = (__file__, contextlib.__spec__.origin)


class TracebackTest(absltest.TestCase):

  def test_exclusion_list(self):
    traceback_util.show_flax_in_tracebacks()
    exclusion_len_wo_flax = len(jax_traceback_util._exclude_paths)
    traceback_util.hide_flax_in_tracebacks()
    exclusion_len_w_flax = len(jax_traceback_util._exclude_paths)
    self.assertLen(
        traceback_util._flax_exclusions,
        exclusion_len_w_flax - exclusion_len_wo_flax)

  def test_simple_exclusion_tracebackhide(self):
    if not TRACEBACKHIDE_SUPPORTED:
      return
    class Test1(nn.Module):
      @nn.remat
      @nn.compact
      def __call__(self, x):
        return Test2()(x)
    class Test2(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        raise ValueError('error here.')
        return x  # pylint: disable=unreachable

    traceback_util.hide_flax_in_tracebacks()
    jax.config.update('jax_traceback_filtering', 'tracebackhide')

    key = random.PRNGKey(0)
    try:
      nn.jit(Test1)().init(key, jnp.ones((5, 3)))
    except ValueError as e:
      tb = e.__traceback__

    filtered_frames = 0
    unfiltered_frames = 0

    for f, _ in traceback.walk_tb(tb):
      if '__tracebackhide__' not in f.f_locals:
        self.assertIn(f.f_code.co_filename, EXPECTED_FILES)
        filtered_frames += 1
      unfiltered_frames += 1

    self.assertEqual(filtered_frames, 3)
    self.assertGreater(unfiltered_frames, filtered_frames)


  def test_simple_exclusion_remove_frames(self):
    class Test1(nn.Module):
      @nn.remat
      @nn.compact
      def __call__(self, x):
        return Test2()(x)
    class Test2(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        raise ValueError('error here.')
        return x  # pylint: disable=unreachable

    traceback_util.hide_flax_in_tracebacks()
    jax.config.update('jax_traceback_filtering', 'remove_frames')

    key = random.PRNGKey(0)
    try:
      nn.jit(Test1)().init(key, jnp.ones((5, 3)))
    except ValueError as e:
      tb_filtered = e.__traceback__
      tb_unfiltered = e.__cause__.__traceback__
      e_cause = e.__cause__

    self.assertIsInstance(e_cause, jax_traceback_util.UnfilteredStackTrace)

    filtered_frames = 0
    for _, _ in traceback.walk_tb(tb_filtered):
      filtered_frames += 1

    unfiltered_frames = 0
    for _, _ in traceback.walk_tb(tb_unfiltered):
      unfiltered_frames += 1

    self.assertEqual(filtered_frames, 3)
    self.assertGreater(unfiltered_frames, filtered_frames)


  def test_dynamic_exclusion(self):

    if not TRACEBACKHIDE_SUPPORTED:
      return

    class Test1(nn.Module):
      @nn.remat
      @nn.compact
      def __call__(self, x):
        return Test2()(x)
    class Test2(nn.Module):
      @nn.jit
      @nn.compact
      def __call__(self, x):
        raise ValueError('error here.')
        return x  # pylint: disable=unreachable

    key = random.PRNGKey(0)

    traceback_util.show_flax_in_tracebacks()
    jax.config.update('jax_traceback_filtering', 'off')
    try:
      nn.jit(Test1)().init(key, jnp.ones((5, 3)))
    except ValueError as e:
      tb_all = e.__traceback__

    traceback_util.hide_flax_in_tracebacks()
    jax.config.update('jax_traceback_filtering', 'tracebackhide')
    try:
      nn.jit(Test1)().init(key, jnp.ones((5, 3)))
    except ValueError as e:
      tb_no_flax = e.__traceback__

    traceback_util.show_flax_in_tracebacks()
    jax.config.update('jax_traceback_filtering', 'tracebackhide')
    try:
      nn.jit(Test1)().init(key, jnp.ones((5, 3)))
    except ValueError as e:
      tb_w_flax = e.__traceback__

    filtered_frames_all = 0
    unfiltered_frames_all = 0
    for f, _ in traceback.walk_tb(tb_all):
      if '__tracebackhide__' not in f.f_locals:
        unfiltered_frames_all += 1
      else:
        filtered_frames_all += 1

    filtered_frames_no_flax = 0
    unfiltered_frames_no_flax = 0
    for f, _ in traceback.walk_tb(tb_no_flax):
      if '__tracebackhide__' not in f.f_locals:
        self.assertIn(f.f_code.co_filename, EXPECTED_FILES)
        unfiltered_frames_no_flax += 1
      else:
        filtered_frames_no_flax += 1

    filtered_frames_w_flax = 0
    unfiltered_frames_w_flax = 0
    for f, _ in traceback.walk_tb(tb_w_flax):
      if '__tracebackhide__' not in f.f_locals:
        unfiltered_frames_w_flax += 1
      else:
        filtered_frames_w_flax += 1

    self.assertEqual(unfiltered_frames_all + filtered_frames_all,
                     unfiltered_frames_w_flax + filtered_frames_w_flax)
    self.assertEqual(unfiltered_frames_all + filtered_frames_all,
                     unfiltered_frames_no_flax + filtered_frames_no_flax)
    self.assertEqual(unfiltered_frames_no_flax, 3)
    self.assertGreater(unfiltered_frames_all, unfiltered_frames_w_flax)
    self.assertGreater(unfiltered_frames_w_flax, unfiltered_frames_no_flax)


if __name__ == '__main__':
  absltest.main()
