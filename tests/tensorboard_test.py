# Copyright 2020 The Flax Authors.
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

# Lint as: python3
"""Tests for flax.metrics.tensorboard."""
import itertools
import tempfile

from absl.testing import absltest
import numpy as onp

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util
import tensorflow.compat.v2 as tf

from flax.metrics.tensorboard import SummaryWriter

def _process_event(event):
  for value in event.summary.value:
    yield {'wall_time': event.wall_time, 'step': event.step, 'value': value}

def _get_event_values(path):
  event_file_generator = directory_watcher.DirectoryWatcher(
      path, event_file_loader.EventFileLoader).Load()
  return itertools.chain.from_iterable(map(_process_event,
          event_file_generator))

def _get_event_values_list(event_values):
  event_value_list = []
  for value_dict in event_values:
    event_value_list.append(value_dict)
  return event_value_list


class TensorboardTest(absltest.TestCase):

  def test_summarywriter_scalar(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    # Write the scalar and check if the event exists and check data.
    float_value = 99.00
    summary_writer.scalar(tag='scalar_test', value=float_value, step=1)
    event_values = _get_event_values(path=log_dir)
    event_values_list = _get_event_values_list(event_values=event_values)

    self.assertLen(event_values_list, 1)
    self.assertEqual(event_values_list[0]['step'], 1)
    self.assertGreater(event_values_list[0]['wall_time'], 0.0)

    summary_value = event_values_list[0]['value']
    self.assertEqual(summary_value.tag, 'scalar_test')
    self.assertEqual(tensor_util.make_ndarray(summary_value.tensor).item(),
        float_value)

  def test_summarywriter_text(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    text = 'hello world.'
    summary_writer.text(tag='text_test', textdata=text, step=1)
    event_values = _get_event_values(path=log_dir)
    event_values_list = _get_event_values_list(event_values=event_values)

    self.assertLen(event_values_list, 1)
    self.assertEqual(event_values_list[0]['step'], 1)
    self.assertGreater(event_values_list[0]['wall_time'], 0.0)

    summary_value = event_values_list[0]['value']
    self.assertEqual(summary_value.tag, 'text_test')
    self.assertEqual(
        tensor_util.make_ndarray(summary_value.tensor).item().decode('utf-8'),
        text)

  def test_summarywriter_image(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    img = tf.random.uniform(shape=[30, 30, 3])
    img = tf.cast(255 * img, tf.uint8)
    summary_writer.image(tag='image_test', image=img, step=1)
    event_values = _get_event_values(path=log_dir)
    event_values_list = _get_event_values_list(event_values=event_values)

    self.assertLen(event_values_list, 1)
    self.assertEqual(event_values_list[0]['step'], 1)
    self.assertGreater(event_values_list[0]['wall_time'], 0.0)

    summary_value = event_values_list[0]['value']
    self.assertEqual(summary_value.tag, 'image_test')

    expected_img = tf.image.decode_image(summary_value.tensor.string_val[2])
    self.assertTrue(onp.allclose(img, expected_img.numpy()))

  def test_summarywriter_audio(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    audio = tf.random.uniform(shape=[2, 48000, 3], minval=-2.0, maxval=2.0)
    summary_writer.audio(tag='audio_test', audiodata=audio, step=1)
    event_values = _get_event_values(path=log_dir)
    event_values_list = _get_event_values_list(event_values=event_values)

    self.assertLen(event_values_list, 1)
    self.assertEqual(event_values_list[0]['step'], 1)
    self.assertGreater(event_values_list[0]['wall_time'], 0.0)

    summary_value = event_values_list[0]['value']
    self.assertEqual(summary_value.tag, 'audio_test')

    expected_audio = \
        tf.audio.decode_wav(summary_value.tensor.string_val[0]).audio
    # audio is trimmed to -1.0 to 1.0
    self.assertFalse(onp.allclose(audio.numpy()[0], expected_audio.numpy()))

    # trim the audio to assert the values.
    trimmed_audio = onp.clip(onp.squeeze(onp.array(audio[0])), -1, 1)

    self.assertTrue(
        onp.allclose(trimmed_audio, expected_audio.numpy(), atol=1e-04))

  def test_summarywriter_histogram(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    histogram = onp.arange(1000)
    summary_writer.histogram(tag='histogram_test', values=histogram, step=1)
    event_values = _get_event_values(path=log_dir)
    event_values_list = _get_event_values_list(event_values=event_values)

    self.assertLen(event_values_list, 1)
    self.assertEqual(event_values_list[0]['step'], 1)
    self.assertGreater(event_values_list[0]['wall_time'], 0.0)

    summary_value = event_values_list[0]['value']
    self.assertEqual(summary_value.tag, 'histogram_test')

    expected_histogram = tensor_util.make_ndarray(summary_value.tensor)
    self.assertTrue(expected_histogram.shape, (30, 3))
    self.assertTrue(
        onp.allclose(expected_histogram[0], (0.0, 33.3, 34.0), atol=1e-01))

if __name__ == '__main__':
  absltest.main()