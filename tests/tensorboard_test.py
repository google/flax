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


class TensorboardTest(absltest.TestCase):

  def parse_and_return_summary_value(self, path):
    """Parse the event file in the given path and return the
    only summary value."""
    event_value_list = []
    event_file_generator = directory_watcher.DirectoryWatcher(
        path, event_file_loader.EventFileLoader).Load()
    event_values = itertools.chain.from_iterable(
        map(_process_event, event_file_generator))
    for value_dict in event_values:
      event_value_list.append(value_dict)

    self.assertLen(event_value_list, 1)
    self.assertEqual(event_value_list[0]['step'], 1)
    self.assertGreater(event_value_list[0]['wall_time'], 0.0)
    return event_value_list[0]['value']

  def test_summarywriter_flush_after_close(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    summary_writer.close()
    with self.assertRaises(AttributeError):
      summary_writer.flush()

  def test_summarywriter_scalar(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    # Write the scalar and check if the event exists and check data.
    float_value = 99.1232
    summary_writer.scalar(tag='scalar_test', value=float_value, step=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'scalar_test')
    self.assertTrue(onp.allclose(
        tensor_util.make_ndarray(summary_value.tensor).item(),
        float_value))

  def test_summarywriter_text(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    text = 'hello world.'
    summary_writer.text(tag='text_test', textdata=text, step=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'text_test')
    self.assertEqual(
        tensor_util.make_ndarray(summary_value.tensor).item().decode('utf-8'),
        text)

  def test_summarywriter_image(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    expected_img = onp.random.uniform(low=0., high=255., size=(30, 30, 3))
    expected_img = expected_img.astype(onp.uint8)
    summary_writer.image(tag='image_test', image=expected_img, step=1)
    summary_value = self.parse_and_return_summary_value(path=log_dir)

    self.assertEqual(summary_value.tag, 'image_test')
    actual_img = tf.image.decode_image(summary_value.tensor.string_val[2])
    self.assertTrue(onp.allclose(actual_img, expected_img))

  def test_summarywriter_image_float_pixel_values(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    expected_img = onp.random.uniform(low=0., high=1., size=(30, 30, 3))
    summary_writer.image(tag='image_test', image=expected_img, step=1)
    summary_value = self.parse_and_return_summary_value(path=log_dir)

    # convert and scale expected_img appropriately to numpy uint8.
    expected_img = tf.image.convert_image_dtype(
        image=expected_img, dtype=onp.uint8)

    self.assertEqual(summary_value.tag, 'image_test')
    actual_img = tf.image.decode_image(summary_value.tensor.string_val[2])
    self.assertTrue(onp.allclose(actual_img, expected_img))

  def test_summarywriter_2dimage_scaled(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    img = onp.random.uniform(low=0., high=255., size=(30, 30))
    img = img.astype(onp.uint8)
    summary_writer.image(tag='2dimage_test', image=img, step=1)
    summary_value = self.parse_and_return_summary_value(path=log_dir)

    self.assertEqual(summary_value.tag, '2dimage_test')
    actual_img = tf.image.decode_image(summary_value.tensor.string_val[2])
    # assert the image was increased in dimension
    self.assertEqual(actual_img.shape, (30, 30, 3))

  def test_summarywriter_single_channel_image_scaled(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    img = onp.random.uniform(low=0., high=255., size=(30, 30, 1))
    img = img.astype(onp.uint8)
    summary_writer.image(tag='2dimage_1channel_test', image=img, step=1)
    summary_value = self.parse_and_return_summary_value(path=log_dir)

    self.assertEqual(summary_value.tag, '2dimage_1channel_test')
    actual_img = tf.image.decode_image(summary_value.tensor.string_val[2])
    # assert the image was increased in dimension
    self.assertEqual(actual_img.shape, (30, 30, 3))

  def test_summarywriter_audio(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    expected_audio_samples = onp.random.uniform(
        low=-1., high=1., size=(2, 48000, 2))
    summary_writer.audio(
        tag='audio_test', audiodata=expected_audio_samples, step=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'audio_test')

    # Assert two audio files are parsed.
    self.assertLen(summary_value.tensor.string_val, 2)

    # Assert values.
    actual_audio_1 = tf.audio.decode_wav(
        summary_value.tensor.string_val[0]).audio
    self.assertTrue(onp.allclose(
        expected_audio_samples[0], actual_audio_1, atol=1e-04))

    actual_audio_2 = tf.audio.decode_wav(
        summary_value.tensor.string_val[1]).audio
    self.assertTrue(onp.allclose(
        expected_audio_samples[1], actual_audio_2, atol=1e-04))

  def test_summarywriter_audio_sampled_output(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    expected_audio_samples = onp.random.uniform(
        low=-1., high=1., size=(2, 48000, 2))
    summary_writer.audio(
        tag='audio_test', audiodata=expected_audio_samples, step=1,
        max_outputs=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'audio_test')

    # Assert only the first audio clip is available.
    self.assertLen(summary_value.tensor.string_val, 1)

    # Assert values.
    actual_audio = tf.audio.decode_wav(summary_value.tensor.string_val[0]).audio

    self.assertTrue(onp.allclose(
        expected_audio_samples[0], actual_audio, atol=1e-04))

  def test_summarywriter_clipped_audio(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    expected_audio_samples = onp.random.uniform(
        low=-2., high=2., size=(2, 48000, 2))
    summary_writer.audio(
        tag='audio_test', audiodata=expected_audio_samples, step=1,
        max_outputs=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'audio_test')

    # Assert one audio files is parsed.
    self.assertLen(summary_value.tensor.string_val, 1)

    # actual_audio is clipped.
    actual_audio = tf.audio.decode_wav(
        summary_value.tensor.string_val[0]).audio
    self.assertFalse(onp.allclose(
        expected_audio_samples[0], actual_audio, atol=1e-04))

    clipped_audio = onp.clip(onp.array(expected_audio_samples[0]), -1, 1)
    self.assertTrue(
        onp.allclose(clipped_audio, actual_audio, atol=1e-04))

  def test_summarywriter_histogram_defaultbins(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    histogram = onp.arange(1000)
    # Histogram will be created for 30 (default) bins.
    summary_writer.histogram(tag='histogram_test', values=histogram, step=1)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'histogram_test')
    actual_histogram = tensor_util.make_ndarray(summary_value.tensor)
    self.assertTrue(actual_histogram.shape, (30, 3))
    self.assertTrue(
        onp.allclose(actual_histogram[0], (0.0, 33.3, 34.0), atol=1e-01))

  def test_summarywriter_histogram_2bins(self):
    log_dir = tempfile.mkdtemp()
    summary_writer = SummaryWriter(log_dir=log_dir)
    histogram = onp.arange(1000)
    summary_writer.histogram(
        tag='histogram_test', values=histogram, step=1, bins=2)

    summary_value = self.parse_and_return_summary_value(path=log_dir)
    self.assertEqual(summary_value.tag, 'histogram_test')
    actual_histogram = tensor_util.make_ndarray(summary_value.tensor)
    self.assertTrue(actual_histogram.shape, (2, 3))
    self.assertTrue(
        onp.allclose(actual_histogram[0], (0.0, 499.5, 500.0), atol=1e-01))
    self.assertTrue(
        onp.allclose(actual_histogram[1], (499.5, 999.0, 500.0), atol=1e-01))

if __name__ == '__main__':
  absltest.main()
