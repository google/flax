"""Benchmark using fake data for the ImageNet example."""
import itertools
import time

from absl import flags
from absl.testing import absltest
from absl.testing.flagsaver import flagsaver
import input_pipeline
import train
from flax.testing import Benchmark
import jax
import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
FLAGS = flags.FLAGS


class ImagenetBenchmarkFakeData(Benchmark):
  """Runs ImageNet using fake data for quickly measuring performance."""

  def setUp(self):
    super(ImagenetBenchmarkFakeData, self).setUp()

    def create_input_iter(batch_size, image_size, dtype, train, cache):
      image_shape = (batch_size, image_size, image_size, 3)
      fake_image = np.random.rand(*image_shape)
      fake_image = fake_image.astype(dtype.as_numpy_dtype)
      fake_image = fake_image.reshape(
          (jax.local_device_count(), -1) + fake_image.shape[1:])

      fake_label = np.random.randint(1, 1000, (batch_size,))
      fake_label = fake_label.astype(np.int32)
      fake_label = fake_label.reshape((jax.local_device_count(), -1))

      fake_batch = {'image': fake_image, 'label': fake_label}
      return itertools.repeat(fake_batch)

    self._real_create_input_iter = train.create_input_iter
    train.create_input_iter = create_input_iter

    self._real_train_images = input_pipeline.TRAIN_IMAGES
    input_pipeline.TRAIN_IMAGES = 1024
    self._real_eval_images = input_pipeline.EVAL_IMAGES
    input_pipeline.EVAL_IMAGES = 512

  def tearDown(self):
    super(ImagenetBenchmarkFakeData, self).tearDown()
    train.create_input_iter = self._real_create_input_iter
    input_pipeline.TRAIN_IMAGES = self._real_train_images
    input_pipeline.EVAL_IMAGES = self._real_eval_images

  @flagsaver
  def test_fake_data(self):
    model_dir = self.get_tmp_model_dir()
    FLAGS.batch_size = 256
    FLAGS.half_precision = True
    FLAGS.num_epochs = 5
    FLAGS.model_dir = model_dir

    start_time = time.time()
    train.main([])
    benchmark_time = time.time() - start_time

    self.report_wall_time(benchmark_time)


if __name__ == '__main__':
  absltest.main()
