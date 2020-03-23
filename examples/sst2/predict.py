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

"""Make predictions with a trained SST-2 model."""

import collections
import functools
import os
from typing import Any, Dict, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import flax
import flax.training.checkpoints
from flax import nn

import flax.examples.sst2.input_pipeline as input_pipeline
import flax.examples.sst2.model as sst2_model

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'batch_size', default=64,
    help=('Batch size for training.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory with model data.'))

flags.DEFINE_string(
    'vocab_path', default=None,
    help=('Path to vocabulary.'))


def get_predictions(logits):
  outputs = jax.nn.sigmoid(logits)
  return (outputs > 0.5).astype(jnp.int32)


@jax.jit
def predict_step(model: nn.Module, inputs: jnp.ndarray, lengths: jnp.ndarray):
  logits = model(inputs, lengths, train=False)
  return get_predictions(logits)


def predict(model: nn.Module, test_ds: tf.data.Dataset):
  result = []
  rng = jax.random.PRNGKey(0)
  with nn.stochastic(rng):
    for ex in tfds.as_numpy(test_ds):
      inputs, lengths, labels = ex['sentence'], ex['length'], ex['label']
      predictions = predict_step(model, inputs, lengths)
      result += predictions.flatten().tolist()
  return np.array(result)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('vocab_path')
  tf.enable_v2_behavior()

  # Prepare data.
  data_source = input_pipeline.SST2DataSource(
      batch_size=FLAGS.batch_size,
      vocab_path=FLAGS.vocab_path)

  # Create model.
  model = sst2_model.load_model(
      FLAGS.model_dir, data_source.vocab_size, FLAGS.batch_size)

  # Make test set predictions.
  test_predictions = predict(model, data_source.test_batches)

  # Let's look at the predictions together with the original sentence.
  num_examples = 10
  test_sentences = tfds.load('glue/sst2', split=tfds.Split.TEST)
  for original, prediction in zip(
      tfds.as_numpy(test_sentences.take(num_examples)),
      test_predictions[:num_examples]):
    logging.info('Sentence:   %s', original['sentence'].decode('utf8'))
    logging.info('Prediction: %s\n\n', 'positive' if prediction else 'negative')


if __name__ == '__main__':
  app.run(main)

