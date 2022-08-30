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

"""Trains an SST2 text classifier."""
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

from absl import logging
from flax import struct
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

import input_pipeline
import models


Array = jnp.ndarray
Example = Dict[str, Array]
TrainState = train_state.TrainState


class Metrics(struct.PyTreeNode):
  """Computed metrics."""
  loss: float
  accuracy: float
  count: Optional[int] = None


@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: Array, logits: Array) -> Array:
  """Sigmoid cross entropy loss."""
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = (logits >= zeros)
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))


def get_initial_params(rng, model):
  """Returns randomly initialized parameters."""
  token_ids = jnp.ones((2, 3), jnp.int32)
  lengths = jnp.ones((2,), dtype=jnp.int32)
  variables = model.init(rng, token_ids, lengths, deterministic=True)
  return variables['params']


def create_train_state(rng, config: ml_collections.ConfigDict, model):
  """Create initial training state."""
  params = get_initial_params(rng, model)
  tx = optax.chain(
      optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum),
      optax.add_decayed_weights(weight_decay=config.weight_decay))
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return state


def compute_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided."""
  if labels.ndim == 1:  # Prevent the labels from broadcasting over the logits.
    labels = jnp.expand_dims(labels, axis=1)
  loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  binary_predictions = (logits >= 0.)
  binary_accuracy = jnp.equal(binary_predictions, labels)
  return Metrics(
      loss=jnp.sum(loss),
      accuracy=jnp.sum(binary_accuracy),
      count=logits.shape[0])


def model_from_config(config: ml_collections.ConfigDict):
  """Builds a text classification model from a config."""
  model = models.TextClassifier(
      embedding_size=config.embedding_size,
      hidden_size=config.hidden_size,
      vocab_size=config.vocab_size,
      output_size=config.output_size,
      dropout_rate=config.dropout_rate,
      word_dropout_rate=config.word_dropout_rate,
      unk_idx=config.unk_idx)
  return model


def train_step(
    state: TrainState,
    batch: Dict[str, Array],
    rngs: Dict[str, Any],
) -> Tuple[TrainState, Metrics]:
  """Train for a single step."""
  # Make sure to get a new RNG at every step.
  step = state.step
  rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

  def loss_fn(params):
    variables = {'params': params}
    logits = state.apply_fn(
        variables, batch['token_ids'], batch['length'],
        deterministic=False,
        rngs=rngs)

    labels = batch['label']
    if labels.ndim == 1:
      labels = jnp.expand_dims(labels, 1)
    loss = jnp.mean(
        sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  value, grads = grad_fn(state.params)
  (_, logits) = value

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return new_state, metrics


def eval_step(state: TrainState, batch: Dict[str, Array],
              rngs: Dict[str, Any]) -> Metrics:
  """Evaluate for a single step. Model should be in deterministic mode."""
  variables = {'params': state.params}
  logits = state.apply_fn(
      variables, batch['token_ids'], batch['length'],
      deterministic=True,
      rngs=rngs)
  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return metrics


def normalize_batch_metrics(
        batch_metrics: Sequence[Metrics]) -> Metrics:
  """Consolidates and normalizes a list of per-batch metrics dicts."""
  # Here we sum the metrics that were already summed per batch.
  total_loss = np.sum([metrics.loss for metrics in batch_metrics])
  total_accuracy = np.sum([metrics.accuracy for metrics in batch_metrics])
  total = np.sum([metrics.count for metrics in batch_metrics])
  # Divide each metric by the total number of items in the data set.
  return Metrics(
      loss=total_loss.item() / total, accuracy=total_accuracy.item() / total)


def batch_to_numpy(batch: Dict[str, tf.Tensor]) -> Dict[str, Array]:
  """Converts a batch with TF tensors to a batch of NumPy arrays."""
  # _numpy() reuses memory, does not make a copy.
  # pylint: disable=protected-access
  return jax.tree_util.tree_map(lambda x: x._numpy(), batch)


def evaluate_model(
        eval_step_fn: Callable[..., Any],
        state: TrainState,
        batches: Union[Iterable[Example], tf.data.Dataset],
        epoch: int,
        rngs: Optional[Dict[str, Any]] = None
) -> Metrics:
  """Evaluate a model on a dataset."""
  batch_metrics = []
  for i, batch in enumerate(batches):
    batch = batch_to_numpy(batch)
    if rngs is not None:  # New RNG for each step.
      rngs = {name: jax.random.fold_in(rng, i) for name, rng in rngs.items()}

    metrics = eval_step_fn(state, batch, rngs)
    batch_metrics.append(metrics)

  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)
  logging.info('eval  epoch %03d loss %.4f accuracy %.2f', epoch,
               metrics.loss, metrics.accuracy * 100)
  return metrics


def train_epoch(train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
                state: TrainState,
                train_batches: tf.data.Dataset,
                epoch: int,
                rngs: Optional[Dict[str, Any]] = None
                ) -> Tuple[TrainState, Metrics]:
  """Train for a single epoch."""
  batch_metrics = []
  for batch in train_batches:
    batch = batch_to_numpy(batch)
    state, metrics = train_step_fn(state, batch, rngs)
    batch_metrics.append(metrics)

  # Compute the metrics for this epoch.
  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)

  logging.info('train epoch %03d loss %.4f accuracy %.2f', epoch,
               metrics.loss, metrics.accuracy * 100)

  return state, metrics


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The final train state that includes the trained parameters.
  """
  # Prepare datasets.
  train_dataset = input_pipeline.TextDataset(
      tfds_name='glue/sst2', split='train')
  eval_dataset = input_pipeline.TextDataset(
      tfds_name='glue/sst2', split='validation')
  train_batches = train_dataset.get_bucketed_batches(
      config.batch_size,
      config.bucket_size,
      max_input_length=config.max_input_length,
      drop_remainder=True,
      shuffle=True,
      shuffle_seed=config.seed)
  eval_batches = eval_dataset.get_batches(batch_size=config.batch_size)

  # Keep track of vocab size in the config so that the embedder knows it.
  config.vocab_size = len(train_dataset.vocab)

  # Compile step functions.
  train_step_fn = jax.jit(train_step)
  eval_step_fn = jax.jit(eval_step)

  # Create model and a state that contains the parameters.
  rng = jax.random.PRNGKey(config.seed)
  model = model_from_config(config)
  state = create_train_state(rng, config, model)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  # Main training loop.
  logging.info('Starting training...')
  for epoch in range(1, config.num_epochs + 1):

    # Train for one epoch.
    rng, epoch_rng = jax.random.split(rng)
    rngs = {'dropout': epoch_rng}
    state, train_metrics = train_epoch(
        train_step_fn, state, train_batches, epoch, rngs)

    # Evaluate current model on the validation data.
    eval_metrics = evaluate_model(eval_step_fn, state, eval_batches, epoch)

    # Write metrics to TensorBoard.
    summary_writer.scalar('train_loss', train_metrics.loss, epoch)
    summary_writer.scalar(
        'train_accuracy',
        train_metrics.accuracy * 100,
        epoch)
    summary_writer.scalar('eval_loss', eval_metrics.loss, epoch)
    summary_writer.scalar(
        'eval_accuracy',
        eval_metrics.accuracy * 100,
        epoch)

  summary_writer.flush()
  return state
