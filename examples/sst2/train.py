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

"""Trains an SST2 text classifier."""
import copy
import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from flax import optim
from flax.metrics import tensorboard
import input_pipeline
import jax
import jax.numpy as jnp
import ml_collections
import models
import numpy as np
import tensorflow as tf

Array = jnp.ndarray
Example = Dict[str, Array]


@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: Array, logits: Array) -> Array:
  """Sigmoid cross entropy loss."""
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = (logits >= zeros)
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))


def get_initial_params_and_state(key, model):
  """Returns randomly initialized parameters and a fresh model state."""
  token_ids = jnp.ones((2, 3), jnp.int32)
  lengths = jnp.ones((2,), dtype=jnp.int32)
  variables = model.init(key, token_ids, lengths)
  state, params = variables.pop('params')
  return params, state


def create_optimizer(params, learning_rate, beta, weight_decay):
  """Returns a momentum optimizer."""
  optimizer_def = optim.Momentum(
      learning_rate=learning_rate,
      beta=beta,
      weight_decay=weight_decay)
  optimizer = optimizer_def.create(params)
  return optimizer


def compute_metrics(*, labels: Array, logits: Array) -> Dict[str, Array]:
  """Computes the metrics, summed across the batch if a batch is provided."""
  if labels.ndim == 1:  # Prevent the labels from broadcasting over the logits.
    labels = jnp.expand_dims(labels, axis=1)
  loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  binary_predictions = (logits >= 0.)
  binary_accuracy = jnp.equal(binary_predictions, labels)
  metrics = {
      'loss': jnp.sum(loss),
      'accuracy': jnp.sum(binary_accuracy),
      'count': logits.shape[0]
  }
  return metrics


def model_from_config(config: ml_collections.ConfigDict):
  """Builds a text classification model from a config."""
  model = models.TextClassifier(
      embedding_size=config.embedding_size,
      hidden_size=config.hidden_size,
      vocab_size=config.vocab_size,
      output_size=config.output_size,
      dropout_rate=config.dropout_rate,
      word_dropout_rate=config.word_dropout_rate,
      unk_idx=config.unk_idx,
      deterministic=config.deterministic)
  return model


def train_step(
    config: Any,
    optimizer: optim.Optimizer,
    model_state: Mapping[str, Any],
    batch: Dict[str, Array],
    rngs: Dict[str, Any],
) -> Tuple[optim.Optimizer, Dict[str, Any], Dict[str, Any]]:
  """Train for a single step."""
  # Make sure to get a new RNG at every step.
  model = model_from_config(config)
  step = optimizer.state.step
  rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

  def loss_fn(params):
    variables = {'params': params, **model_state}
    logits, new_model_state = model.apply(
        variables, batch['token_ids'], batch['length'],
        rngs=rngs, mutable=list(model_state.keys()))

    labels = batch['label']
    if labels.ndim == 1:
      labels = jnp.expand_dims(labels, 1)
    loss = jnp.mean(
        sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss, (logits, new_model_state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  value, grad = grad_fn(optimizer.target)
  (_, (logits, new_model_state)) = value
  optimizer = optimizer.apply_gradient(grad)

  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return optimizer, metrics, new_model_state


def eval_step(config: Any, params: Dict[str, Any],
              model_state: Mapping[str, Any], batch: Dict[str, Array],
              rngs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Evaluate for a single step. Model should be in deterministic mode."""
  model = model_from_config(config)
  variables = {'params': params, **model_state}
  logits, new_model_state = model.apply(
      variables, batch['token_ids'], batch['length'],
      rngs=rngs,
      mutable=list(model_state.keys()))
  metrics = compute_metrics(labels=batch['label'], logits=logits)
  return metrics, new_model_state


def normalize_batch_metrics(
        batch_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
  """Consolidates and normalizes a list of per-batch metrics dicts."""
  # Here we sum the metrics that were already summed per batch.
  metric_names = batch_metrics[0].keys()
  summed_metrics = {
      k: np.sum([metrics[k] for metrics in batch_metrics]) for k in metric_names
  }
  # Divide each metric by the total number of items in the data set.
  total = np.float(summed_metrics.pop('count'))
  metrics = jax.tree_map(lambda x: x.item() / total, summed_metrics)
  return metrics


def evaluate_model(
        eval_step_fn: Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]],
        params: Dict[str, Any],
        model_state: Mapping[str, Any],
        batches: Union[Iterable[Example], tf.data.Dataset],
        epoch: int,
        rngs: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Mapping[str, Any]]:
  """Evaluate a model on a dataset."""
  batch_metrics = []
  for i, batch in enumerate(batches):
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
    if rngs is not None:  # New RNG for each step.
      rngs = {name: jax.random.fold_in(rng, i) for name, rng in rngs.items()}

    metrics, model_state = eval_step_fn(params, model_state, batch, rngs)
    batch_metrics.append(metrics)

  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)
  logging.info('eval  epoch %03d loss %.4f accuracy %.2f', epoch,
               metrics['loss'], metrics['accuracy'] * 100)
  return metrics, model_state


def train_epoch(train_step_fn: Callable[..., Tuple[optim.Optimizer,
                                                   Dict[str, Any], Any]],
                optimizer: optim.Optimizer,
                model_state: Mapping[str, Any], train_batches: tf.data.Dataset,
                epoch: int, rngs: Optional[Dict[str, Any]] = None):
  """Train for a single epoch."""
  batch_metrics = []
  for batch in train_batches:
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
    optimizer, metrics, model_state = train_step_fn(
        optimizer, model_state, batch, rngs)
    batch_metrics.append(metrics)

  # Compute the metrics for this epoch.
  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)

  logging.info('train epoch %03d loss %.4f accuracy %.2f', epoch,
               metrics['loss'], metrics['accuracy'] * 100)

  return optimizer, metrics, model_state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> optim.Optimizer:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The trained optimizer.
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

  # Prepare configs.
  config.vocab_size = len(train_dataset.vocab)
  eval_config = copy.deepcopy(config)
  eval_config.deterministic = True

  # Compile step functions.
  train_step_fn = jax.jit(functools.partial(train_step, config))
  eval_step_fn = jax.jit(functools.partial(eval_step, eval_config))

  # Initialize parameters.
  rng = jax.random.PRNGKey(config.seed)
  init_model = model_from_config(eval_config)
  params, model_state = get_initial_params_and_state(rng, init_model)
  del init_model
  
  # Remove intermediates for training. Otherwise our model state will fill up
  # with intermediate outputs (exported using self.sow() commands). This will
  # cause model_state to have a new shape on each step, triggering a new trace.
  model_state, _ = model_state.pop('intermediates')  
  
  optimizer = create_optimizer(
      params, config.learning_rate, config.momentum, config.weight_decay)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  # Main training loop.
  logging.info('Starting training...')
  for epoch in range(1, config.num_epochs + 1):

    # Train for one epoch.
    rng, epoch_rng = jax.random.split(rng)
    rngs = {'dropout': epoch_rng}
    optimizer, train_metrics, model_state = train_epoch(
        train_step_fn, optimizer, model_state, train_batches, epoch, rngs)

    # Evaluate current model on the validation data.
    eval_metrics, _ = evaluate_model(
        eval_step_fn, optimizer.target, model_state, eval_batches, epoch)

    # Write metrics to TensorBoard.
    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar(
        'train_accuracy',
        train_metrics['accuracy'] * 100,
        epoch)
    summary_writer.scalar('eval_loss', eval_metrics['loss'], epoch)
    summary_writer.scalar(
        'eval_accuracy',
        eval_metrics['accuracy'] * 100,
        epoch)

  summary_writer.flush()
  return optimizer
