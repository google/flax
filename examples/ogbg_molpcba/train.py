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

"""Library file for executing training and evaluation on ogbg-molpcba."""

import os
from typing import Any, Dict, Iterable, Tuple, Optional

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.core
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax
import sklearn.metrics
import tensorflow as tf

import input_pipeline
import models


def create_model(config: ml_collections.ConfigDict,
                 deterministic: bool) -> nn.Module:
  """Creates a Flax model, as specified by the config."""
  if config.model == 'GraphNet':
    return models.GraphNet(
        latent_size=config.latent_size,
        num_mlp_layers=config.num_mlp_layers,
        message_passing_steps=config.message_passing_steps,
        output_globals_size=config.num_classes,
        dropout_rate=config.dropout_rate,
        skip_connections=config.skip_connections,
        layer_norm=config.layer_norm,
        use_edge_model=config.use_edge_model,
        deterministic=deterministic)
  if config.model == 'GraphConvNet':
    return models.GraphConvNet(
        latent_size=config.latent_size,
        num_mlp_layers=config.num_mlp_layers,
        message_passing_steps=config.message_passing_steps,
        output_globals_size=config.num_classes,
        dropout_rate=config.dropout_rate,
        skip_connections=config.skip_connections,
        layer_norm=config.layer_norm,
        deterministic=deterministic)
  raise ValueError(f'Unsupported model: {config.model}.')


def create_optimizer(
    config: ml_collections.ConfigDict) -> optax.GradientTransformation:
  """Creates an optimizer, as specified by the config."""
  if config.optimizer == 'adam':
    return optax.adam(
        learning_rate=config.learning_rate)
  if config.optimizer == 'sgd':
    return optax.sgd(
        learning_rate=config.learning_rate,
        momentum=config.momentum)
  raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


def binary_cross_entropy_with_mask(*, logits: jnp.ndarray, labels: jnp.ndarray,
                                   mask: jnp.ndarray):
  """Binary cross entropy loss for unnormalized logits, with masked elements."""
  assert logits.shape == labels.shape == mask.shape
  assert len(logits.shape) == 2

  # To prevent propagation of NaNs during grad().
  # We mask over the loss for invalid targets later.
  labels = jnp.where(mask, labels, -1)

  # Numerically stable implementation of BCE loss.
  # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
  positive_logits = (logits >= 0)
  relu_logits = jnp.where(positive_logits, logits, 0)
  abs_logits = jnp.where(positive_logits, logits, -logits)
  return relu_logits - (logits * labels) + (
      jnp.log(1 + jnp.exp(-abs_logits)))


def predictions_match_labels(*, logits: jnp.ndarray, labels: jnp.ndarray,
                             **kwargs) -> jnp.ndarray:
  """Returns a binary array indicating where predictions match the labels."""
  del kwargs  # Unused.
  preds = (logits > 0)
  return (preds == labels).astype(jnp.float32)


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  """Adds a prefix to the keys of a dict, returning a new dict."""
  return {f'{prefix}_{key}': val for key, val in result.items()}


@flax.struct.dataclass
class MeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('labels', 'logits', 'mask'))):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    values = super().compute()
    labels = values['labels']
    logits = values['logits']
    mask = values['mask']

    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    probs = jax.nn.sigmoid(logits)
    num_tasks = labels.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      is_labeled = mask[:, task]
      if len(np.unique(labels[is_labeled, task])) >= 2:
        average_precisions[task] = sklearn.metrics.average_precision_score(
            labels[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

  accuracy: metrics.Average.from_fun(predictions_match_labels)
  loss: metrics.Average.from_output('loss')
  mean_average_precision: MeanAveragePrecision


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

  accuracy: metrics.Average.from_fun(predictions_match_labels)
  loss: metrics.Average.from_output('loss')


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Replaces the globals attribute with a constant feature for each graph."""
  return graphs._replace(
      globals=jnp.ones([graphs.n_node.shape[0], 1]))


def get_predicted_logits(state: train_state.TrainState,
                         graphs: jraph.GraphsTuple,
                         rngs: Optional[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
  """Get predicted logits from the network for input graphs."""
  pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
  logits = pred_graphs.globals
  return logits


def get_valid_mask(labels: jnp.ndarray,
                   graphs: jraph.GraphsTuple) -> jnp.ndarray:
  """Gets the binary mask indicating only valid labels and graphs."""
  # We have to ignore all NaN values - which indicate labels for which
  # the current graphs have no label.
  labels_mask = ~jnp.isnan(labels)

  # Since we have extra 'dummy' graphs in our batch due to padding, we want
  # to mask out any loss associated with the dummy graphs.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  graph_mask = jraph.get_graph_padding_mask(graphs)

  # Combine the mask over labels with the mask over graphs.
  return labels_mask & graph_mask[:, None]


@jax.jit
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray]
) -> Tuple[train_state.TrainState, metrics.Collection]:
  """Performs one update step over the current batch of graphs."""

  def loss_fn(params, graphs):
    curr_state = state.replace(params=params)

    # Extract labels.
    labels = graphs.globals

    # Replace the global feature for graph classification.
    graphs = replace_globals(graphs)

    # Compute logits and resulting loss.
    logits = get_predicted_logits(curr_state, graphs, rngs)
    mask = get_valid_mask(labels, graphs)
    loss = binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask)
    mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

    return mean_loss, (loss, logits, labels, mask)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss, logits, labels, mask)), grads = grad_fn(state.params, graphs)
  state = state.apply_gradients(grads=grads)

  metrics_update = TrainMetrics.single_from_model_output(
      loss=loss, logits=logits, labels=labels, mask=mask)
  return state, metrics_update


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> metrics.Collection:
  """Computes metrics over a set of graphs."""

  # The target labels our model has to predict.
  labels = graphs.globals

  # Replace the global feature for graph classification.
  graphs = replace_globals(graphs)

  # Get predicted logits, and corresponding probabilities.
  logits = get_predicted_logits(state, graphs, rngs=None)

  # Get the mask for valid labels and graphs.
  mask = get_valid_mask(labels, graphs)

  # Compute the various metrics.
  loss = binary_cross_entropy_with_mask(logits=logits, labels=labels, mask=mask)

  return EvalMetrics.single_from_model_output(
      loss=loss, logits=logits, labels=labels, mask=mask)


def evaluate_model(state: train_state.TrainState,
                   datasets: Dict[str, tf.data.Dataset],
                   splits: Iterable[str]) -> Dict[str, metrics.Collection]:
  """Evaluates the model on metrics over the specified splits."""

  # Loop over each split independently.
  eval_metrics = {}
  for split in splits:
    split_metrics = None

    # Loop over graphs.
    for graphs in datasets[split].as_numpy_iterator():
      split_metrics_update = evaluate_step(state, graphs)

      # Update metrics.
      if split_metrics is None:
        split_metrics = split_metrics_update
      else:
        split_metrics = split_metrics.merge(split_metrics_update)
    eval_metrics[split] = split_metrics

  return eval_metrics  # pytype: disable=bad-return-type


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the TensorBoard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  # We only support single-host training.
  assert jax.process_count() == 1

  # Create writer for logs.
  writer = metric_writers.create_default_writer(workdir)
  writer.write_hparams(config.to_dict())

  # Get datasets, organized by split.
  logging.info('Obtaining datasets.')
  datasets = input_pipeline.get_datasets(
      config.batch_size,
      add_virtual_node=config.add_virtual_node,
      add_undirected_edges=config.add_undirected_edges,
      add_self_loops=config.add_self_loops)
  train_iter = iter(datasets['train'])

  # Create and initialize the network.
  logging.info('Initializing network.')
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  init_graphs = next(datasets['train'].as_numpy_iterator())
  init_graphs = replace_globals(init_graphs)
  init_net = create_model(config, deterministic=True)
  params = jax.jit(init_net.init)(init_rng, init_graphs)
  parameter_overview.log_parameter_overview(params)

  # Create the optimizer.
  tx = create_optimizer(config)

  # Create the training state.
  net = create_model(config, deterministic=False)
  state = train_state.TrainState.create(
      apply_fn=net.apply, params=params, tx=tx)

  # Set up checkpointing of the model.
  # The input pipeline cannot be checkpointed in its current form,
  # due to the use of stateful operations.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Create the evaluation state, corresponding to a deterministic model.
  eval_net = create_model(config, deterministic=True)
  eval_state = state.replace(apply_fn=eval_net.apply)

  # Hooks called periodically during training.
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  hooks = [report_progress, profiler]

  # Begin training loop.
  logging.info('Starting training.')
  train_metrics = None
  for step in range(initial_step, config.num_train_steps + 1):

    # Split PRNG key, to ensure different 'randomness' for every step.
    rng, dropout_rng = jax.random.split(rng)

    # Perform one step of training.
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      graphs = jax.tree_util.tree_map(np.asarray, next(train_iter))
      state, metrics_update = train_step(
          state, graphs, rngs={'dropout': dropout_rng})

      # Update metrics.
      if train_metrics is None:
        train_metrics = metrics_update
      else:
        train_metrics = train_metrics.merge(metrics_update)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 10, step)
    for hook in hooks:
      hook(step)

    # Log, if required.
    is_last_step = (step == config.num_train_steps - 1)
    if step % config.log_every_steps == 0 or is_last_step:
      writer.write_scalars(step,
                           add_prefix_to_keys(train_metrics.compute(), 'train'))
      train_metrics = None

    # Evaluate on validation and test splits, if required.
    if step % config.eval_every_steps == 0 or is_last_step:
      eval_state = eval_state.replace(params=state.params)

      splits = ['validation', 'test']
      with report_progress.timed('eval'):
        eval_metrics = evaluate_model(eval_state, datasets, splits=splits)
      for split in splits:
        writer.write_scalars(
            step, add_prefix_to_keys(eval_metrics[split].compute(), split))

    # Checkpoint model, if required.
    if step % config.checkpoint_every_steps == 0 or is_last_step:
      with report_progress.timed('checkpoint'):
        ckpt.save(state)

  return state
