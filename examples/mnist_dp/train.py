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

"""MNIST-DP example.

Library file which executes the training and evaluation loop for MNIST-DP.
"""

import functools
import logging
from typing import Dict, Optional, Tuple

import chex
from clu import metric_writers
import input_pipeline
import models
import optimizers
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import rdp_accountant

_NUM_CLASSES = 10


@jax.jit
def compute_loss(state: train_state.TrainState, images: jnp.ndarray,
                 labels: jnp.ndarray):
  logits = state.apply_fn(state.params, images)
  one_hot_labels = jax.nn.one_hot(labels, _NUM_CLASSES)
  loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  loss = jnp.mean(loss)
  return loss, logits


@jax.jit
def apply_model(state: train_state.TrainState, images: jnp.ndarray,
                labels: jnp.ndarray):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params: optax.Params, images: jnp.ndarray, labels: jnp.ndarray):
    curr_state = state.replace(params=params)
    return compute_loss(curr_state, images, labels)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params, images, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def apply_dp_model(state: train_state.TrainState, images: jnp.ndarray,
                   labels: jnp.ndarray):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params: optax.Params, image: jnp.ndarray, label: jnp.ndarray):
    curr_state = state.replace(params=params)
    image = jnp.expand_dims(image, axis=0)
    label = jnp.expand_dims(label, axis=0)
    return compute_loss(curr_state, image, label)[0]

  grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(
      state.params, images, labels)
  loss, logits = compute_loss(state, images, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state: train_state.TrainState, labels: jnp.ndarray,
                 grads: optax.Updates) -> train_state.TrainState:
  try:
    new_opt_state = state.opt_state._replace(labels=labels.astype(jnp.int32))
    state = state.replace(opt_state=new_opt_state)
  except AttributeError:
    pass
  return state.apply_gradients(grads=grads)


def evaluate_model(state: train_state.TrainState, dataset: tf.data.Dataset):
  """Evaluates the model on the given dataset."""
  test_loss = []
  test_accuracy = []
  matches_by_class = jnp.zeros(_NUM_CLASSES)
  counts_by_class = jnp.zeros(_NUM_CLASSES)
  for samples in dataset:
    samples = jax.tree_map(jnp.asarray, samples)
    images, labels = samples['image'], samples['label']
    loss, logits = compute_loss(state, images, labels)
    logits_match_labels = (jnp.argmax(logits, -1) == labels)
    accuracy = jnp.mean(logits_match_labels)
    matches_by_class = matches_by_class.at[labels].add(logits_match_labels)
    counts_by_class = counts_by_class.at[labels].add(1)

    test_loss.append(loss)
    test_accuracy.append(accuracy)

  test_accuracy_by_class = matches_by_class / counts_by_class
  return np.mean(test_loss), np.mean(test_accuracy), np.mean(
      test_accuracy_by_class)


def get_estimation_samples(
    train_ds: tf.data.Dataset) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Returns images and labels to estimate clipping thresholds."""
  samples = next(iter(train_ds))
  samples = jax.tree_map(jnp.asarray, samples)
  return samples['image'], samples['label']


def estimate_clipping_thresholds(
    apply_fn, params: optax.Params,
    l2_norm_clip_percentile: float,
    estimation_images: jnp.ndarray,
    estimation_labels: jnp.ndarray) -> chex.ArrayTree:
  """Estimates clipping thresholds."""
  def loss_fn(params: optax.Params, image: jnp.ndarray, label: jnp.ndarray):
    dummy_state = train_state.TrainState.create(apply_fn=apply_fn,
                                                params=params,
                                                tx=optax.identity())
    image = jnp.expand_dims(image, axis=0)
    label = jnp.expand_dims(label, axis=0)
    return compute_loss(dummy_state, image, label)[0]

  grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(
      params, estimation_images, estimation_labels)
  grad_norms = jax.tree_map(jax.vmap(jnp.linalg.norm), grads)
  l2_norms_clip = jax.tree_map(
      lambda g_norm: jnp.percentile(g_norm, l2_norm_clip_percentile),
      grad_norms)
  return l2_norms_clip


def create_model(rng: chex.PRNGKey):
  """Creates the model and initial parameters."""
  model = models.CNN()
  params = model.init(rng, jnp.ones([2, 28, 28, 1]))
  return model, params


def create_optimizer(
    apply_fn, params: optax.Params, config: ml_collections.ConfigDict,
    estimation_samples: Optional[Tuple[jnp.ndarray, ...]],
    init_seed: Optional[int] = None
) -> optax.GradientTransformation:
  """Creates the optimizer."""
  if config.differentially_private_training:
    l2_norms_clip = estimate_clipping_thresholds(
        apply_fn, params, config.l2_norm_clip_percentile, *estimation_samples)
    if init_seed is None:
      init_seed = config.dp_rng_seed
    privacy_params = {
        'l2_norms_clip': l2_norms_clip,
        'init_seed': init_seed,
        'noise_multiplier': config.noise_multiplier,
    }

  if config.optimizer == 'sgd':
    base_params = {
        'learning_rate': config.learning_rate,
        'momentum': config.momentum,
        'nesterov': config.nesterov,
    }
    if config.differentially_private_training:
      return optimizers.dpsgd(**base_params, **privacy_params)
    return optax.sgd(**base_params)

  if config.optimizer == 'adam':
    base_params = {
        'learning_rate': config.learning_rate,
    }
    if config.differentially_private_training:
      return optimizers.dpadam(**base_params, **privacy_params)
    return optax.adam(**base_params)

  raise ValueError(f'Unsupported optimizer: {config.optimizer}')


def create_train_state(rng: chex.PRNGKey,
                       config: ml_collections.ConfigDict,
                       estimation_samples: Optional[Tuple[jnp.ndarray, ...]]):
  """Creates initial `TrainState`."""
  model, params = create_model(rng)
  tx = create_optimizer(model.apply, params, config, estimation_samples)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def get_apply_model_fn(config: ml_collections.ConfigDict):
  """Returns an apply_model function."""
  if config.differentially_private_training:
    return apply_dp_model
  return apply_model


def dpsgd_privacy_accountant(num_training_steps: int,
                             noise_multiplier: float,
                             target_delta: float,
                             sampling_probability: float) -> float:
  """Computes epsilon after a given number of training steps with DP-SGD/Adam.

  Assumes there is only one affected term on removal of a node.
  Returns np.inf if the noise multiplier is too small.

  Args:
    num_training_steps: Number of training steps.
    noise_multiplier: Noise multiplier that scales the sensitivity.
    target_delta: Privacy parameter delta to choose epsilon for.
    sampling_probability: The probability of sampling a single sample every
      batch. For uniform sampling without replacement,
      this is (batch_size / num_samples).

  Returns:
    Privacy parameter epsilon.
  """
  if noise_multiplier < 1e-20:
    return np.inf

  orders = np.arange(1, 200, 0.1)[1:]
  rdp_const = rdp_accountant.compute_rdp(sampling_probability,
                                         noise_multiplier,
                                         num_training_steps, orders)
  epsilon = rdp_accountant.get_privacy_spent(orders, rdp_const,
                                             target_delta=target_delta)[0]
  return epsilon


def get_privacy_accountant(config: ml_collections.ConfigDict):
  """Returns an privacy accounting function."""
  if not config.differentially_private_training:
    return lambda step: None

  return functools.partial(
      dpsgd_privacy_accountant,
      noise_multiplier=config.noise_multiplier,
      target_delta=1 / (10 * config.num_training_nodes),
      sampling_probability=config.batch_size / config.num_training_nodes)


def get_max_training_epsilon(
    config: ml_collections.ConfigDict) -> Optional[float]:
  """Returns the privacy budget for DP training."""
  if not config.differentially_private_training:
    return None
  return config.max_training_epsilon


def log_metrics(step: int, metrics: Dict[str, float],
                summary_writer: metric_writers.MetricWriter):
  """Logs all metrics."""
  summary_writer.write_scalars(step, metrics)
  summary_writer.flush()

  logging.info(
      'num_steps:% 3d, train_loss: %.4f, micro_train_accuracy: %.2f',
      step, metrics['train_loss'], metrics['micro_train_accuracy'] * 100)
  logging.info(
      'num_steps:% 3d, val_loss: %.4f, micro_val_accuracy: %.2f',
      step, metrics['val_loss'], metrics['micro_val_accuracy'] * 100)
  logging.info(
      'num_steps:% 3d, test_loss: %.4f, micro_test_accuracy: %.2f',
      step, metrics['test_loss'], metrics['micro_test_accuracy'] * 100)
  if 'epsilon' in metrics:
    logging.info(
        'num_steps:% 3d, epsilon: %.3f',
        step, metrics['epsilon'])


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  datasets = input_pipeline.get_datasets(config)
  rng = jax.random.PRNGKey(config.train_rng_seed)

  summary_writer = metric_writers.create_default_writer(workdir)
  summary_writer.write_hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  estimation_samples = get_estimation_samples(datasets['train'])
  state = create_train_state(init_rng, config, estimation_samples)
  apply_model_fn = get_apply_model_fn(config)
  privacy_accountant = get_privacy_accountant(config)

  train_loss = []
  train_accuracy = []
  eval_cadence = config.evaluation_cadence
  max_epsilon = get_max_training_epsilon(config)

  for step, samples in enumerate(datasets['train']):
    samples = jax.tree_map(jnp.asarray, samples)
    images, labels = samples['image'], samples['label']
    grads, loss, accuracy = apply_model_fn(
        state, images, labels)
    state = update_model(state, labels, grads)

    train_loss.append(loss)
    train_accuracy.append(accuracy)

    if step % eval_cadence == (eval_cadence - 1):
      epsilon = privacy_accountant(step + 1)
      if max_epsilon is not None and epsilon >= max_epsilon:
        break

      mean_train_loss = np.mean(train_loss)
      micro_train_accuracy = np.mean(train_accuracy)
      mean_val_loss, micro_val_accuracy, macro_val_accuracy = evaluate_model(
          state, datasets['validation'])
      mean_test_loss, micro_test_accuracy, macro_test_accuracy = evaluate_model(
          state, datasets['test'])

      all_metrics = {
          'train_loss': mean_train_loss,
          'micro_train_accuracy': micro_train_accuracy,
          'val_loss': mean_val_loss,
          'micro_val_accuracy': micro_val_accuracy,
          'macro_val_accuracy': macro_val_accuracy,
          'test_loss': mean_test_loss,
          'micro_test_accuracy': micro_test_accuracy,
          'macro_test_accuracy': macro_test_accuracy,
      }
      if epsilon is not None:
        all_metrics['epsilon'] = epsilon

      log_metrics(step, all_metrics, summary_writer)

      train_loss = []
      train_accuracy = []

  return state
