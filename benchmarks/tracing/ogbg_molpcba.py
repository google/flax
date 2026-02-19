# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OGBG-MolPCBA helper functions for benchmarking."""

from typing import Any

from clu import metrics
import flax
import flax.linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.ogbg_molpcba import models
from flax.examples.ogbg_molpcba.configs import default as ogbg_config
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import optax


def create_model(
    config: ml_collections.ConfigDict, deterministic: bool
) -> nn.Module:
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
        deterministic=deterministic,
    )
  if config.model == 'GraphConvNet':
    return models.GraphConvNet(
        latent_size=config.latent_size,
        num_mlp_layers=config.num_mlp_layers,
        message_passing_steps=config.message_passing_steps,
        output_globals_size=config.num_classes,
        dropout_rate=config.dropout_rate,
        skip_connections=config.skip_connections,
        layer_norm=config.layer_norm,
        deterministic=deterministic,
    )
  raise ValueError(f'Unsupported model: {config.model}.')


def create_optimizer(
    config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:
  if config.optimizer == 'adam':
    return optax.adam(learning_rate=config.learning_rate)
  if config.optimizer == 'sgd':
    return optax.sgd(
        learning_rate=config.learning_rate, momentum=config.momentum
    )
  raise ValueError(f'Unsupported optimizer: {config.optimizer}.')


def binary_cross_entropy_with_mask(
    *, logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray
):
  assert logits.shape == labels.shape == mask.shape
  assert len(logits.shape) == 2

  labels = jnp.where(mask, labels, -1)

  positive_logits = logits >= 0
  relu_logits = jnp.where(positive_logits, logits, 0)
  abs_logits = jnp.where(positive_logits, logits, -logits)
  return relu_logits - (logits * labels) + (jnp.log(1 + jnp.exp(-abs_logits)))


def predictions_match_labels(
    *, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs
) -> jnp.ndarray:
  del kwargs
  preds = logits > 0
  return (preds == labels).astype(jnp.float32)


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))


def get_predicted_logits(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: dict[str, jnp.ndarray] | None,
) -> jnp.ndarray:
  pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
  logits = pred_graphs.globals
  return logits


def get_valid_mask(
    labels: jnp.ndarray, graphs: jraph.GraphsTuple
) -> jnp.ndarray:
  labels_mask = ~jnp.isnan(labels)
  graph_mask = jraph.get_graph_padding_mask(graphs)
  return labels_mask & graph_mask[:, None]


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  accuracy: metrics.Average.from_fun(predictions_match_labels)
  loss: metrics.Average.from_output('loss')


@jax.jit
def ogbg_train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rngs: dict[str, jnp.ndarray],
) -> tuple[train_state.TrainState, metrics.Collection]:

  def loss_fn(params, graphs):
    curr_state = state.replace(params=params)

    labels = graphs.globals

    graphs = replace_globals(graphs)

    logits = get_predicted_logits(curr_state, graphs, rngs)
    mask = get_valid_mask(labels, graphs)
    loss = binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask
    )
    mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

    return mean_loss, (loss, logits, labels, mask)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss, logits, labels, mask)), grads = grad_fn(state.params, graphs)
  state = state.apply_gradients(grads=grads)

  metrics_update = TrainMetrics.single_from_model_output(
      loss=loss, logits=logits, labels=labels, mask=mask
  )
  return state, metrics_update


def get_fake_graphs(config: ml_collections.ConfigDict) -> jraph.GraphsTuple:
  rng = jax.random.key(0)
  n_graphs = config.batch_size
  n_total_nodes = n_graphs * 20
  n_total_edges = n_graphs * 40

  nodes = jax.random.normal(rng, (n_total_nodes, 9))
  edges = jax.random.normal(rng, (n_total_edges, 3))
  senders = jax.random.randint(rng, (n_total_edges,), 0, n_total_nodes)
  receivers = jax.random.randint(rng, (n_total_edges,), 0, n_total_nodes)
  n_node = jnp.full((n_graphs,), 20, dtype=jnp.int32)
  n_edge = jnp.full((n_graphs,), 40, dtype=jnp.int32)
  globals_ = jax.random.bernoulli(
      rng, shape=(n_graphs, config.num_classes)
  ).astype(jnp.float32)

  return jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      senders=senders,
      receivers=receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=globals_,
  )


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  rng = jax.random.key(0)
  rng, init_rng, dropout_rng = jax.random.split(rng, 3)

  graphs = get_fake_graphs(config)

  init_net = create_model(config, deterministic=True)
  init_graphs = replace_globals(graphs)
  params = jax.jit(init_net.init)(init_rng, init_graphs)

  tx = create_optimizer(config)

  net = create_model(config, deterministic=False)
  state = train_state.TrainState.create(
      apply_fn=net.apply, params=params, tx=tx
  )

  return (
      ogbg_train_step,
      (state, graphs, {'dropout': dropout_rng}),
      {},
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ogbg_molpcba_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, ogbg_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ogbg_molpcba_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, ogbg_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()
