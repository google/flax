# Copyright 2024 The Flax Authors.
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

"""Definition of the GNN model."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

from flax import linen as nn
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jraph


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
  """Adds the nodes, edges and global features from other_graphs to graphs."""
  return graphs._replace(
      nodes=graphs.nodes + other_graphs.nodes,
      edges=graphs.edges + other_graphs.edges,
      globals=graphs.globals + other_graphs.globals,
  )


def _layer_norm_kernel(
    x_ref,       # Input block [block_rows, dim]
    gamma_ref,   # Scale parameter [dim]
    beta_ref,    # Bias parameter [dim]
    o_ref,       # Output block [block_rows, dim]
    *,
    eps: float,
):
  """Fused LayerNorm: single read, compute mean/var/norm, single write."""
  x = x_ref[...].astype(jnp.float32)
  gamma = gamma_ref[...].astype(jnp.float32)
  beta = beta_ref[...].astype(jnp.float32)

  mean = jnp.mean(x, axis=-1, keepdims=True)
  diff = x - mean
  var = jnp.mean(diff * diff, axis=-1, keepdims=True)
  rstd = jax.lax.rsqrt(var + eps)

  o_ref[...] = (diff * rstd * gamma + beta).astype(o_ref.dtype)


def _fallback_layer_norm(x, gamma, beta, eps):
  """Fallback LayerNorm implementation for VJP backward pass calculation."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  diff = x - mean
  var = jnp.mean(diff * diff, axis=-1, keepdims=True)
  rstd = jax.lax.rsqrt(var + eps)
  return diff * rstd * gamma + beta


def _fused_layer_norm_impl(x, gamma, beta, eps, block_size, interpret):
  """Underlying implementation of Pallas fused LayerNorm with dynamic padding."""
  original_shape = x.shape
  dim = x.shape[-1]
  x_2d = x.reshape(-1, dim)
  num_rows = x_2d.shape[0]

  if num_rows == 0:
    return x

  # Ensure block_size aligns with TPU tile requirements (multiple of 8)
  block_size = min(block_size, max(num_rows, 8))
  if block_size % 8 != 0:
    block_size = block_size + (8 - block_size % 8)

  pad_rows = (block_size - num_rows % block_size) % block_size
  if pad_rows > 0:
    x_2d = jnp.pad(x_2d, ((0, pad_rows), (0, 0)))

  num_blocks = x_2d.shape[0] // block_size

  kernel = functools.partial(_layer_norm_kernel, eps=eps)

  result = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(x_2d.shape, x.dtype),
      grid=(num_blocks,),
      in_specs=[
          pl.BlockSpec((block_size, dim), lambda i: (i, 0)),
          pl.BlockSpec((dim,), lambda i: (0,)),
          pl.BlockSpec((dim,), lambda i: (0,)),
      ],
      out_specs=pl.BlockSpec((block_size, dim), lambda i: (i, 0)),
      interpret=interpret,
  )(x_2d, gamma, beta)

  if pad_rows > 0:
    result = result[:num_rows, :]

  return result.reshape(original_shape)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _fused_layer_norm_fwd_bwd(x, gamma, beta, eps, block_size, interpret):
  """Custom VJP decorated forward/backward definition for fused LayerNorm."""
  return _fused_layer_norm_impl(x, gamma, beta, eps, block_size, interpret)


def _fwd(x, gamma, beta, eps, block_size, interpret):
  out = _fused_layer_norm_impl(x, gamma, beta, eps, block_size, interpret)
  return out, (x, gamma, beta)


def _bwd(eps, block_size, interpret, res, g):
  del block_size, interpret
  x, gamma, beta = res
  fn = lambda x_val, g_val, b_val: _fallback_layer_norm(
      x_val, g_val, b_val, eps
  )
  _, vjp_fn = jax.vjp(fn, x, gamma, beta)
  return vjp_fn(g)


_fused_layer_norm_fwd_bwd.defvjp(_fwd, _bwd)


def fused_layer_norm(
    x: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    *,
    eps: float = 1e-5,
    block_size: int = 128,
    interpret: bool = False,
) -> jax.Array:
  """Fused LayerNorm wrapper with dynamic padding and custom VJP."""
  return _fused_layer_norm_fwd_bwd(x, gamma, beta, eps, block_size, interpret)


class PallasLayerNorm(nn.Module):
  """Flax Module Drop-in for LayerNorm using Pallas."""

  epsilon: float = 1e-5
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32
  block_size: int = 128
  interpret: Any = None  # Dynamically set based on JAX backend if None

  @nn.compact
  def __call__(self, x: Any):
    x = jnp.asarray(x, self.dtype)
    dim = x.shape[-1]  # pytype: disable=attribute-error  # jax-ndarray
    scale = self.param('scale', nn.initializers.ones, (dim,), self.param_dtype)
    bias = self.param('bias', nn.initializers.zeros, (dim,), self.param_dtype)
    if self.interpret is not None:
      interpret = self.interpret
    else:
      interpret = jax.default_backend() == 'cpu'
    return fused_layer_norm(
        x,
        scale,
        bias,
        eps=self.epsilon,
        block_size=self.block_size,
        interpret=interpret,
    )  # pytype: disable=wrong-arg-types


class MLP(nn.Module):
  """A multi-layer perceptron."""

  feature_sizes: Sequence[int]
  dropout_rate: float = 0
  deterministic: bool = True
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for size in self.feature_sizes:
      x = nn.Dense(features=size)(x)
      x = self.activation(x)
      x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(
          x
      )
    return x


class GraphNet(nn.Module):
  """A complete Graph Network model defined with Jraph."""

  latent_size: int
  num_mlp_layers: int
  message_passing_steps: int
  output_globals_size: int
  dropout_rate: float = 0
  skip_connections: bool = True
  use_edge_model: bool = True
  layer_norm: bool = True
  deterministic: bool = True

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # We will first linearly project the original features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.latent_size),
        embed_edge_fn=nn.Dense(self.latent_size),
        embed_global_fn=nn.Dense(self.latent_size),
    )
    processed_graphs = embedder(graphs)

    # Now, we will apply a Graph Network once for each message-passing round.
    mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
    for _ in range(self.message_passing_steps):
      if self.use_edge_model:
        update_edge_fn = jraph.concatenated_args(
            MLP(
                mlp_feature_sizes,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
            )
        )
      else:
        update_edge_fn = None

      update_node_fn = jraph.concatenated_args(
          MLP(
              mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic,
          )
      )
      update_global_fn = jraph.concatenated_args(
          MLP(
              mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic,
          )
      )

      graph_net = jraph.GraphNetwork(
          update_node_fn=update_node_fn,
          update_edge_fn=update_edge_fn,
          update_global_fn=update_global_fn,
      )

      if self.skip_connections:
        processed_graphs = add_graphs_tuples(
            graph_net(processed_graphs), processed_graphs
        )
      else:
        processed_graphs = graph_net(processed_graphs)

      if self.layer_norm:
        processed_graphs = processed_graphs._replace(
            nodes=PallasLayerNorm()(processed_graphs.nodes),
            edges=PallasLayerNorm()(processed_graphs.edges),
            globals=PallasLayerNorm()(processed_graphs.globals),
        )

    # Since our graph-level predictions will be at globals, we will
    # decode to get the required output logits.
    decoder = jraph.GraphMapFeatures(
        embed_global_fn=nn.Dense(self.output_globals_size)
    )
    processed_graphs = decoder(processed_graphs)

    return processed_graphs


class GraphConvNet(nn.Module):
  """A Graph Convolution Network + Pooling model defined with Jraph."""

  latent_size: int
  num_mlp_layers: int
  message_passing_steps: int
  output_globals_size: int
  dropout_rate: float = 0
  skip_connections: bool = True
  layer_norm: bool = True
  deterministic: bool = True
  pooling_fn: Callable[
      [jnp.ndarray, jnp.ndarray, jnp.ndarray],  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      jnp.ndarray,
  ] = jraph.segment_mean

  def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pooling operation, taken from Jraph."""

    # Equivalent to jnp.sum(n_node), but JIT-able.
    sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
    # To aggregate nodes from each graph to global features,
    # we first construct tensors that map the node to the corresponding graph.
    # Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
    n_graph = graphs.n_node.shape[0]
    node_graph_indices = jnp.repeat(
        jnp.arange(n_graph),
        graphs.n_node,
        axis=0,
        total_repeat_length=sum_n_node,
    )
    # We use the aggregation function to pool the nodes per graph.
    pooled = self.pooling_fn(graphs.nodes, node_graph_indices, n_graph)  # pytype: disable=wrong-arg-types  # jax-ndarray
    return graphs._replace(globals=pooled)

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # We will first linearly project the original node features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.latent_size))
    processed_graphs = embedder(graphs)

    # Now, we will apply the GCN once for each message-passing round.
    for _ in range(self.message_passing_steps):
      mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
      update_node_fn = jraph.concatenated_args(
          MLP(
              mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic,
          )
      )
      graph_conv = jraph.GraphConvolution(
          update_node_fn=update_node_fn, add_self_edges=True
      )

      if self.skip_connections:
        processed_graphs = add_graphs_tuples(
            graph_conv(processed_graphs), processed_graphs
        )
      else:
        processed_graphs = graph_conv(processed_graphs)

      if self.layer_norm:
        processed_graphs = processed_graphs._replace(
            nodes=PallasLayerNorm()(processed_graphs.nodes),
        )

    # We apply the pooling operation to get a 'global' embedding.
    processed_graphs = self.pool(processed_graphs)

    # Now, we decode this to get the required output logits.
    decoder = jraph.GraphMapFeatures(
        embed_global_fn=nn.Dense(self.output_globals_size)
    )
    processed_graphs = decoder(processed_graphs)

    return processed_graphs
