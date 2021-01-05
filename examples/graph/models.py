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

"""Graph neural network building blocks."""

from flax import nn

import jax
import jax.numpy as jnp


class GraphConvBlock(nn.Module):
  """Graph convolutional network (GCN) block.

  Similar to https://arxiv.org/abs/1609.02907 but without normalization.
  """

  def apply(self, node_x, edge_x, sources, targets, features):
    """Computes forward pass.

    Args:
      node_x: node features with shape of `[num_nodes, num_features]`.
      edge_x: `None` or edge features with shape of `[num_edges, num_features]`.
      sources: Array of source node indices with shape of `[num_edges]`.
      targets: Array of target node indices with shape of `[num_edges]`.
      features: Integer, number of features.

    Returns:
      Output of shape `[num_nodes, num_features]`.
    """
    self.features = features
    self.num_nodes = node_x.shape[0]

    # Run one round of message passing.
    message_x = self._build_messages(node_x, edge_x, sources, targets)
    aggregated_messages = self._propagate_messages(message_x, sources, targets)
    return self._update_nodes(node_x, aggregated_messages)

  def _update_nodes(self, node_x, aggregated_messages):
    # Add self-connection (concatenate) and apply Dense layer.
    node_x = jnp.concatenate((aggregated_messages, node_x), axis=1)
    return nn.Dense(node_x, features=self.features)

  def _build_messages(self, node_x, edge_x, sources, targets):
    del edge_x  # Unused.
    source_x = jnp.take(node_x, sources, axis=0)
    return source_x

  def _propagate_messages(self, message_x, sources, targets):
    del sources  # We only propagate to targets.
    aggregation_fn = jax.ops.segment_sum  # Sum aggregation.
    return aggregation_fn(message_x, targets, num_segments=self.num_nodes)


class MLP(nn.Module):
  """Simple multi-layer perceptron (MLP) module."""

  def apply(self, x, features):
    x = nn.Dense(x, features=features)
    x = nn.relu(x)
    x = nn.Dense(x, features=features)
    return x


class MessagePassingBlock(nn.Module):
  """Message passing neural network (MPNN) block.

  Similar to https://arxiv.org/abs/1704.01212.
  """

  def apply(self, node_x, edge_x, sources, targets, features):
    """Computes forward pass.

    Args:
      node_x: node features with shape of `[num_nodes, num_features]`.
      edge_x: `None` or edge features with shape of `[num_edges, num_features]`.
      sources: Array of source node indices with shape of `[num_edges]`.
      targets: Array of target node indices with shape of `[num_edges]`.
      features: Integer, number of features.

    Returns:
      Output of shape `[num_nodes, num_features]`.
    """
    self.features = features
    self.num_nodes = node_x.shape[0]

    # Run one round of message passing.
    message_x = self._build_messages(node_x, edge_x, sources, targets)
    aggregated_messages = self._propagate_messages(message_x, sources, targets)
    return self._update_nodes(node_x, aggregated_messages)

  def _update_nodes(self, node_x, aggregated_messages):
    # Add self-connection (concatenate) and apply MLP.
    node_x = jnp.concatenate((aggregated_messages, node_x), axis=1)
    return MLP(node_x, features=self.features)

  def _build_messages(self, node_x, edge_x, sources, targets):
    source_x = jnp.take(node_x, sources, axis=0)
    target_x = jnp.take(node_x, targets, axis=0)
    message_x = jnp.concatenate((source_x, target_x), axis=1)

    if edge_x is not None:  # Concatenate edge features, if available.
      message_x = jnp.concatenate((message_x, edge_x), axis=1)

    return MLP(message_x, features=self.features)  # Transform output.

  def _propagate_messages(self, message_x, sources, targets):
    del sources  # We only propagate to targets.
    aggregation_fn = jax.ops.segment_sum  # Sum aggregation.
    return aggregation_fn(message_x, targets, num_segments=self.num_nodes)
