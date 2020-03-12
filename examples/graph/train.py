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

"""Social graph example.

This script trains a simple graph neural network (GNN) for semi-supervised
node classification on Zachary's karate club (Wayne W. Zachary, "An Information
Flow Model for Conflict and Fission in Small Groups," Journal of Anthropological
Research, 1977).

Zachary's karate club is often used as a "hello world" example for network
analysis. The graph describes the social network of members of a university
karate club, where an undirected edge is present if two members interact
frequently outside of club activities. The club famously split into two parts
due to a conflict between the president of the club (John A.) and the part-time
karate instructor (Mr. Hi).

This example is adapted from https://arxiv.org/abs/1609.02907 (Appendix A). We
classify nodes based on the student-teacher assignment (John A. or Mr. Hi) in
a semi-supervised setting. During training, only the labels for John A.'s and
Mr. Hi's node are provided, while all other club members are unlabeled.
"""

from absl import app

from flax import nn
from flax import optim

from models import GraphConvBlock

import jax
from jax import random
import jax.numpy as jnp


class GNN(nn.Module):
  """Simple graph neural network (GNN) module.

  This example uses the graph convolutional network (GCN) architecture
  proposed in https://arxiv.org/abs/1609.02907. Try replacing `GraphConvBlock`
  with `MessagePassingBlock` for a more expressive GNN architecture (which,
  however, is more likely to overfit in this example).
  """

  def apply(self, node_x, edge_x, sources, targets):
    """Computes GNN forward pass.

    Args:
      node_x: node features with shape of `[num_nodes, num_features]`.
      edge_x: `None` or edge features with shape of `[num_edges, num_features]`.
      sources: Array of source node indices with shape of `[num_edges]`.
      targets: Array of target node indices with shape of `[num_edges]`.

    Returns:
      Output of shape `[num_nodes, num_features]`.
    """

    node_x = GraphConvBlock(node_x, edge_x, sources, targets, features=32)
    node_x = nn.relu(node_x)

    node_x = GraphConvBlock(node_x, edge_x, sources, targets, features=2)
    node_x = nn.log_softmax(node_x)

    return node_x


def get_karate_club_data():
  """Get Zachary's karate club social network.

  Social network of karate club members, obtained from: Wayne W. Zachary, "An
  Information Flow Model for Conflict and Fission in Small Groups," Journal
  of Anthropological Research, 1977.

  We use a sparse representation of the graph (as a directed edge list).
  Note that using dense representations (e.g., as an adjacency matrix) can
  sometimes allow for faster message passing operations, depending on the
  size/sparsity of the graph and the hardware used (GPU/TPU). This sparse
  representation, however, allows for a simple batching procedure: to create
  a batch of multiple graphs, concatenate their node/edge features and edge
  lists, while adding offsets to the node indices (to avoid duplicate indices).

  Returns:
    A tuple containing node features, node labels and edge indices.
  """

  # Edge list of Zachary's karate club.
  edge_list = [
      (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
      (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
      (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
      (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
      (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
      (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33),
      (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33),
      (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
      (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33),
      (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32),
      (31, 33), (32, 33)
  ]

  # Add inverted edges to make graph undirected.
  edge_list += [(target, source) for source, target in edge_list]

  # Extract arrays of source and target nodes.
  sources = jnp.array([source for source, target in edge_list])
  targets = jnp.array([target for source, target in edge_list])

  # Student-teacher assignment (before split) as in Zachary (1977).
  # Part-time karate instructor: Mr. Hi, node 0 (labeled as 0).
  # President: John A., node 33 (labeled as 1).
  node_labels = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                           0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  node_feats = jnp.eye(len(node_labels))  # Unique one-hot features.

  return node_feats, node_labels, sources, targets


def semi_supervised_cross_entropy_loss(logits):
  # Only use labels of first (instructor) and last (president) nodes.
  return -(logits[0, 0] + logits[-1, 1])


def compute_accuracy(logits, labels):
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return accuracy


@jax.jit
def train_step(optimizer, node_feats, sources, targets):
  def loss_fn(model):
    logits = model(node_feats, None, sources, targets)
    loss = semi_supervised_cross_entropy_loss(logits)
    return loss
  loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


@jax.jit
def eval_step(model, node_feats, sources, targets, node_labels):
  logits = model(node_feats, None, sources, targets)
  accuracy = compute_accuracy(logits, node_labels)
  return accuracy


def train():
  """Run main training loop."""
  rng = random.PRNGKey(0)

  # Get Zachary's karate club graph dataset.
  node_feats, node_labels, sources, targets = get_karate_club_data()

  # Create model and optimizer.
  _, initial_params = GNN.init(
      rng, node_x=node_feats, edge_x=None, sources=sources, targets=targets)
  model = nn.Model(GNN, initial_params)
  optimizer = optim.Adam(learning_rate=0.01).create(model)

  # Train for 20 iterations.
  for iteration in range(20):
    optimizer, loss = train_step(optimizer, node_feats, sources, targets)

    accuracy = eval_step(  # Model is stored in `optimizer.target`.
        optimizer.target, node_feats, sources, targets, node_labels)

    print('iteration: %d, loss: %.4f, accuracy: %.2f'
          % (iteration+1, loss, accuracy * 100))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train()

if __name__ == '__main__':
  app.run(main)
