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

"""Library file which executes the PPO training."""

import functools
from typing import Any, Callable, Tuple, List

from absl import logging
import flax
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.random
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

import agent
import models
import test_episodes


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float):
  """Use Generalized Advantage Estimation (GAE) to compute advantages.

  As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
  key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

  Args:
    rewards: array shaped (actor_steps, num_agents), rewards from the game
    terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                    and ones for non-terminal states
    values: array shaped (actor_steps, num_agents), values estimated by critic
    discount: RL discount usually denoted with gamma
    gae_param: GAE parameter usually denoted with lambda

  Returns:
    advantages: calculated advantages shaped (actor_steps, num_agents)
  """
  assert rewards.shape[0] + 1 == values.shape[0], ('One more value needed; Eq. '
                                                   '(12) in PPO paper requires '
                                                   'V(s_{t+1}) for delta_t')
  advantages = []
  gae = 0.
  for t in reversed(range(len(rewards))):
    # Masks used to set next state value to 0 for terminal states.
    value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
    delta = rewards[t] + value_diff
    # Masks[t] used to ensure that values before and after a terminal state
    # are independent of each other.
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    advantages.append(gae)
  advantages = advantages[::-1]
  return jnp.array(advantages)


def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    minibatch: Tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float):
  """Evaluate the loss function.

  Compute loss as a sum of three components: the negative of the PPO clipped
  surrogate objective, the value function loss and the negative of the entropy
  bonus.

  Args:
    params: the parameters of the actor-critic model
    apply_fn: the actor-critic model's apply function
    minibatch: Tuple of five elements forming one experience batch:
               states: shape (batch_size, 84, 84, 4)
               actions: shape (batch_size, 84, 84, 4)
               old_log_probs: shape (batch_size,)
               returns: shape (batch_size,)
               advantages: shape (batch_size,)
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss

  Returns:
    loss: the PPO loss, scalar quantity
  """
  states, actions, old_log_probs, returns, advantages = minibatch
  log_probs, values = agent.policy_action(apply_fn, params, states)
  values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
  probs = jnp.exp(log_probs)

  value_loss = jnp.mean(jnp.square(returns - values), axis=0)

  entropy = jnp.sum(-probs*log_probs, axis=1).mean()

  log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
  ratios = jnp.exp(log_probs_act_taken - old_log_probs)
  # Advantage normalization (following the OpenAI baselines).
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  pg_loss = ratios * advantages
  clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                            1. + clip_param)
  ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

  return ppo_loss + vf_coeff*value_loss - entropy_coeff*entropy

@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    trajectories: Tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float):
  """Compilable train step.

  Runs an entire epoch of training (i.e. the loop over minibatches within
  an epoch is included here for performance reasons).

  Args:
    state: the train state
    trajectories: Tuple of the following five elements forming the experience:
                  states: shape (steps_per_agent*num_agents, 84, 84, 4)
                  actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                  old_log_probs: shape (steps_per_agent*num_agents, )
                  returns: shape (steps_per_agent*num_agents, )
                  advantages: (steps_per_agent*num_agents, )
    batch_size: the minibatch size, static argument
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss

  Returns:
    optimizer: new optimizer after the parameters update
    loss: loss summed over training steps
  """
  iterations = trajectories[0].shape[0] // batch_size
  trajectories = jax.tree_util.tree_map(
      lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories)
  loss = 0.
  for batch in zip(*trajectories):
    grad_fn = jax.value_and_grad(loss_fn)
    l, grads = grad_fn(state.params, state.apply_fn, batch, clip_param, vf_coeff,
                      entropy_coeff)
    loss += l
    state = state.apply_gradients(grads=grads)
  return state, loss

def get_experience(
    state: train_state.TrainState,
    simulators: List[agent.RemoteSimulator],
    steps_per_actor: int):
  """Collect experience from agents.

  Runs `steps_per_actor` time steps of the game for each of the `simulators`.
  """
  all_experience = []
  # Range up to steps_per_actor + 1 to get one more value needed for GAE.
  for _ in range(steps_per_actor + 1):
    sim_states = []
    for sim in simulators:
      sim_state = sim.conn.recv()
      sim_states.append(sim_state)
    sim_states = np.concatenate(sim_states, axis=0)
    log_probs, values = agent.policy_action(state.apply_fn, state.params, sim_states)
    log_probs, values = jax.device_get((log_probs, values))
    probs = np.exp(np.array(log_probs))
    for i, sim in enumerate(simulators):
      probabilities = probs[i]
      action = np.random.choice(probs.shape[1], p=probabilities)
      sim.conn.send(action)
    experiences = []
    for i, sim in enumerate(simulators):
      sim_state, action, reward, done = sim.conn.recv()
      value = values[i, 0]
      log_prob = log_probs[i][action]
      sample = agent.ExpTuple(sim_state, action, reward, value, log_prob, done)
      experiences.append(sample)
    all_experience.append(experiences)
  return all_experience

def process_experience(
    experience: List[List[agent.ExpTuple]],
    actor_steps: int,
    num_agents: int,
    gamma: float,
    lambda_: float):
  """Process experience for training, including advantage estimation.

  Args:
    experience: collected from agents in the form of nested lists/namedtuple
    actor_steps: number of steps each agent has completed
    num_agents: number of agents that collected experience
    gamma: dicount parameter
    lambda_: GAE parameter

  Returns:
    trajectories: trajectories readily accessible for `train_step()` function
  """
  obs_shape = (84, 84, 4)
  exp_dims = (actor_steps, num_agents)
  values_dims = (actor_steps + 1, num_agents)
  states = np.zeros(exp_dims + obs_shape, dtype=np.float32)
  actions = np.zeros(exp_dims, dtype=np.int32)
  rewards = np.zeros(exp_dims, dtype=np.float32)
  values = np.zeros(values_dims, dtype=np.float32)
  log_probs = np.zeros(exp_dims, dtype=np.float32)
  dones = np.zeros(exp_dims, dtype=np.float32)

  for t in range(len(experience) - 1):  # experience[-1] only for next_values
    for agent_id, exp_agent in enumerate(experience[t]):
      states[t, agent_id, ...] = exp_agent.state
      actions[t, agent_id] = exp_agent.action
      rewards[t, agent_id] = exp_agent.reward
      values[t, agent_id] = exp_agent.value
      log_probs[t, agent_id] = exp_agent.log_prob
      # Dones need to be 0 for terminal states.
      dones[t, agent_id] = float(not exp_agent.done)
  for a in range(num_agents):
    values[-1, a] = experience[-1][a].value
  advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
  returns = advantages + values[:-1, :]
  # After preprocessing, concatenate data from all agents.
  trajectories = (states, actions, log_probs, returns, advantages)
  trajectory_len = num_agents * actor_steps
  trajectories = tuple(map(
      lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))
  return trajectories


@functools.partial(jax.jit, static_argnums=1)
def get_initial_params(key: np.ndarray, model: nn.Module):
  input_dims = (1, 84, 84, 4)  # (minibatch, height, width, stacked frames)
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)['params']
  return initial_params


def create_train_state(params, model: nn.Module,
                       config: ml_collections.ConfigDict, train_steps: int) -> train_state.TrainState:
  if config.decaying_lr_and_clip_param:
    lr = optax.linear_schedule(
        init_value=config.learning_rate, end_value=0.,
        transition_steps=train_steps)
  else:
    lr = config.learning_rate
  tx = optax.adam(lr)
  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx)
  return state


def train(
    model: models.ActorCritic,
    config: ml_collections.ConfigDict,
    model_dir: str):
  """Main training loop.

  Args:
    model: the actor-critic model
    config: object holding hyperparameters and the training information
    model_dir: path to dictionary where checkpoints and logging info are stored

  Returns:
    optimizer: the trained optimizer
  """
  
  game = config.game + 'NoFrameskip-v4'
  simulators = [agent.RemoteSimulator(game)
                for _ in range(config.num_agents)]
  summary_writer = tensorboard.SummaryWriter(model_dir)
  summary_writer.hparams(dict(config))
  loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
  log_frequency = 40
  checkpoint_frequency = 500
  # train_step does multiple steps per call for better performance
  # compute number of steps per call here to convert between the number of
  # train steps and the inner number of optimizer steps
  iterations_per_step = (config.num_agents * config.actor_steps
      // config.batch_size)

  initial_params = get_initial_params(jax.random.PRNGKey(0), model)
  state = create_train_state(initial_params, model, config,
                             loop_steps * config.num_epochs * iterations_per_step)
  del initial_params
  state = checkpoints.restore_checkpoint(model_dir, state)
  # number of train iterations done by each train_step

  start_step = int(state.step) // config.num_epochs // iterations_per_step
  logging.info('Start training from step: %s', start_step)

  for step in range(start_step, loop_steps):
    # Bookkeeping and testing.
    if step % log_frequency == 0:
      score = test_episodes.policy_test(1, state.apply_fn, state.params, game)
      frames = step * config.num_agents * config.actor_steps
      summary_writer.scalar('game_score', score, frames)
      logging.info('Step %s:\nframes seen %s\nscore %s\n\n', step, frames, score)

    # Core training code.
    alpha = 1. - step / loop_steps if config.decaying_lr_and_clip_param else 1.
    all_experiences = get_experience(
        state, simulators, config.actor_steps)
    trajectories = process_experience(
        all_experiences, config.actor_steps, config.num_agents, config.gamma,
        config.lambda_)
    clip_param = config.clip_param * alpha
    for _ in range(config.num_epochs):
      permutation = np.random.permutation(
          config.num_agents * config.actor_steps)
      trajectories = tuple(x[permutation] for x in trajectories)
      state, _ = train_step(
          state, trajectories, config.batch_size,
          clip_param=clip_param,
          vf_coeff=config.vf_coeff,
          entropy_coeff=config.entropy_coeff)
    if (step + 1) % checkpoint_frequency == 0:
      checkpoints.save_checkpoint(model_dir, state, step + 1)
  return state
