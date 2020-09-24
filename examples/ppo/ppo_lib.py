"""Library file which executes the PPO training"""

import functools
from typing import Tuple, List
from absl import flags
import jax
import jax.random
import jax.numpy as jnp
import numpy as onp
import flax

import agent
import remote
import test_episodes

@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
@jax.jit
def gae_advantages(rewards, terminal_masks, values, discount, gae_param):
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
  assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed; "
    "Eq. (12) in PPO paper requires V(s_{t+1}) to calculate delta_t")
  advantages, gae = [], 0.
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

@functools.partial(jax.jit, static_argnums=(6))
def train_step(
  optimizer: flax.optim.base.Optimizer,
  trajectories: Tuple[onp.array, onp.array, onp.array, onp.array, onp.array],
  clip_param: float,
  vf_coeff: float,
  entropy_coeff: float,
  lr: float,
  batch_size: int):
  """Compilable train step.

  Runs an entire epoch of training (i.e. the loop over
  minibatches within an epoch is included here for performance reasons).

  Args:
    optimizer: optimizer for the actor-critic model
    trajectories: Tuple of the following five elements forming the experience:
                  states: shape (steps_per_agent*num_agents, 84, 84, 4)
                  actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                  old_log_probs: shape (steps_per_agent*num_agents, )
                  returns: shape (steps_per_agent*num_agents, )
                  advantages: (steps_per_agent*num_agents, )
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss
    lr: learning rate, varies between optimization steps
        if decaying_lr_and_clip_param is set to true
    batch_size: the minibatch size, static argument

  Returns:
    optimizer: new optimizer after the parameters update
    loss: loss summed over training steps
    grad_norm: gradient norm from last step (summed over parameters)
  """
  def loss_fn(model, minibatch, clip_param, vf_coeff, entropy_coeff):
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = model(states)
    values = values[:, 0]  # convert shapes: (batch, 1) to (batch, )
    probs = jnp.exp(log_probs)
    entropy = jnp.sum(-probs*log_probs, axis=1).mean()
    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    PG_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                               1. + clip_param)
    PPO_loss = -jnp.mean(jnp.minimum(PG_loss, clipped_loss), axis=0)
    value_loss = jnp.mean(jnp.square(returns - values), axis=0)
    return PPO_loss + vf_coeff*value_loss - entropy_coeff*entropy

  iterations = trajectories[0].shape[0] // batch_size
  trajectories = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories)
  loss = 0.
  for batch in zip(*trajectories):
    grad_fn = jax.value_and_grad(loss_fn)
    l, grad = grad_fn(optimizer.target, batch, clip_param, vf_coeff,
                      entropy_coeff)
    loss += l
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    grad_norm = sum(jnp.square(g).sum() for g in jax.tree_leaves(grad))
  return optimizer, loss, grad_norm

def get_experience(
  model: flax.optim.base.Optimizer,
  simulators: List[remote.RemoteSimulator],
  steps_per_actor: int):
  """Collect experience from agents.

  Runs `steps_per_actor` time steps of the game for each of the `simulators`.
  """
  all_experience = []
  # Range up to steps_per_actor + 1 to get one more value needed for GAE.
  for _ in range(steps_per_actor + 1):
    states = []
    for sim in simulators:
      state = sim.conn.recv()
      states.append(state)
    states = onp.concatenate(states, axis=0)
    log_probs, values = agent.policy_action(model, states)
    log_probs, values = jax.device_get((log_probs, values))
    probs = onp.exp(onp.array(log_probs))
    for i, sim in enumerate(simulators):
      probabilities = probs[i]
      action = onp.random.choice(probs.shape[1], p=probabilities)
      # In principle, one could avoid sending value and log prob back and forth.
      sim.conn.send((action, values[i, 0], log_probs[i][action]))
    experiences = []
    for sim in simulators:
      sample = sim.conn.recv()
      experiences.append(sample)
    all_experience.append(experiences)
  return all_experience

def process_experience(
  experience: List[List[remote.ExpTuple]],
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
  states = onp.zeros(exp_dims + obs_shape, dtype=onp.float32)
  actions = onp.zeros(exp_dims, dtype=onp.int32)
  rewards = onp.zeros(exp_dims, dtype=onp.float32)
  values = onp.zeros(values_dims, dtype=onp.float32)
  log_probs = onp.zeros(exp_dims, dtype=onp.float32)
  dones = onp.zeros(exp_dims, dtype=onp.float32)

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
  trajectories = tuple(map(
    lambda x: onp.reshape(
      x, (num_agents * actor_steps,) + x.shape[2:]),
      trajectories))
  return trajectories

def train(
  optimizer: flax.optim.base.Optimizer,
  game: str,
  steps_total: int,
  num_agents: int,
  flags_: flags._flagvalues.FlagValues):
  """Main training loop.

  Args:
    optimizer: optimizer for the actor-critic model
    game: string specifying the Atari game from gym package
    steps total: total number of frames (env steps) to train on
    num_agents: number of separate processes with agents running the envs

  Returns:
    optimizer: the trained optimizer
  """
  simulators = [remote.RemoteSimulator(game) for i in range(num_agents)]
  loop_steps = steps_total // (num_agents * flags_.actor_steps)
  for s in range(loop_steps):
    # Bookkeeping and testing.
    print(f"\n training loop step {s}")

    if (s + 1) % (20000 // (num_agents * flags_.actor_steps)) == 0:
      test_episodes.policy_test(1, optimizer.target, game)

    if flags_.decaying_lr_and_clip_param:
      alpha = 1. - s/loop_steps
    else:
      alpha = 1.

    # Core training code.
    all_experiences = get_experience(optimizer.target, simulators,
                                     flags_.actor_steps)
    trajectories = process_experience(
      all_experiences, flags_.actor_steps, flags_.num_agents, flags_.gamma,
      flags_.lambda_)
    lr = flags_.learning_rate * alpha
    clip_param = flags_.clip_param * alpha
    for e in range(flags_.num_epochs):
      permutation = onp.random.permutation(num_agents * flags_.actor_steps)
      trajectories = tuple(map(lambda x: x[permutation], trajectories))
      optimizer, loss, last_iter_grad_norm = train_step(
        optimizer, trajectories, clip_param, flags_.vf_coeff,
        flags_.entropy_coeff, lr, flags_.batch_size)
      print(f"epoch {e} loss {loss} grad norm {last_iter_grad_norm}")
  return optimizer