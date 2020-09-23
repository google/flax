import jax
import jax.random
import jax.numpy as jnp
import numpy as onp
import flax
import time
import functools
import queue
from absl import flags
from absl import app
import threading
from typing import Tuple, List

import models
import agent
import remote
import test_episodes
import env_utils

FLAGS = flags.FLAGS

# default hyperparameters taken from PPO paper and openAI baselines 2
# https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py

flags.DEFINE_float(
  'learning_rate', default=2.5e-4,
  help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_integer(
  'batch_size', default=256,
  help=('Batch size for training.')
)

flags.DEFINE_integer(
  'num_agents', default=8,
  help=('Number of agents playing in parallel.')
)

flags.DEFINE_integer(
  'actor_steps', default=128,
  help=('Batch size for training.')
)

flags.DEFINE_integer(
  'num_epochs', default=3,
  help=('Number of epochs per each unroll of the policy.')
)

flags.DEFINE_float(
  'gamma', default=0.99,
  help=('Discount parameter.')
)

flags.DEFINE_float(
  'lambda_', default=0.95,
  help=('Generalized Advantage Estimation parameter.')
)

flags.DEFINE_float(
  'clip_param', default=0.1,
  help=('The PPO clipping parameter used to clamp ratios in loss function.')
)

flags.DEFINE_float(
  'vf_coeff', default=0.5,
  help=('Weighs value function loss in the total loss.')
)

flags.DEFINE_float(
  'entropy_coeff', default=0.01,
  help=('Weighs entropy bonus in the total loss.')
)

flags.DEFINE_boolean(
  'decaying_lr_and_clip_param', default=True,
  help=(('Linearly decay learning rate and clipping parameter to zero during '
          'the training.'))
)

@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
@jax.jit
def gae_advantages(rewards, terminal_masks, values, discount, gae_param):
  """Use Generalized Advantage Estimation (GAE) to compute advantages.

  As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
  key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.
  """
  assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed; "
          "Eq. (12) in PPO paper requires V(s_{t+1}) to calculate delta_t")
  advantages, gae = [], 0.
  for t in reversed(range(len(rewards))):
    # masks to set next state value to 0 for terminal states
    value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
    delta = rewards[t] + value_diff
    # masks[t] to ensure that values before and after a terminal state
    # are independent of each other
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    advantages.append(gae)
  advantages = advantages[::-1]
  return jnp.array(advantages)

@jax.jit
def train_step(optimizer, trn_data, clip_param, vf_coeff, entropy_coeff, lr):
  """Compilable train step.

  Runs an entire epoch of training (i.e. the loop over
  minibatches within an epoch is included here for performance reasons).

  Args:
    optimizer: optimizer for the actor-critic model
    trn_data: Tuple of the following five elements forming the experience:
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

  Returns:
    optimizer: new optimizer after the parameters update
    loss: loss summed over training steps
    grad_norm: gradient norm from last step (summed over parameters)
  """
  def loss_fn(model, minibatch, clip_param, vf_coeff, entropy_coeff):
    states, actions, old_log_probs, returns, advantages = minibatch
    shapes = list(map(lambda x : x.shape, minibatch))
    log_probs, values = model(states)
    values = values[:, 0] # convert shapes: (batch, 1) to (batch, )
    probs = jnp.exp(log_probs)
    entropy = jnp.sum(-probs*log_probs, axis=1).mean()
    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # adv. normalization (following the OpenAI baselines)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    PG_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                               1. + clip_param)
    assert(PG_loss.shape == clipped_loss.shape)
    PPO_loss = -jnp.mean(jnp.minimum(PG_loss, clipped_loss), axis=0)
    assert(values.shape == returns.shape)
    value_loss = jnp.mean(jnp.square(returns - values), axis=0)
    return PPO_loss + vf_coeff*value_loss - entropy_coeff*entropy

  batch_size = FLAGS.batch_size
  iterations = trn_data[0].shape[0] // batch_size
  trn_data = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trn_data)
  loss = 0.
  for batch in zip(*trn_data):
    grad_fn = jax.value_and_grad(loss_fn)
    l, grad = grad_fn(optimizer.target, batch, clip_param, vf_coeff,
                      entropy_coeff)
    loss += l
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    grad_norm = sum(jnp.square(g).sum() for g in jax.tree_leaves(grad))
  return optimizer, loss, grad_norm


def thread_inference(
  policy_q: queue.Queue,
  experience_q: queue.Queue,
  simulators: List[remote.RemoteSimulator],
  steps_per_actor: int):
  """Worker function for a separate inference thread.

  Runs `steps_per_actor` time steps of the game for each of the `simulators`.
  """
  while(True):
    optimizer, step = policy_q.get()
    all_experience = []
    for _ in range(steps_per_actor + 1): # +1 to get one more value
                                         # needed for GAE
      states = []
      for sim in simulators:
        state = sim.conn.recv()
        states.append(state)
      states = onp.concatenate(states, axis=0)

      # perform inference
      # policy_optimizer, step = policy_q.get()
      log_probs, values = agent.policy_action(optimizer.target, states)
      log_probs, values = jax.device_get((log_probs, values))

      probs = onp.exp(onp.array(log_probs))
      # print("probs after onp conversion", probs)

      for i, sim in enumerate(simulators):
        # probs[i] should sum up to 1, but there are float round errors
        # if using jnp.array directly, it required division by probs[i].sum()
        # better solutions can be thought of
        # issue might be a result of the network using jnp.int 32, , not 64
        probabilities = probs[i] # / probs[i].sum()
        action = onp.random.choice(probs.shape[1], p=probabilities)
        #in principle, one could avoid sending value and log prob back and forth
        sim.conn.send((action, values[i, 0], log_probs[i][action]))

      # get experience from simulators
      experiences = []
      for sim in simulators:
        sample = sim.conn.recv()
        experiences.append(sample)
      all_experience.append(experiences)

    experience_q.put(all_experience)


def train(
  optimizer: flax.optim.base.Optimizer,
  game: str,
  steps_total: int,
  num_agents: int,
  train_device,
  inference_device):
  """Main training loop.

  Args:
    optimizer: optimizer for the actor-critic model
    game: string specifying the Atari game from Gym package
    steps total: total number of frames (env steps) to train on
    num_agents: number of separate processes with agents running the envs
    train_device : device used for training
    inference_device :  device used for inference

  Returns:
    None
  """
  simulators = [remote.RemoteSimulator(game) for i in range(num_agents)]
  policy_q = queue.Queue(maxsize=1)
  experience_q = queue.Queue(maxsize=1)
  inference_thread = threading.Thread(
    target=thread_inference,
    args=(policy_q, experience_q, simulators, FLAGS.actor_steps),
    daemon=True)
  inference_thread.start()
  t1 = time.time()
  loop_steps = steps_total // (num_agents * FLAGS.actor_steps)
  for s in range(loop_steps):
    print(f"\n training loop step {s}")
    #bookkeeping and testing
    if (s + 1) % (10000 // (num_agents * FLAGS.actor_steps)) == 0:
      print(f"      Frames processed {s * num_agents * FLAGS.actor_steps}, " +
            f"time elapsed {time.time() - t1}")
      t1 = time.time()
    if (s + 1) % (20000 // (num_agents * FLAGS.actor_steps)) == 0:
      test_episodes.policy_test(1, optimizer.target, game)

    if FLAGS.decaying_lr_and_clip_param:
      alpha = 1. - s/loop_steps
    else:
      alpha = 1.
    # send the up-to-date policy model and current step to inference thread
    step = s*num_agents
    policy_q.put((optimizer, step))

    # perform PPO training
    # experience is a list of list of tuples, here we preprocess this data to
    # get required input for GAE and then for training
    # initial version, needs improvement in terms of speed & readability
    all_experiences = experience_q.get()
    if s >= 0: #avoid training when there's no data yet
      obs_shape = (84, 84, 4)
      exp_dims = (FLAGS.actor_steps, FLAGS.num_agents)
      values_dims = (FLAGS.actor_steps + 1, FLAGS.num_agents)
      states = onp.zeros(exp_dims + obs_shape, dtype=onp.float32)
      actions = onp.zeros(exp_dims, dtype=onp.int32)
      rewards = onp.zeros(exp_dims, dtype=onp.float32)
      values = onp.zeros(values_dims, dtype=onp.float32)
      log_probs = onp.zeros(exp_dims, dtype=onp.float32)
      dones = onp.zeros(exp_dims, dtype=onp.float32)

      assert(len(all_experiences[0]) == FLAGS.num_agents)
      for t in range(len(all_experiences) - 1): #last only for next_values
        for agent_id, exp_agent in enumerate(all_experiences[t]):
          states[t, agent_id, ...] = exp_agent.state
          actions[t, agent_id] = exp_agent.action
          rewards[t, agent_id] = exp_agent.reward
          values[t, agent_id] = exp_agent.value
          log_probs[t, agent_id] = exp_agent.log_prob
          # dones need to be 0 for terminal states
          dones[t, agent_id] = float(not exp_agent.done)
      for a in range(num_agents):
        values[-1, a] = all_experiences[-1][a].value
      # calculate advantages w. GAE
      advantages = gae_advantages(rewards, dones, values,
                                  FLAGS.gamma, FLAGS.lambda_)
      returns = advantages + values[:-1, :]
      # after preprocessing, concatenate data from all agents
      trn_data = (states, actions, log_probs, returns, advantages)
      trn_data = tuple(map(
        lambda x: onp.reshape(
          x, (FLAGS.num_agents * FLAGS.actor_steps, ) + x.shape[2:]), trn_data))
      print(f"Step {s}: rewards variance {rewards.var()}")
      lr = FLAGS.learning_rate * alpha
      clip_param = FLAGS.clip_param * alpha
      for e in range(FLAGS.num_epochs): #possibly compile this loop inside a jit
        shapes = list(map(lambda x : x.shape, trn_data))
        permutation = onp.random.permutation(num_agents * FLAGS.actor_steps)
        trn_data = tuple(map(lambda x: x[permutation], trn_data))
        optimizer, loss, last_iter_grad_norm = train_step(optimizer, trn_data,
          clip_param, FLAGS.vf_coeff, FLAGS.entropy_coeff, lr)
        print(f"epoch {e} loss {loss} grad norm {last_iter_grad_norm}")
    #end of PPO training

  return None

def main(argv):
  game = "Pong"
  game += "NoFrameskip-v4"
  num_actions = env_utils.get_num_actions(game)
  print(f"Playing {game} with {num_actions} actions")
  num_agents = FLAGS.num_agents
  total_frames = 40000000
  train_device = jax.devices()[0]
  inference_device = jax.devices()[1]
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model = models.create_model(subkey, num_outputs=num_actions)
  optimizer = models.create_optimizer(model, learning_rate=FLAGS.learning_rate)
  del model
  # jax.device_put(optimizer.target, device=train_device)
  train(optimizer, game, total_frames, num_agents, train_device,
        inference_device)

if __name__ == '__main__':
  app.run(main)
