import jax
import jax.random
import jax.numpy as jnp
import numpy as onp
import flax
import time
from functools import partial
from typing import Tuple, List
from queue import Queue
import threading


from models import create_model, create_optimizer
from agent import policy_action
from remote import RemoteSimulator
from test_episodes import test

@partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
@jax.jit
def gae_advantages(rewards, terminal_masks, values, discount, gae_param):
  """Use Generalized Advantage Estimation (GAE) to compute advantages
  Eqs. (11-12) in PPO paper arXiv: 1707.06347.
  Uses key observation that A_{t} = \delta_t + \gamma*\lambda*A_{t+1}.
  """
  assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed; "
          "Eq. (12) in PPO paper requires V(s_{t+1}) to calculate \delta_t")
  advantages, gae = [], 0.
  for t in reversed(range(len(rewards))):
    # masks to set next state value to 0 for terminal states
    value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
    delta = rewards[t] + value_diff
    # masks[t] to ensure that values before and after a terminal state
    # are independent of each other
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    advantages = [gae] + advantages
  return jnp.array(advantages)

@jax.jit
def train_step(optimizer, trn_data, clip_param, vf_coeff, entropy_coeff):
  def loss_fn(model, minibatch, clip_param, vf_coeff, entropy_coeff):
    states, actions, old_log_probs, returns, advantages = minibatch
    shapes = list(map(lambda x : x.shape, minibatch))
    assert(shapes[0] == (BATCH_SIZE, 84, 84, 4))
    assert(all(s == (BATCH_SIZE,) for s in shapes[1:]))
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

  batch_size = BATCH_SIZE
  iterations = trn_data[0].shape[0] // batch_size
  trn_data = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trn_data)
  loss = 0.
  for batch in zip(*trn_data):
    grad_fn = jax.value_and_grad(loss_fn)
    l, grad = grad_fn(optimizer.target, batch, clip_param, vf_coeff,
    entropy_coeff)
    loss += l
    optimizer = optimizer.apply_gradient(grad)
    grad_norm = sum(jnp.square(g).sum() for g in jax.tree_leaves(grad))
  return optimizer, loss, grad_norm


def thread_inference(
  q1 : Queue,
  q2: Queue,
  simulators : List[RemoteSimulator],
  steps_per_actor : int):
  """Worker function for a separate thread used for inference and running
  the simulators in order to maximize the GPU/TPU usage. Runs
  `steps_per_actor` time steps of the game for each of the `simulators`.
  """

  while(True):
    optimizer, step = q1.get()
    all_experience = []
    for _ in range(steps_per_actor + 1): # +1 to get one more value
                                         # needed for GAE
      states = []
      for sim in simulators:
        state = sim.conn.recv()
        states.append(state)
      states = onp.concatenate(states, axis=0)

      # perform inference
      # policy_optimizer, step = q1.get()
      log_probs, values = policy_action(optimizer.target, states)

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

    q2.put(all_experience)


def train(
  optimizer : flax.optim.base.Optimizer,
  steps_total : int,
  num_agents : int,
  train_device,
  inference_device):

  simulators = [RemoteSimulator() for i in range(num_agents)]
  q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
  inference_thread = threading.Thread(target=thread_inference,
                        args=(q1, q2, simulators, STEPS_PER_ACTOR), daemon=True)
  inference_thread.start()
  t1 = time.time()

  for s in range(steps_total // num_agents):
    print(f"\n training loop step {s}")
    #bookkeeping and testing
    if (s + 1) % (10000 // (num_agents*STEPS_PER_ACTOR)) == 0:
      print(f"      Frames processed {s*num_agents*STEPS_PER_ACTOR}, " +
            f"time elapsed {time.time()-t1}")
      t1 = time.time()
    if (s + 1) % (20000 // (num_agents*STEPS_PER_ACTOR)) == 0:
      test(1, optimizer.target, render=False)


    # send the up-to-date policy model and current step to inference thread
    step = s*num_agents
    q1.put((optimizer, step))

    # perform PPO training
    # experience is a list of list of tuples, here we preprocess this data to
    # get required input for GAE and then for training
    # initial version, needs improvement in terms of speed & readability
    all_experiences = q2.get()
    if s >= 0: #avoid training when there's no data yet
      obs_shape = (84, 84, 4)
      states = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS) + obs_shape,
                          dtype=onp.float32)
      actions = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS), dtype=onp.int32)
      rewards = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS), dtype=onp.float32)
      values = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS), dtype=onp.float32)
      log_probs = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS),
                              dtype=onp.float32)
      dones = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS), dtype=onp.float32)

      assert(len(all_experiences) == STEPS_PER_ACTOR + 1)
      assert(len(all_experiences[0]) == NUM_AGENTS)
      for t in range(len(all_experiences) - 1): #last only for next_values
        for agent_id, exp_agent in enumerate(all_experiences[t]):
          states[t, agent_id, ...] = exp_agent[0]
          actions[t, agent_id] = exp_agent[1]
          rewards[t, agent_id] =exp_agent[2]
          values[t, agent_id] = exp_agent[3]
          log_probs[t, agent_id] = exp_agent[4]
          # dones need to be 0 for terminal states
          dones[t, agent_id] = float(not exp_agent[5])
      for a in range(num_agents):
        values[-1, a] = all_experiences[-1][a][3]
      # calculate advantages w. GAE
      advantages = gae_advantages(rewards, dones, values, DISCOUNT, GAE_PARAM)
      returns = advantages + values[:-1, :]
      assert(returns.shape == advantages.shape == (STEPS_PER_ACTOR, NUM_AGENTS))
      # after preprocessing, concatenate data from all agents
      trn_data = (states, actions, log_probs, returns, advantages)
      trn_data = tuple(map(
        lambda x: onp.reshape(x,
         (NUM_AGENTS * STEPS_PER_ACTOR , ) + x.shape[2:]), trn_data)
      )
      print(f"Step {s}: rewards variance {rewards.var()}")
      dr = dones.ravel()
      print(f"fraction of terminal states {1.-(dr.sum()/dr.shape[0])}")
      for e in range(NUM_EPOCHS): #possibly compile this loop inside a jit
        shapes = list(map(lambda x : x.shape, trn_data))
        assert(shapes[0] == (NUM_AGENTS * STEPS_PER_ACTOR, 84, 84, 4))
        assert(all(s == (NUM_AGENTS * STEPS_PER_ACTOR,) for s in shapes[1:]))
        permutation = onp.random.permutation(NUM_AGENTS * STEPS_PER_ACTOR)
        trn_data = tuple(map(lambda x: x[permutation], trn_data))
        optimizer, loss, last_iter_grad_norm = train_step(optimizer, trn_data,
            CLIP_PARAM, VF_COEFF, ENTROPY_COEFF)
        print(f"Step {s} epoch {e} loss {loss} grad norm {last_iter_grad_norm}")
    #end of PPO training

  return None

# PPO paper and openAI baselines 2
# https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
STEPS_PER_ACTOR = 128
NUM_AGENTS = 8
NUM_EPOCHS = 3
BATCH_SIZE = 32 * 8

DISCOUNT = 0.99 #usually denoted with \gamma
GAE_PARAM = 0.95 #usually denoted with \lambda

VF_COEFF = 0.5 #weighs value function loss in total loss
ENTROPY_COEFF = 0.01 # weighs entropy bonus in the total loss

LR = 2.5e-4

CLIP_PARAM = 0.1

# openAI baselines 1
# https://github.com/openai/baselines/blob/master/baselines/ppo1/run_atari.py
# STEPS_PER_ACTOR = 256
# NUM_AGENTS = 8
# NUM_EPOCHS = 4
# BATCH_SIZE = 64

# DISCOUNT = 0.99 #usually denoted with \gamma
# GAE_PARAM = 0.95 #usually denoted with \lambda

# VF_COEFF = 1. #weighs value function loss in total loss
# ENTROPY_COEFF = 0.01 # weighs entropy bonus in the total loss

# LR = 1e-3

# CLIP_PARAM = 0.2



def main():
  num_agents = NUM_AGENTS
  total_frames = 4000000
  train_device = jax.devices()[0]
  inference_device = jax.devices()[1]
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model = create_model(subkey)
  optimizer = create_optimizer(model, learning_rate=LR)
  del model
  # jax.device_put(optimizer.target, device=train_device)
  train(optimizer, total_frames, num_agents, train_device, inference_device)

if __name__ == '__main__':
  main()
