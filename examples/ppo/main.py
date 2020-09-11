import jax
import jax.random
import jax.numpy as jnp
import numpy as onp
import flax
import time
from typing import Tuple, List
from queue import Queue
import threading


from models import create_model, create_optimizer
from agent import policy_action
from remote import RemoteSimulator
from test_episodes import test

# @jax.jit
def gae_advantages(rewards, terminal_masks, values, discount, gae_param):
  """Use Generalized Advantage Estimation (GAE) to compute advantages
  Eqs. (11-12) in PPO paper arXiv: 1707.06347"""
  assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed; "
          "Eq. (12) in PPO paper requires V(s_{t+1}) to calculate \delta_t")
  return_values, gae = [], 0
  for t in reversed(range(len(rewards))):
    #masks to set next state value to 0 for terminal states
    value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
    delta = rewards[t] + value_diff
    # masks[t] to ensure that values before and after a terminal state
    # are independent of each other
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    return_values.insert(0, gae + values[t])
  return onp.array(return_values) #jnp after vectorization

# @jax.jit
def train_step(optimizer, trn_data, clip_param, vf_coeff, entropy_coeff,
                batch_size):
  def loss_fn(model, minibatch, clip_param, vf_coeff, entropy_coeff):
    states, actions, old_log_probs, returns, advantages = minibatch
    probs, values = model(states)
    log_probs = jnp.log(probs)
    entropy = jnp.sum(-probs*log_probs, axis=1).mean()
    # from all probs from the forward pass, we need to choose ones
    # corresponding to actually taken actions
    # log_probs_act_taken = log_probs[jnp.arange(probs.shape[0]), actions])
    # above hits "Indexing mode not yet supported."
    log_probs_act_taken = jnp.log(jnp.array(
                        [probs[i, actions[i]]for i in range(actions.shape[0])]))
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    PG_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                               1. + clip_param)
    PPO_loss = -jnp.mean(jnp.minimum(PG_loss, clipped_loss))
    value_loss = jnp.mean(jnp.square(returns - values))
    return PPO_loss + vf_coeff*value_loss - entropy_coeff*entropy

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
  return optimizer, loss


def thread_inference(
  q1 : Queue,
  q2: Queue,
  simulators : List[RemoteSimulator],
  steps_per_actor : int):
  """Worker function for a separate thread used for inference and running
  the simulators in order to maximize the GPU/TPU usage. Runs
  `steps_per_actor` time steps of the game for each of the `simulators`."""

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
      # print(f"states type {type(states)}")
      probs, values = policy_action(optimizer.target, states)

      probs = onp.array(probs)
      # print("probs after onp conversion", probs)

      for i, sim in enumerate(simulators):
        # probs[i] should sum up to 1, but there are float round errors
        # if using jnp.array directly, it required division by probs[i].sum()
        # better solutions can be thought of
        # issue might be a result of the network using jnp.int 32, , not 64
        probabilities = probs[i] # / probs[i].sum()
        action = onp.random.choice(probs.shape[1], p=probabilities)
        #in principle, one could avoid sending value and log prob back and forth
        sim.conn.send((action, values[i], onp.log(probs[i][action])))

      # get experience from simulators
      experiences = []
      for sim in simulators:
        sample = sim.conn.recv()
        experiences.append(sample)
      all_experience.append(experiences)

    q2.put(all_experience)


def train(
  optimizer : flax.optim.base.Optimizer,
  # target_model : nn.base.Model,
  steps_total : int, # maybe rename to frames_total
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
    print(f"training loop step {s}")
    #bookkeeping and testing
    if (s + 1) % (10000 // (num_agents*STEPS_PER_ACTOR)) == 0:
      print(f"Frames processed {s*num_agents*STEPS_PER_ACTOR}" +
            f"time elapsed {time.time()-t1}")
      t1 = time.time()
    if (s + 1) % (50000 // (num_agents*STEPS_PER_ACTOR)) == 0:
      test(1, optimizer.target, render=False)


    # send the up-to-date policy model and current step to inference thread
    step = s*num_agents
    q1.put((optimizer, step))

    # perform PPO training
    # experience is a list of list of tuples, here we preprocess this data to
    # get required input for GAE and then for training
    # initial version, needs improvement in terms of speed & readability
    if s > 0: #avoid training when there's no data yet
      obs_shape = (84, 84, 4)
      states = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS) + obs_shape,
                          dtype=onp.float32)
      actions = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS), dtype=onp.int32)
      rewards = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS), dtype=onp.float32)
      values = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS), dtype=onp.float32)
      log_probs = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS),
                              dtype=onp.float32)
      dones = onp.zeros((STEPS_PER_ACTOR + 1, NUM_AGENTS), dtype=onp.float32)

      # experiences state, action, reward, value, log_prob, done)
      for time_step, exp in enumerate(all_experiences):
        for agent_id, exp_agent in enumerate(exp):
          states[time_step, agent_id, ...] = exp_agent[0]
          actions[time_step, agent_id] = exp_agent[1]
          rewards[time_step, agent_id] =exp_agent[2]
          values[time_step, agent_id] = exp_agent[3]
          log_probs[time_step, agent_id] = exp_agent[4]
          # dones need to be 0 for terminal states
          dones[time_step, agent_id] = float(not exp_agent[5])

      #calculate returns using GAE (needs to be vectorized instead of foor loop)
      returns = onp.zeros((STEPS_PER_ACTOR, NUM_AGENTS))
      for i in range(NUM_AGENTS):
        returns[:, i] = gae_advantages(rewards[:-1, i], dones[:-1, i],
                          values[:, i], DISCOUNT, GAE_PARAM)
      advantages = returns - values[:-1, :]

      #getting rid of unnecessary data (one more value was needed for GAE)
      states = states[:-1, ...].copy()
      actions = actions[:-1, ...].copy()
      log_probs = log_probs[:-1, ...].copy()
      # after all the preprocessing, we discard the information
      # about from which agent the data comes by reshaping
      trn_data = (states, actions, log_probs, returns, advantages)
      trn_data = tuple(map(
        lambda x: onp.reshape(x,
         (NUM_AGENTS * STEPS_PER_ACTOR , ) + x.shape[2:]), trn_data)
      )
      for _ in range(NUM_EPOCHS): #possibly compile this loop inside a jit
        permutation = onp.random.permutation(NUM_AGENTS * STEPS_PER_ACTOR)
        trn_data = tuple(map(lambda x: x[permutation], trn_data))
        optimizer, _ = train_step(optimizer, trn_data, CLIP_PARAM, VF_COEFF,
                              ENTROPY_COEFF, BATCH_SIZE)
    #end of PPO training

    #collect new data from the inference thread
    all_experiences = q2.get()

  return None


STEPS_PER_ACTOR = 128
NUM_AGENTS = 8
NUM_EPOCHS = 3
BATCH_SIZE = 32 * 8

DISCOUNT = 0.99 #usually denoted with \gamma
GAE_PARAM = 0.95 #usually denoted with \lambda

VF_COEFF = 1 #weighs value function loss in total loss
ENTROPY_COEFF = 0.01 # weighs entropy bonus in the total loss

LR = 2.5e-4

CLIP_PARAM = 0.1

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
model = create_model(subkey)
optimizer = create_optimizer(model, learning_rate=LR)
del model

def main():
  num_agents = NUM_AGENTS
  total_frames = 4000000
  train_device = jax.devices()[0]
  inference_device = jax.devices()[1]
  jax.device_put(optimizer.target, device=train_device)
  train(optimizer, total_frames, num_agents, train_device, inference_device)

if __name__ == '__main__':
  main()
