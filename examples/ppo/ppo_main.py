from absl import flags
from absl import app
import jax
import jax.random

import ppo_lib
import models
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

flags.DEFINE_string(
  'game', default='Pong',
  help=('The Atari game used.')
)

flags.DEFINE_string(
  'logdir', default='/tmp/ppo_training',
  help=('Directory to set .')
)

def main(argv):
  game = "Pong"
  game += "NoFrameskip-v4"
  num_actions = env_utils.get_num_actions(game)
  print(f"Playing {game} with {num_actions} actions")
  num_agents = FLAGS.num_agents
  total_frames = 40000000
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model = models.create_model(subkey, num_outputs=num_actions)
  optimizer = models.create_optimizer(model, learning_rate=FLAGS.learning_rate)
  del model
  optimizer = ppo_lib.train(optimizer, game, total_frames, FLAGS)

if __name__ == '__main__':
  app.run(main)
