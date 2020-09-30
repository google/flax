# Proximal Policy Optimization

Uses the Proximal Policy Optimization algorithm ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))
to learn playing Atari games.

## Requirements

This example depends on the `gym`, `opencv-python` and `atari-py` packages
in addition to `jax` and `flax`.

## Supported setups

The example should run with other configurations and hardware, but was explicitly
tested on the following:

| Hardware | Game | Training time | Total frames seen | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| 1x V100 GPU  | Qbert  |  9h 27m 8s | 40M | [2020-09-30](https://tensorboard.dev/experiment/1pacpbxxRz2di3NIOFkHoA/#scalars) |

## How to run

Running `python ppo_main.py` will run the example with default
(hyper)parameters, i.e. for 40M frames on the Pong game.

By default logging info and checkpoints will be stored in `/tmp/ppo_training`
directory. This can be overriden as follows:

```python ppo_main.py --logdir=/my_fav_directory```

You can also override the default (hyper)parameters, for example

```python ppo_main.py --config.game=Seaquest --config.total_frames=20000000 --config.decaying_lr_and_clip_param=False --logdir=/tmp/seaquest```

will train the model on 20M Seaquest frames with constant (i.e. not linearly
decaying) learning rate and PPO clipping parameter. Checkpoints and tensorboard
files will be saved in `/tmp/seaquest`.

Unit tests can be run using `python ppo_lib_test.py`.

## How to run on Google Cloud TPU

It is also possible to run this code on Google Cloud TPU. For detailed
instructions on the required setup, please refer to the [WMT example readme](https://github.com/google/flax/tree/master/examples/wmt).

## Owners

Jonathan Heek @jheek, Wojciech Rzadkowski @wrzadkow