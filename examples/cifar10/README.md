# CIFAR-10 examples for Flax

Adapted from the Flax ImageNet example by Geoff French.

We provide various experiments you can run to re-create Wide ResNet results
found in the literature. We also provide a PyramidNet implementation,
although it does not achieve published results.


## How to run

A full description of the command line arguments available is given later.
We will now give some examples for re-creating results from various papers.


### Wide ResNet

See https://arxiv.org/abs/1605.07146

We provide a Wide ResNet architecture, implemented in `models/wideresnet.py`.
A wide ResNet with 26 layers and 10x width is chosen by passing the value
`wrn26_10` as the `arch` command line argument. We use the default training
schedule; we train for 200 epochs and drop the learning rate by a factor of
0.2 at epochs 60, 120 and 160. Training on 8 TPUv2 cores took 1 hour 10
minutes. We got 3.87% error, the paper reports 3.89%:

```
> python train.py --arch=wrn26_10
```


### Wide ResNet with shake-shake regularization

See https://arxiv.org/abs/1705.07485

We provide a Wide ResNet architecture with shake-shake. It is implemented in
`models/wideresnet_shakeshake.py`. Pass `wrn26_6_ss` as the `arch` command line
argument to use it.

To re-create the results in the paper, we apply cosine annealing to the
learning rate and run for 1800 epochs. Training on 8 TPUv2 cores took 7 hours
57 minutes. We got 2.7% error, the paper reports 2.86%:

```
> python train.py --arch=wrn26_6_ss --lr_schedule=cosine --num_epochs=1800
```

Alternatively you can use a standard Wide ResNet learning rate schedule
described above. Training on 8 TPUv2 cores took 53 minutes. We got
3.29% error:

```
> python train.py --arch=wrn26_6_ss
```


### Pyramid Net with ShakeDrop regularization

PyramidNet: https://arxiv.org/abs/1610.02915
ShakeDrop regularization: https://arxiv.org/abs/1802.02375

We provide a Pyramid net architecture with shake-drop. It is implemented in
`models/pyramidnet.py`. Pass `pyramid` as the `arch` command line
argument to use it.

Unfortunately we do not match the results in the literature.

To re-create the experimental conditions in the ShakeDrop paper, use a value
of 0.0001 for L2-regularization, train for 300 epochs and drop the learning
rate by a factor of 0.1 at epochs 150 and 225. Training on 8 TPUv2 cores
took 12 hours. We got 3.26% error, the PyramidNet paper reports 3.31%, and
the ShakeDrop paper reports 3.08% for this configuration.

```
> python train.py --arch=pyramid --lr_sched_steps="[[150,0.1],[225,0.01]]" \
    --num_epochs=300 --l2_reg=0.0001
```

Alternatively you can apply cosine annealing to the learning rate and run for
1800 epochs. Training on 8 TPUv2 cores took 3 days 26 minutes. We got 2.64%
error. The Fast AutoAugment code repo reports 2.7% error
(https://github.com/kakaobrain/fast-autoaugment).

```
> python train.py --arch=pyramid --lr_schedule=cosine --num_epochs=1800 \
    --l2_reg=0.0001
```


## Command line arguments

The network architecture and training regime and be controlled via command
line arguments:

`--learning_rate` (default=0.1): set the initial learning rate
`--momentum` (default=0.9): the momentum value used for SGD
`--lr_schedule`: Choose the learning rate schedule:
    `constant`: a constant learning rate
    `stepped` (default): a stepped learning rate that
        changes the learning rate at specific points during training
    `cosine`: anneal the learning rate with half a cosine wave
`--lr_sched_steps` (default=`[[60, 0.2], [120, 0.04], [160, 0.008]]`):
    Define the steps used for a stepped learning rate the steps are specified
    using Python syntax; e.g. to drop the LR by a factor of 0.1 at epochs 60,
    120 and 160 use:
    `--lr_sched_steps="[[60, 0.1], [120, 0.01], [160, 0.001]]"`
`--num_epochs` (default=200): the number of epochs to train for
`--l2_reg` (default=0.005): the amount of L2 regularization to apply
    (on all parameters)
`--batch_size` (default=256): mini-batch size
`--arch` (default=wrn26_10): network architecture
    `wrn26_10`: Wide ResNet, 26 layers, 10x width
    `wrn26_6_ss`: Wide ResNeXt, 26 2x96d with shake-shake regularization
    `pyramid`: PyramidNet, pyramid alpha=200, 272 layers, ShakeDrop
        regularization
`--wrn_dropout_rate` (default=0.3): DropOut rate used in Wide ResNet
    (on all residual blocks)
`--rng` (default=0): Random seed used for network initialization and
    stochasticity during training


## Issues

We apply L2 regularization to weights and biases, where as it is mostly *not*
applied to biases.


## References

Various code was used as a reference to help replicate baseline results.
In particular the Fast AutoAugment codebase was particularly helpful
as its config files describe training regimes for the wide ResNet
with shake-shake and PyramidNet baselines.

Shake-shake author's implementation:
https://github.com/xgastaldi/shake-shake

PyTorch implementation of shake-shake:
https://github.com/owruby/shake-shake_pytorch

Fast AutoAugment that uses a number of different models:
https://github.com/kakaobrain/fast-autoaugment
