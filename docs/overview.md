# Overview

## Background: JAX

[JAX](https://github.com/google/jax) is NumPy + autodiff + GPU/TPU

It allows for fast scientific computing and machine learning
with the normal NumPy API
(+ additional APIs for special accelerator ops when needed)

JAX comes with powerful primitives, which you can compose arbitrarily:

* Autodiff (`jax.grad`): Efficient any-order gradients w.r.t any variables
* JIT compilation (`jax.jit`): Trace any function ⟶ fused accelerator ops
* Vectorization (`jax.vmap`): Automatically batch code written for individual samples
* Parallelization (`jax.pmap`): Automatically parallelize code across multiple accelerators (including across hosts, e.g. for TPU pods)

If you don't know JAX but just want to learn what you need to use Flax, you can check our [JAX for the impatient](notebooks/jax_for_the_impatient) notebook.

## Flax

[Flax](https://github.com/google/flax) is a high-performance neural network library for
JAX that is **designed for flexibility**:
Try new forms of training by forking an example and by modifying the training
loop, not by adding features to a framework.

Flax is being developed in close collaboration with the JAX team and 
comes with everything you need to start your research, including:

* **Neural network API** (`flax.linen`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* **Optimizers** (`flax.optim`): SGD, Momentum, Adam, LARS, Adagrad, LAMB, RMSprop

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Code Examples

See the [What does Flax look like](https://github.com/google/flax#what-does-flax-look-like) section of our README.


## TPU support

We currently have a [LM1b/Wikitext-2 language model with a Transformer architecture](https://colab.research.google.com/github/google/flax/blob/main/examples/lm1b/Colab_Language_Model.ipynb)
that's been tuned. You can run it directly via Colab.

At present, Cloud TPUs are network-attached, and Flax users typically feed in data from one or more additional VMs

When working with large-scale input data, it is important to create large enough VMs with sufficient network bandwidth to avoid having the TPUs bottlenecked waiting for input

TODO: Add an example for running on Google Cloud.
