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

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Code Examples

See the [What does Flax look like](https://github.com/google/flax#what-does-flax-look-like) section of our README.


## TPU support

All of our examples should run on TPU. See the following docs for more instructions:

* [Launching jobs on Google Cloud](https://github.com/google/flax/tree/main/examples/cloud): provides a simple script that can be used to create a new VM on Google Cloud, train an example on that VM and then shutting it down.
* [Flax Examples](https://github.com/google/flax/tree/main/examples): Some of our examples requiring GPU/TPU support have instructions on how to run them on these devices (see `imagenet` and `wmt`).
* [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm): A brief introduction to working with JAX and Cloud TPU.
