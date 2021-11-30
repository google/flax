# Flax Examples

## Core examples

The examples from this directory.

Each example is designed to be **self-contained and easily forkable**, while
reproducing relevant results in different areas of machine learning.

As discussed in [#231], we decided to go for a standard pattern for all examples
including the simplest ones (like MNIST). This makes every example a bit more
verbose, but once you know one example, you know the structure of all of them.
Having unit tests and integration tests is also very useful when you fork these
examples.

Some of the examples below have a link "🕹Interactive🕹" that lets you run them
directly in Colab.

Image classification

- [MNIST](https://github.com/google/flax/tree/main/examples/mnist/) -
  [🕹Interactive🕹](https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb):
  Convolutional neural network for MNIST classification (featuring simple
  code).

- [ImageNet](https://github.com/google/flax/tree/main/examples/imagenet/) -
  [🕹Interactive🕹](https://colab.research.google.com/github/google/flax/blob/main/examples/imagenet/imagenet.ipynb):
  Resnet-50 on ImageNet with weight decay (featuring multi host SPMD, custom
  preprocessing, checkpointing, dynamic scaling, mixed precision).

Reinforcement Learning

- [Proximal Policy Optimization](https://github.com/google/flax/tree/main/examples/ppo/):
  Learning to play Atari games (featuring single host SPMD, RL setup).

Natural language processing

-  [Sequence to sequence for number
   addition](https://github.com/google/flax/tree/main/examples/seq2seq/):
   (featuring simple code, LSTM state handling, on the fly data generation).
-  [Parts-of-speech
   tagging](https://github.com/google/flax/tree/main/examples/nlp_seq/): Simple
   transformer encoder model using the universal dependency dataset.
-  [Sentiment
   classification](https://github.com/google/flax/tree/main/examples/sst2/):
   with a LSTM model.
-  [Transformer encoder/decoder model trained on
   WMT](https://github.com/google/flax/tree/main/examples/wmt/):
   Translating English/German (featuring multihost SPMD, dynamic bucketing,
   attention cache, packed sequences, recipe for TPU training on GCP).
-  [Transformer encoder trained on one billion word
   benchmark](https://github.com/google/flax/tree/main/examples/lm1b/):
   for autoregressive language modeling, based on the WMT example above.

Generative models

-  [Variational
   auto-encoder](https://github.com/google/flax/tree/main/examples/vae/):
   Trained on binarized MNIST (featuring simple code, vmap).
-  [PixelCNN++](https://github.com/google/flax/tree/main/examples/pixelcnn/):
   Trained on cifar10 (featuring single host SPMD, checkpointing, Polyak decay).

[#231]: https://github.com/google/flax/issues/231

## Community Examples

In addition to the curated list of official Flax examples, there is a growing
community of people using Flax to build new types of machine learning models. We
are happy to showcase any example built by the community here! If you want to
submit your own example, we suggest that you start by forking one of the
official Flax example, and start from there.

|             Link             |       Author       |             Task type             |                               Reference                               |
| ---------------------------- | ------------------ | --------------------------------- | --------------------------------------------------------------------- |
| [matthias-wright/flaxmodels] | [@matthias-wright] | Various                           | Various                                                               |
| [google/vision_transformer]  | [@andsteing]       | Image classification, fine-tuning | https://arxiv.org/abs/2010.11929 and https://arxiv.org/abs/2105.01601 |
| [JAX-RL]                     | [@henry-prior]     | Reinforcement learning            | N/A                                                                   |
| [DCGAN] Colab                | [@bkkaggle]        | Image Synthesis                   | https://arxiv.org/abs/1511.06434                                      |
| [BigBird Fine-tuning]        | [@vasudevgupta7]   | Question-Answering                | https://arxiv.org/abs/2007.14062                                      |
| [jax-resnet]                 | [@n2cholas]        | Various resnet implementations    | `torch.hub`                                                           |

[matthias-wright/flaxmodels]: https://github.com/matthias-wright/flaxmodels
[google/vision_transformer]: https://github.com/google-research/vision_transformer
[JAX-RL]: https://github.com/henry-prior/jax-rl
[DCGAN]: https://github.com/bkkaggle/jax-dcgan
[BigBird Fine-tuning]: https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/big_bird
[jax-resnet]: https://github.com/n2cholas/jax-resnet
[@matthias-wright]: https://github.com/matthias-wright
[@andsteing]: https://github.com/andsteing
[@henry-prior]: https://github.com/henry-prior
[@bkkaggle]: https://github.com/bkkaggle
[@vasudevgupta7]: https://github.com/vasudevgupta7
[@n2cholas]: https://github.com/n2cholas