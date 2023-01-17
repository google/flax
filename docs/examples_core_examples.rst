Core examples
=============

Core examples are hosted on the GitHub Flax repository in the `examples <https://github.com/google/flax/tree/main/examples>`__
directory.

Each example is designed to be **self-contained and easily forkable**, while
reproducing relevant results in different areas of machine learning.

As discussed in `#231 <https://github.com/google/flax/issues/231>`__, we decided
to go for a standard pattern for all examples including the simplest ones (like MNIST).
This makes every example a bit more verbose, but once you know one example, you
know the structure of all of them. Having unit tests and integration tests is also
very useful when you fork these examples.

Some of the examples below have a link "Interactive🕹" that lets you run them
directly in Colab.

Image classification
********************

- :octicon:`mark-github;0.9em` `MNIST <https://github.com/google/flax/tree/main/examples/mnist/>`__ -
  `Interactive🕹 <https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb>`__:
  Convolutional neural network for MNIST classification (featuring simple
  code).

- :octicon:`mark-github;0.9em` `ImageNet <https://github.com/google/flax/tree/main/examples/imagenet/>`__ -
  `Interactive🕹 <https://colab.research.google.com/github/google/flax/blob/main/examples/imagenet/imagenet.ipynb>`__:
  Resnet-50 on ImageNet with weight decay (featuring multi host SPMD, custom
  preprocessing, checkpointing, dynamic scaling, mixed precision).

Reinforcement learning
**********************

- :octicon:`mark-github;0.9em` `Proximal Policy Optimization <https://github.com/google/flax/tree/main/examples/ppo/>`__:
  Learning to play Atari games (featuring single host SPMD, RL setup).

Natural language processing
***************************

-  :octicon:`mark-github;0.9em` `Sequence to sequence for number
   addition <https://github.com/google/flax/tree/main/examples/seq2seq/>`__:
   (featuring simple code, LSTM state handling, on the fly data generation).
-  :octicon:`mark-github;0.9em` `Parts-of-speech
   tagging <https://github.com/google/flax/tree/main/examples/nlp_seq/>`__: Simple
   transformer encoder model using the universal dependency dataset.
-  :octicon:`mark-github;0.9em` `Sentiment
   classification <https://github.com/google/flax/tree/main/examples/sst2/>`__:
   with a LSTM model.
-  :octicon:`mark-github;0.9em` `Transformer encoder/decoder model trained on
   WMT <https://github.com/google/flax/tree/main/examples/wmt/>`__:
   Translating English/German (featuring multihost SPMD, dynamic bucketing,
   attention cache, packed sequences, recipe for TPU training on GCP).
-  :octicon:`mark-github;0.9em` `Transformer encoder trained on one billion word
   benchmark <https://github.com/google/flax/tree/main/examples/lm1b/>`__:
   for autoregressive language modeling, based on the WMT example above.

Generative models
*****************

-  :octicon:`mark-github;0.9em` `Variational
   auto-encoder <https://github.com/google/flax/tree/main/examples/vae/>`__:
   Trained on binarized MNIST (featuring simple code, vmap).

Graph modeling
**************

- :octicon:`mark-github;0.9em` `Graph Neural Networks <https://github.com/google/flax/tree/main/examples/ogbg_molpcba/>`__:
  Molecular predictions on ogbg-molpcba from the Open Graph Benchmark.

Contributing to core Flax examples
**********************************

Most of the `core Flax examples on GitHub <https://github.com/google/flax/tree/main/examples>`__
follow a structure that the Flax dev team found works well with Flax projects.
The team strives to make these examples easy to explore and fork. In particular
(as per GitHub Issue `#231 <https://github.com/google/flax/issues/231>`__):

- README: contains links to paper, command line, `TensorBoard <https://tensorboard.dev/>`__ metrics.
- Focus: an example is about a single model/dataset.
- Configs: we use ``ml_collections.ConfigDict`` stored under ``configs/``.
- Tests: executable ``main.py`` loads ``train.py`` which has ``train_test.py``.
- Data: is read from `TensorFlow Datasets <https://www.tensorflow.org/datasets>`__.
- Standalone: every directory is self-contained.
- Requirements: versions are pinned in ``requirements.txt``.
- Boilerplate: is reduced by using `clu <https://pypi.org/project/clu/>`__.
- Interactive: the example can be explored with a `Colab <https://colab.research.google.com/>`__.