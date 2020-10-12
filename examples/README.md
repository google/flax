# Flax Examples
 
## Official examples maintained by the Flax core team
 
This folder contains a collection of examples of implementations of various architectures and training procedures.
 
Each examples is designed to be self-contained and easily forkable, while reproducing "key results" in different areas of machine learning. Official Flax examples come with significant maintenance expectations, including (but not limited to since guidelines are WIP - [#231](https://github.com/google/flax/issues/231)):
* Tested benchmarks on single-GPU, multi-GPU, and TPU configurations (e.g. [imagenet](imagenet/))
* Unit tests (e.g. that one training step succeeds)
* Adherence to Flax best practices and code updates as those best practices evolve
* An "owner" who is a member of the Flax core team who keeps the example up-to-date
* A minimal implementation of a single model on a single dataset (users can fork and change the configuration to run on another dataset)

| Example | Description | Features |
| ------- | ----------- | -------- |
| [graph](graph/README.md) | Graph convolutional network to label nodes in small toy dataset | Data inlined, simple code |
| [imagenet](imagenet/README.md) | Resnet-50 on imagenet with weight decay | Multi host SPMD, tfds `imagenet`, custom preprocessing, checkpointing, dynamic scaling, mixed precision |
| [lm1b](lm1b/README.md) | Transformer encoder for next token prediction | Single host SPMD, tfds `lm1b`, checkpointing, dynamic bucketing, attention cache, Colab |
| [mnist](mnist/README.md) | Convolutional neural network for MNIST classification | Tfds `mnist`, simple code |
| [pixelcnn](pixelcnn/README.md) | PixelCNN++ for CIFAR-10 generation | Single host SPMD, tfds `cifar10`, checkpointing, Polyak decay |
| [seq2seq](seq2seq/README.md) | LSTM encoder/decoder for completing `42+1234=` sequences | On the fly data generation, LSTM state handling, simple code |
| [vae](vae/README.md) | Variational auto-encoder for binarized MNIST images | Tfds `binarized_mnist`, vmap, simple code |
| [wmt](wmt/README.md) | Transformer for translating en/de | Multi host SPMD, tfds `wmt1{4,7}_translate`, SentencePiece tokenization, checkpointing, dynamic bucketing, attention cache, packed sequences, recipe for TPU training on GCP |

## Flax examples from the community
 
In addition to the curated list of official Flax examples, there is a growing community of people using Flax to build new types of machine learning models. We are happy to showcase any example built by the community here! If you want to submit your own example, we suggest that you start by forking one of the official Flax example, and start from there.
 
Here are some of the models that people have built with Flax:

| Link  | Author | Task type | Reference |
| ------------- | ------------- | ------------ | ---------- |
| [Gaussian Processes regression](https://github.com/danieljtait/ladax/tree/master/examples)  | @danieljtait | Regression | N/A |  |
| [JAX-RL](https://github.com/henry-prior/jax-rl)  | @henry-prior  | Reinforcement learning | N/A |
| [DQN](https://github.com/joaogui1/RL-JAX/tree/master/DQN)  | @joaogui1  | Reinforcement learning | N/A |
| [Various CIFAR SOTA Models](https://github.com/google-research/google-research/tree/master/flax_models/cifar) | @PForet | Image Classification | N/A |
| [DCGAN](https://github.com/bkkaggle/jax-dcgan) Colab | @bkkaggle | Image Synthesis | https://arxiv.org/abs/1511.06434 |

If you are using Flax for your research, or have re-implemented interesting models in Flax, file a pull request against this README and add a link in the table.
 
## Looking for "FOO" implemented in Flax?

We use GitHub issues to keep track of which models people are most interested in seeing re-implemented in Flax. If you can't find what you're looking for in the [list](https://github.com/google/flax/labels/example%20request), file an issue with this [template](https://github.com/google/flax/issues/new?assignees=&template=example_request.md&title=).
 
If the model you are looking for has already been requested by others, upvote the issue to help us see which ones are the most requested.

## Looking to implement something in Flax?
 
Consider looking at the list of requested models in the GitHub issue tracker under the ["example requested" label](https://github.com/google/flax/labels/example%20request) and go ahead and build it!
