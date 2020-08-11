# Flax Examples
 
## Official examples maintained by the Flax core team
 
This folder contains a collection of examples of implementations of various architectures and training procedures.
 
Each examples is designed to be self-contained and easily forkable, while reproducing "key results" in different areas of machine learning. Official Flax examples come with significant maintenance expectations, including (but not limited to since guidelines are WIP - [#231](https://github.com/google/flax/issues/231)):
* Tested benchmarks on single-GPU, multi-GPU, and TPU configurations (e.g. [imagenet](imagenet/))
* Unit tests (e.g. that one training step succeeds)
* Adherence to Flax best practices and code updates as those best practices evolve
* An "owner" who is a member of the Flax core team who keeps the example up-to-date
* A minimal implementation of a single model on a single dataset (users can fork and change the configuration to run on another dataset)
 
## Flax examples from the community
 
In addition to the curated list of official Flax examples, there is a growing community of people using Flax to build new types of machine learning models. We are happy to showcase any example built by the community here! If you want to submit your own example, we suggest that you start by forking one of the official Flax example, and start from there.
 
Here are some of the models that people have built with Flax:

| Link  | Author | Task type | Reference |
| ------------- | ------------- | ------------ | ---------- |
| [Gaussian Processes regression](https://github.com/danieljtait/ladax/tree/master/examples)  | @danieljtait | Regression | N/A |  |
| [JAX-RL](https://github.com/henry-prior/jax-rl)  | @henry-prior  | Reinforcement learning | N/A |
| [DQN](https://github.com/joaogui1/RL-JAX/tree/master/DQN)  | @joaogui1  | Reinforcement learning | N/A |
| [Various CIFAR SOTA Models](https://github.com/google-research/google-research/tree/master/flax_cifar) | @PForet | Image Classification | N/A |

If you are using Flax for your research, or have re-implemented interesting models in Flax, file a pull request against this README and add a link in the table.
 
## Looking for "FOO" implemented in Flax?

We use GitHub issues to keep track of which models people are most interested in seeing re-implemented in Flax. If you can't find what you're looking for in the [list](https://github.com/google/flax/labels/example%20request), file an issue with this [template](https://github.com/google/flax/issues/new?assignees=&template=example_request.md&title=).
 
If the model you are looking for has already been requested by others, upvote the issue to help us see which ones are the most requested.

## Looking to implement something in Flax?
 
Consider looking at the list of requested models in the GitHub issue tracker under the ["example requested" label](https://github.com/google/flax/labels/example%20request) and go ahead and build it!
