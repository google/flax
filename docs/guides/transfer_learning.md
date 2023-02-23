---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Transfer learning

+++

This guide demonstrates various parts of the transfer learning workflow with Flax. Depending on the task, a pretrained model can be used just as a feature extractor or it can be fine-tuned as part of a larger model.

This guide demonstrates how to:

* Load a pretrained model from HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) and extract a specific sub-module from that pretrained model.
* Create a classifier model.
* Transfer the pretrained parameters to the new model structure.
* Create an optimizer for training different parts of the model separately with [Optax](https://optax.readthedocs.io/).
* Set up the model for training.

<details><summary><b>Performance Note</b></summary>

Depending on your task, some of the content in this guide may be suboptimal. For example, if you are only going to train a linear classifier on top of a pretrained model, it may be better to just extract the feature embeddings once, which can result in much faster training, and you can use specialized algorithms for linear regression or logistic classification. This guide shows how to do transfer learning with all the model parameters.

</details><br>

+++

## Setup

```{code-cell} ipython3
:tags: [skip-execution]

# Note that the Transformers library doesn't use the latest Flax version.
! pip install -q transformers[flax]
# Install/upgrade Flax and JAX. For JAX installation with GPU/TPU support,
# visit https://github.com/google/jax#installation.
! pip install -U -q flax jax jaxlib
```

## Create a function for model loading

To load a pre-trained classifier, for convenience first create a function that returns a [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics) and its pretrained variables.

In the code below, the `load_model` function uses HuggingFace's `FlaxCLIPVisionModel` model from the [Transformers](https://huggingface.co/docs/transformers/index) library and extracts a `FlaxCLIPModule` module.

```{code-cell} ipython3
%%capture
from IPython.display import clear_output
from transformers import FlaxCLIPModel

# Note: FlaxCLIPModel is not a Flax Module
def load_model():
  clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
  clear_output(wait=False) # Clear the loading messages
  module = clip.module # Extract the Flax Module
  variables = {'params': clip.params} # Extract the parameters
  return module, variables
```

Note that `FlaxCLIPVisionModel` itself is not a Flax `Module` which is why we need to do this extra step.

### Extracting a submodule

Calling `load_model` from the snippet above returns the `FlaxCLIPModule`, which is composed of `text_model` and `vision_model` submodules.

An easy way to extract the `vision_model` sub-Module defined inside `.setup()` and its variables is to use [`flax.linen.Module.bind`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.bind) on the `clip` Module immediately followed by [`flax.linen.Module.unbind`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.unbind) on the `vision_model` sub-Module.

```{code-cell} ipython3
import flax.linen as nn

clip, clip_variables = load_model()
vision_model, vision_model_vars = clip.bind(clip_variables).vision_model.unbind()
```

### Creating a classifier

To create a classifier define a new Flax [`Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics) consisting of a `backbone` (the pretrained vision model) and a `head` (the classifier) submodules.

```{code-cell} ipython3
from typing import Callable
import jax.numpy as jnp
import jax

class Classifier(nn.Module):
  num_classes: int
  backbone: nn.Module
  

  @nn.compact
  def __call__(self, x):
    x = self.backbone(x).pooler_output
    x = nn.Dense(
      self.num_classes, name='head', kernel_init=nn.zeros)(x)
    return x
```

To construct a classifier `model`, the `vision_model` Module is passed as the `backbone` to `Classifier`. Then the model's `params` can be randomly initialized by passing fake data that is used to infer the parameter shapes.

```{code-cell} ipython3
num_classes = 3
model = Classifier(num_classes=num_classes, backbone=vision_model)

x = jnp.empty((1, 224, 224, 3))
variables = model.init(jax.random.PRNGKey(1), x)
params = variables['params']
```

## Transfering the parameters
Since `params` are currently random, the pretrained parameters from `vision_model_vars` have to be transfered to the `params` structure at the appropriate location. This can be done by unfreezing `params`, updating the `backbone` parameters, and freezing the `params` again:

```{code-cell} ipython3
from flax.core.frozen_dict import freeze

params = params.unfreeze()
params['backbone'] = vision_model_vars['params']
params = freeze(params)
```

**Note:** if the model contains other variable collections such as `batch_stats`, these have to be transfered as well.

## Optimization

If you need to to train different parts of the model separately, you have three options:

1. Use `stop_gradient`.
2. Filter the parameters for `jax.grad`.
3. Use multiple optimizers for different parameters.

For most situations we recommend using multiple optimizers via [Optax](https://optax.readthedocs.io/)'s [`multi_transform`](https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform) as its both efficient and can be easily extended to implement many fine-tunning strategies. 

### **optax.multi_transform**

To use `optax.multi_transform` following must be defined:

1. The parameter partitions.
2. A mapping between partitions and their optimizer.
3. A pytree with the same shape as the parameters but its leaves containing the corresponding partition label.

To freeze layers with `optax.multi_transform` for the model above, the following setup can be used:

* Define the `trainable` and `frozen` parameter partitions.
* For the `trainable` parameters select the Adam (`optax.adam`) optimizer.
- For the `frozen` parameters select the `optax.set_to_zero` optimizer. This dummy optimizer zeros-out the gradients so no training is done.
- Map parameters to partitions using [`flax.traverse_util.path_aware_map`](https://flax.readthedocs.io/en/latest/api_reference/flax.traverse_util.html#flax.traverse_util.path_aware_map), mark the leaves from the `backbone` as `frozen`, and the rest as `trainable`.

```{code-cell} ipython3
from flax import traverse_util
import optax

partition_optimizers = {'trainable': optax.adam(5e-3), 'frozen': optax.set_to_zero()}
param_partitions = freeze(traverse_util.path_aware_map(
  lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params))
tx = optax.multi_transform(partition_optimizers, param_partitions)

# visualize a subset of the param_partitions structure
flat = list(traverse_util.flatten_dict(param_partitions).items())
freeze(traverse_util.unflatten_dict(dict(flat[:2] + flat[-2:])))
```

To implement [differential learning rates](https://blog.slavv.com/differential-learning-rates-59eff5209a4f), the `optax.set_to_zero` can be replaced with any other optimizer, different optimizers and partitioning schemes can be selected depending on the task. For more information on advanced optimizers, refer to Optax's [Combining Optimizers](https://optax.readthedocs.io/en/latest/api.html#combining-optimizers) documentation.

## Creating the `TrainState`

Once the module, params, and optimizer are defined, the `TrainState` can be constructed as usual:

```{code-cell} ipython3
from flax.training.train_state import TrainState

state = TrainState.create(
  apply_fn=model.apply,
  params=params,
  tx=tx)
```

Since the optimizer takes care of the freezing or fine-tunning strategy, the `train_step` requires no additional changes, training can proceed normally.
