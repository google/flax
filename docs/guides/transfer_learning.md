---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: 'Python 3.8.10 (''.venv'': venv)'
  language: python
  name: python3
---

# Transfer learning

+++

This guide demonstrates various parts of the transfer learning workflow with Flax. Depending on your task, you can use a pretrained model as a feature extractor or fine-tune the entire model. This guide uses simple classification as a default task. You will learn how to:

* Load a pretrained model from HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) and extract a specific sub-module from that pretrained model.
* Create the classifier model.
* Transfer the pretrained parameters to the new model structure.
* Set up optimization for training different parts of the model separately with [Optax](https://optax.readthedocs.io/).
* Set up the model for training.

**Note:** Depending on your task, some of the content in this guide may be suboptimal. For example, if you are only going to train a linear classifier on top of a pretrained model, it may be better to just extract the feature embeddings once, which can result in much faster training, and you can use specialized algorithms for linear regression or logistic classification. This guide shows how to do transfer learning with all the model parameters.

+++

## Setup

```{code-cell} ipython3
:tags: [skip-execution]

# Note that the Transformers library doesn't use the latest Flax version.
! pip install transformers[flax]
# Install/upgrade Flax and JAX. For JAX installation with GPU/TPU support,
# visit https://github.com/google/jax#installation.
! pip install -U flax jax jaxlib
```

## Create a function for model loading

To load a pre-trained classifier, you can create a custom function that will return a [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics) and its pretrained variables.

In the code below, the `load_model` function uses HuggingFace's `FlaxCLIPVisionModel` model from the [Transformers](https://huggingface.co/docs/transformers/index) library and extracts a `FlaxCLIPModule` module (note that it is not a Flax `Module`):

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

### Extract a sub-model from the loaded trained model

Calling `load_model` from the snippet above returns the `FlaxCLIPModule`, which is composed of text and vision sub-modules.

Suppose you want to extract the `vision_model` sub-module defined inside `.setup()` and its variables. To do this you can use [`nn.apply`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.apply) to run a helper function that will grant you access to submodules and their variables:

```{code-cell} ipython3
import flax.linen as nn

clip, clip_variables = load_model()

def extract_submodule(clip):
    vision_model = clip.vision_model.clone()
    variables = clip.vision_model.variables
    return vision_model, variables

vision_model, vision_model_variables = nn.apply(extract_submodule, clip)(clip_variables)
```

Notice that here `.clone()` was used to get an unbounded copy of `vision_model`, this is important to avoid leakage as bounded modules contain their variables.

### Create the classifier

Next create a `Classifier` model with [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics), consisting of a `backbone` (the pretrained vision model) and a `head` (the classifier).

```{code-cell} ipython3
import jax.numpy as jnp
import jax

class Classifier(nn.Module):
  num_classes: int
  backbone: nn.Module

  @nn.compact
  def __call__(self, x):
    x = self.backbone(x).pooler_output
    x = nn.Dense(self.num_classes, name='head')(x)
    return x
```

Then, pass the `vision_model` sub-module as the backbone to the `Classifier` to create the complete model.

You can randomly initialize the model's variables using some toy data for demonstration purposes.

```{code-cell} ipython3
num_classes = 3
model = Classifier(num_classes=num_classes, backbone=vision_model)

x = jnp.ones((1, 224, 224, 3))
variables = model.init(jax.random.PRNGKey(1), x)
```

## Transfer the parameters

Since `variables` are randomly initialized, you now have to transfer the parameters from `vision_model_variables` to the complete `variables` at the appropriate location. This can be done by unfreezing the `variables`, updating the `backbone` parameters, and freezing the `variables` again:

```{code-cell} ipython3
from flax.core.frozen_dict import freeze

variables = variables.unfreeze()
variables['params']['backbone'] = vision_model_variables['params']
variables = freeze(variables)
```

## Optimization

If you need to to train different parts of the model separately, you have two options:

1. Use `stop_gradient`.
2. Filter the parameters for `jax.grad`.
3. Use multiple optimizers for different parameters.

While each could be useful in different situations, its recommended to use use multiple optimizers via [Optax](https://optax.readthedocs.io/)'s [`optax.multi_transform`](https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform) because it is efficient and can be easily extended to implement differential learning rates. To use `optax.multi_transform` you have to do two things:

1. Define some parameter partitions.
2. Create a mapping between partitions and their optimizer.
3. Create a pytree with the same shape as the parameters but its leaves containing the corresponding partition label.

## Freeze layers

To freeze layers with `optax.multi_transform`, create the `trainable` and `frozen` parameter partitions.

In the example below:

- For the `trainable` parameters use the Adam (`optax.adam`) optimizer.
- For the `frozen` parameters use `optax.set_to_zero`, which zeros-out the gradients.
- To map parameters to partitions, you can use the [`flax.traverse_util.path_aware_map`](https://flax.readthedocs.io/en/latest/api_reference/flax.traverse_util.html#flax.traverse_util.path_aware_map) function, by leveraging the `path` argument you can map the `backbone` parameters to `frozen` and the rest to `trainable`.

```{code-cell} ipython3
from flax import traverse_util
import optax

partition_optimizers = {'trainable': optax.adam(5e-3), 'frozen': optax.set_to_zero()}
param_partitions = freeze(traverse_util.path_aware_map(
  lambda path, v: 'frozen' if 'backbone' in path else 'trainable', variables['params']))
tx = optax.multi_transform(partition_optimizers, param_partitions)

# visualize a subset of the param_partitions structure
flat = list(traverse_util.flatten_dict(param_partitions).items())
freeze(traverse_util.unflatten_dict(dict(flat[:2] + flat[-2:])))
```

To implement _differential learning rates_ simply replace `optax.set_to_zero` with the optimizer of your choice, you can choose different optimizers and partitioning schemes depending on your needs.

For more information on advanced optimizers, refer to Optax's [Combining Optimizers](https://optax.readthedocs.io/en/latest/api.html#combining-optimizers) documentation.

## Create the `TrainState` object for model training

Once you define your module, variables, and optimizer, you can construct the `TrainState` object and proceed to train the model as you normally would.

```{code-cell} ipython3
from flax.training.train_state import TrainState

state = TrainState.create(
  apply_fn=model.apply,
  params=variables['params'],
  tx=tx)
```
