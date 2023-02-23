---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "yf-nWLh0naJi"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/guides/flax_basics.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/guides/flax_basics.ipynb)

# Flax Basics

This notebook will walk you through the following workflow:

*   Instantiating a model from Flax built-in layers or third-party models.
*   Initializing parameters of the model and manually written training.
*   Using optimizers provided by Flax to ease training.
*   Serialization of parameters and other objects.
*   Creating your own models and managing state.

+++ {"id": "KyANAaZtbs86"}

## Setting up our environment

Here we provide the code needed to set up the environment for our notebook.

```{code-cell}
:id: qdrEVv9tinJn
:outputId: e30aa464-fa52-4f35-df96-716c68a4b3ee
:tags: [skip-execution]

# Install the latest JAXlib version.
!pip install --upgrade -q pip jax jaxlib
# Install Flax at head:
!pip install --upgrade -q git+https://github.com/google/flax.git
```

```{code-cell}
:id: kN6bZDaReZO2

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
```

+++ {"id": "pCCwAbOLiscA"}

## Linear regression with Flax

In the previous *JAX for the impatient* notebook, we finished up with a linear regression example. As we know, linear regression can also be written as a single dense neural network layer, which we will show in the following so that we can compare how it's done.

A dense layer is a layer that has a kernel parameter $W\in\mathcal{M}_{m,n}(\mathbb{R})$ where $m$ is the number of features as an output of the model, and $n$ the dimensionality of the input, and a bias parameter $b\in\mathbb{R}^m$. The dense layers returns $Wx+b$ from an input $x\in\mathbb{R}^n$.

This dense layer is already provided by Flax in the `flax.linen` module (here imported as `nn`).

```{code-cell}
:id: zWX2zEtphT4Y

# We create one dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)
```

+++ {"id": "UmzP1QoQYAAN"}

Layers (and models in general, we'll use that word from now on) are subclasses of the `linen.Module` class.

### Model parameters & initialization

Parameters are not stored with the models themselves. You need to initialize parameters by calling the `init` function, using a PRNGKey and dummy input data.

```{code-cell}
:id: K529lhzeYtl8
:outputId: 06feb9d2-db50-4f41-c169-6df4336f43a5

key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (10,)) # Dummy input data
params = model.init(key2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params) # Checking output shapes
```

+++ {"id": "NH7Y9xMEewmO"}

*Note: JAX and Flax, like NumPy, are row-based systems, meaning that vectors are represented as row vectors and not column vectors. This can be seen in the shape of the kernel here.*

The result is what we expect: bias and kernel parameters of the correct size. Under the hood:

*   The dummy input data `x` is used to trigger shape inference: we only declared the number of features we wanted in the output of the model, not the size of the input. Flax finds out by itself the correct size of the kernel.
*   The random PRNG key is used to trigger the initialization functions (those have default values provided by the module here).
* Initialization functions are called to generate the initial set of parameters that the model will use. Those are functions that take as arguments `(PRNG Key, shape, dtype)` and return an Array of shape `shape`.
* The init function returns the initialized set of parameters (you can also get the output of the forward pass on the dummy input with the same syntax by using the `init_with_output` method instead of `init`.

+++ {"id": "3yL9mKk7naJn"}

The output shows that the parameters are stored in a `FrozenDict` instance, which helps deal with the functional nature of JAX by preventing any mutation of the underlying dict and making the user aware of it. Read more about it in the [`flax.core.frozen_dict.FrozenDict` API docs](https://flax.readthedocs.io/en/latest/api_reference/flax.core.frozen_dict.html#flax.core.frozen_dict.FrozenDict).

As a consequence, the following doesn't work:

```{code-cell}
:id: HtOFWeiynaJo
:outputId: 689b4230-2a3d-4823-d103-2858e6debc4d

try:
    params['new_key'] = jnp.ones((2,2))
except ValueError as e:
    print("Error: ", e)
```

+++ {"id": "M1qo9M3_naJo"}

To conduct a forward pass with the model with a given set of parameters (which are never stored with the model), we just use the `apply` method by providing it the parameters to use as well as the input:

```{code-cell}
:id: J8ietJecWiuK
:outputId: 7bbe6bb4-94d5-4574-fbb5-aa0fcd1c84ae

model.apply(params, x)
```

+++ {"id": "lVsjgYzuSBGL"}

### Gradient descent

If you jumped here directly without going through the JAX part, here is the linear regression formulation we're going to use: from a set of data points $\{(x_i,y_i), i\in \{1,\ldots, k\}, x_i\in\mathbb{R}^n,y_i\in\mathbb{R}^m\}$, we try to find a set of parameters $W\in \mathcal{M}_{m,n}(\mathbb{R}), b\in\mathbb{R}^m$ such that the function $f_{W,b}(x)=Wx+b$ minimizes the mean squared error:

$$\mathcal{L}(W,b)\rightarrow\frac{1}{k}\sum_{i=1}^{k} \frac{1}{2}\|y_i-f_{W,b}(x_i)\|^2_2$$

Here, we see that the tuple $(W,b)$ matches the parameters of the Dense layer. We'll perform gradient descent using those. Let's first generate the fake data we'll use. The data is exactly the same as in the JAX part's linear regression pytree example.

```{code-cell}
:id: bFIiMnL4dl-e
:outputId: 6eae59dc-0632-4f53-eac8-c22a7c646a52

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)
```

+++ {"id": "ZHkioicCiUbx"}

We copy the same training loop that we used in the JAX pytree linear regression example with `jax.value_and_grad()`, but here we can use `model.apply()` instead of having to define our own feed-forward function (`predict_pytree()` in the [JAX example](https://flax.readthedocs.io/en/latest/guides/jax_for_the_impatient.html#linear-regression-with-pytrees)).

```{code-cell}
:id: JqJaVc7BeNyT

# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x, y):
    pred = model.apply(params, x)
    return jnp.inner(y-pred, y-pred) / 2.0
  # Vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
```

+++ {"id": "wGKru__mi15v"}

And finally perform the gradient descent.

```{code-cell}
:id: ePEl1ndse0Jq
:outputId: 50d975b3-4706-4d8a-c4b8-2629ab8e3ac4

learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)

@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

for i in range(101):
  # Perform one gradient update.
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  params = update_params(params, learning_rate, grads)
  if i % 10 == 0:
    print(f'Loss step {i}: ', loss_val)
```

+++ {"id": "zqEnJ9Poyb6q"}

### Optimizing with Optax

Flax used to use its own `flax.optim` package for optimization, but with
[FLIP #1009](https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md)
this was deprecated in favor of
[Optax](https://github.com/deepmind/optax).

Basic usage of Optax is straightforward:

1.   Choose an optimization method (e.g. `optax.adam`).
2.   Create optimizer state from parameters (for the Adam optimizer, this state will contain the [momentum values](https://optax.readthedocs.io/en/latest/api.html#optax.adam)).
3.   Compute the gradients of your loss with `jax.value_and_grad()`.
4.   At every iteration, call the Optax `update` function to update the internal
     optimizer state and create an update to the parameters. Then add the update
     to the parameters with Optax's `apply_updates` method.

Note that Optax can do a lot more: it's designed for composing simple gradient
transformations into more complex transformations that allows to implement a
wide range of optimizers. There is also support for changing optimizer
hyperparameters over time ("schedules"), applying different updates to different
parts of the parameter tree ("masking") and much more. For details please refer
to the
[official documentation](https://optax.readthedocs.io/en/latest/).

```{code-cell}
:id: Ce77uDJx1bUF

import optax
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)
```

```{code-cell}
:id: PTSv0vx13xPO
:outputId: eec0c096-1d9e-4b3c-f8e5-942ee63828ec

for i in range(101):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)
```

+++ {"id": "0eAPPwtpXYu7"}

### Serializing the result

Now that we're happy with the result of our training, we might want to save the model parameters to load them back later. Flax provides a serialization package to enable you to do that.

```{code-cell}
:id: BiUPRU93XnAZ
:outputId: b97e7d83-3e40-4a80-b1fe-1f6ceff30a0c

from flax import serialization
bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)
```

+++ {"id": "eielPo2KZByd"}

To load the model back, you'll need to use a template of the model parameter structure, like the one you would get from the model initialization. Here, we use the previously generated `params` as a template. Note that this will produce a new variable structure, and not mutate in-place.

*The point of enforcing structure through template is to avoid users issues downstream, so you need to first have the right model that generates the parameters structure.*

```{code-cell}
:id: MOhoBDCOYYJ5
:outputId: 13acc4e1-8757-4554-e2c8-d594ba6e67dc

serialization.from_bytes(params, bytes_output)
```

+++ {"id": "8mNu8nuOhDC5"}

## Defining your own models

Flax allows you to define your own models, which should be a bit more complicated than a linear regression. In this section, we'll show you how to build simple models. To do so, you'll need to create subclasses of the base `nn.Module` class.

*Keep in mind that we imported* `linen as nn` *and this only works with the new linen API*

+++ {"id": "1sllHAdRlpmQ"}

### Module basics

The base abstraction for models is the `nn.Module` class, and every type of predefined layers in Flax (like the previous `Dense`) is a subclass of `nn.Module`. Let's take a look and start by defining a simple but custom multi-layer perceptron i.e. a sequence of Dense layers interleaved with calls to a non-linear activation function.

```{code-cell}
:id: vbfrfbkxgPhg
:outputId: b59c679c-d164-4fd6-92db-b50f0d310ec3

class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
```

+++ {"id": "DDITIjXitEZl"}

As we can see, a `nn.Module` subclass is made of:

*   A collection of data fields (`nn.Module` are Python dataclasses) - here we only have the `features` field of type `Sequence[int]`.
*   A `setup()` method that is being called at the end of the `__postinit__` where you can register submodules, variables, parameters you will need in your model.
*   A `__call__` function that returns the output of the model from a given input.
*   The model structure defines a pytree of parameters following the same tree structure as the model: the params tree contains one `layers_n` sub dict per layer, and each of those contain the parameters of the associated Dense layer. The layout is very explicit.

*Note: lists are mostly managed as you would expect (WIP), there are corner cases you should be aware of as pointed out* [here](https://github.com/google/flax/issues/524)

Since the module structure and its parameters are not tied to each other, you can't directly call `model(x)` on a given input as it will return an error. The `__call__` function is being wrapped up in the `apply` one, which is the one to call on an input:

```{code-cell}
:id: DEYrVA6dnaJu
:outputId: 4af16ec5-b52a-43b0-fc47-1f8ab25e7058

try:
    y = model(x) # Returns an error
except AttributeError as e:
    print(e)
```

+++ {"id": "I__UrmShnaJu"}

Since here we have a very simple model, we could have used an alternative (but equivalent) way of declaring the submodules inline in the `__call__` using the `@nn.compact` annotation like so:

```{code-cell}
:id: ZTCbdpQ4suSK
:outputId: 183a74ef-f54e-4848-99bf-fee4c174ba6d

class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # providing a name is optional though!
      # the default autonames would be "Dense_0", "Dense_1", ...
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
```

+++ {"id": "es7YHjgexT-L"}

There are, however, a few differences you should be aware of between the two declaration modes:

*   In `setup`, you are able to name some sublayers and keep them around for further use (e.g. encoder/decoder methods in autoencoders).
*   If you want to have multiple methods, then you **need** to declare the module using `setup`, as the `@nn.compact` annotation only allows one method to be annotated.
*   The last initialization will be handled differently. See these notes for more details (TODO: add notes link).

+++ {"id": "-ykceROJyp7W"}

### Module parameters

In the previous MLP example, we relied only on predefined layers and operators (`Dense`, `relu`). Let's imagine that you didn't have a Dense layer provided by Flax and you wanted to write it on your own. Here is what it would look like using the `@nn.compact` way to declare a new modules:

```{code-cell}
:id: wK371Pt_vVfR
:outputId: 83b5fea4-071e-4ea0-8fa8-610e69fb5fd5

class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.features))  # shape info.
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),) # TODO Why not jnp.dot?
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)
```

+++ {"id": "MKyhfzVpzC94"}

Here, we see how to both declare and assign a parameter to the model using the `self.param` method. It takes as input `(name, init_fn, *init_args)` :

*   `name` is simply the name of the parameter that will end up in the parameter structure.
*   `init_fn` is a function with input `(PRNGKey, *init_args)` returning an Array, with `init_args` being the arguments needed to call the initialisation function.
*   `init_args` are the arguments to provide to the initialization function.

Such params can also be declared in the `setup` method; it won't be able to use shape inference because Flax is using lazy initialization at the first call site.

+++ {"id": "QmSpxyqLDr58"}

### Variables and collections of variables

As we've seen so far, working with models means working with:

*   A subclass of `nn.Module`;
*   A pytree of parameters for the model (typically from `model.init()`);

However this is not enough to cover everything that we would need for machine learning, especially neural networks. In some cases, you might want your neural network to keep track of some internal state while it runs (e.g. batch normalization layers). There is a way to declare variables beyond the parameters of the model with the `variable` method.

For demonstration purposes, we'll implement a simplified but similar mechanism to batch normalization: we'll store running averages and subtract those to the input at training time. For proper batchnorm, you should use (and look at) the implementation [here](https://github.com/google/flax/blob/main/flax/linen/normalization.py).

```{code-cell}
:id: J6_tR-nPzB1i
:outputId: 75465fd6-cdc8-497c-a3ec-7f709b5dde7a

class BiasAdderWithRunningMean(nn.Module):
  decay: float = 0.99

  @nn.compact
  def __call__(self, x):
    # easy pattern to detect if we're initializing via empty variable tree
    is_initialized = self.has_variable('batch_stats', 'mean')
    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s),
                            x.shape[1:])
    mean = ra_mean.value # This will either get the value or trigger init
    bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
    if is_initialized:
      ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

    return x - ra_mean.value + bias


key1, key2 = random.split(random.PRNGKey(0), 2)
x = jnp.ones((10,5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print('initialized variables:\n', variables)
y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
print('updated state:\n', updated_state)
```

+++ {"id": "5OHBbMJng3ic"}

Here, `updated_state` returns only the state variables that are being mutated by the model while applying it on data. To update the variables and get the new parameters of the model, we can use the following pattern:

```{code-cell}
:id: IbTsCAvZcdBy
:outputId: 09a8bdd1-eaf8-401a-cf7c-386a7a5aa87b

for val in [1.0, 2.0, 3.0]:
  x = val * jnp.ones((10,5))
  y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
  old_state, params = variables.pop('params')
  variables = freeze({'params': params, **updated_state})
  print('updated state:\n', updated_state) # Shows only the mutable part
```

+++ {"id": "GuUSOSKegKIM"}

From this simplified example, you should be able to derive a full BatchNorm implementation, or any layer involving a state. To finish, let's add an optimizer to see how to play with both parameters updated by an optimizer and state variables.

*This example isn't doing anything and is only for demonstration purposes.*

```{code-cell}
:id: TUgAbUPpnaJw
:outputId: 0906fbab-b866-4956-d231-b1374415d448

from functools import partial

@partial(jax.jit, static_argnums=(0, 1))
def update_step(tx, apply_fn, x, opt_state, params, state):

  def loss(params):
    y, updated_state = apply_fn({'params': params, **state},
                                x, mutable=list(state.keys()))
    l = ((x - y) ** 2).sum()
    return l, updated_state

  (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state, params, state

x = jnp.ones((10,5))
variables = model.init(random.PRNGKey(0), x)
state, params = variables.pop('params')
del variables
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for _ in range(3):
  opt_state, params, state = update_step(tx, model.apply, x, opt_state, params, state)
  print('Updated state: ', state)
```

+++ {"id": "eWUmx5EjtWge"}

Note that the above function has a quite verbose signature and it would not actually
work with `jax.jit()` because the function arguments are not "valid JAX types".

Flax provides a handy wrapper - `TrainState` - that simplifies the above code. Check out [`flax.training.train_state.TrainState`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState) to learn more.

+++ {"id": "_GL0PsCwnaJw"}

### Exporting to Tensorflow's SavedModel with jax2tf

JAX released an experimental converter called [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf), which allows converting trained Flax models into Tensorflow's SavedModel format (so it can be used for [TF Hub](https://www.tensorflow.org/hub), [TF.lite](https://www.tensorflow.org/lite), [TF.js](https://www.tensorflow.org/js), or other downstream applications). The repository contains more documentation and has various examples for Flax.
