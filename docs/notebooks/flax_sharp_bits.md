---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# 🔪 Flax - The Sharp Bits 🔪

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/flax_sharp_bits.ipynb)

Flax exposes the full power of JAX. And just like when using JAX, there are certain _["sharp bits"](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)_ you may experience when working with Flax. This evolving document is designed to assist you with them.

First, install and/or update Flax:

```{code-cell} ipython3
:tags: [skip-execution]

! pip install -qq flax
```

## 🔪 `flax.linen.Dropout` layer and randomness

### TL;DR

When working on a model with dropout (subclassed from [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics)), add the `'dropout'` PRNGkey only during the forward pass.

1. Start with [`jax.random.split()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.split.html#jax-random-split) to explicitly create PRNG keys for `'params'` and `'dropout'`.
2. Add the [`flax.linen.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Dropout.html#flax.linen.Dropout) layer(s) to your model (subclassed from Flax [`Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics)).
3. When initializing the model ([`flax.linen.init()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#init-apply)), there's no need to pass in an extra `'dropout'` PRNG key—just the `'params'` key like in a "simpler" model.
4. During the forward pass with [`flax.linen.apply()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#init-apply), pass in `rngs={'dropout': dropout_key}`.

Check out a full example below.

### Why this works

- Internally, `flax.linen.Dropout` makes use of [`flax.linen.Module.make_rng`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.make_rng) to create a key for dropout (check out the [source code](https://github.com/google/flax/blob/5714e57a0dc8146eb58a7a06ed768ed3a17672f9/flax/linen/stochastic.py#L72)).
- Every time `make_rng` is called (in this case, it's done implicitly in `Dropout`), you get a new PRNG key split from the main/root PRNG key.
- `make_rng` still _guarantees full reproducibility_.

### Background 

The [dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) stochastic regularization technique randomly removes hidden and visible units in a network. Dropout is a random operation, requiring a PRNG state, and Flax (like JAX) uses [Threefry](https://github.com/google/jax/blob/main/docs/jep/263-prng.md) PRNG that is splittable. 

> Note: Recall that JAX has an explicit way of giving you PRNG keys: you can fork the main PRNG state (such as `key = jax.random.PRNGKey(seed=0)`) into multiple new PRNG keys with `key, subkey = jax.random.split(key)`. Refresh your memory in [🔪 JAX - The Sharp Bits 🔪 Randomness and PRNG keys](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers).

Flax provides an _implicit_ way of handling PRNG key streams via [Flax `Module`](https://flax.readthedocs.io/en/latest/guides/flax_basics.html#module-basics)'s [`flax.linen.Module.make_rng`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.make_rng) helper function. It allows the code in Flax `Module`s (or its sub-`Module`s) to "pull PRNG keys". `make_rng` guarantees to provide a unique key each time you call it.

> Note: Recall that [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module) is the base class for all neural network modules. All layers and models are subclassed from it.

### Example

Remember that each of the Flax PRNG streams has a name. The example below uses the `'params'` stream for initializing parameters, as well as the `'dropout'` stream. The PRNG key provided to [`flax.linen.init()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#init-apply) is the one that seeds the `'params'` PRNG key stream. To draw PRNG keys during the forward pass (with dropout), provide a PRNG key to seed that stream (`'dropout'`) when you call `Module.apply()`.

```{code-cell} ipython3
# Setup.
import jax
import jax.numpy as jnp
import flax.linen as nn
```

```{code-cell} ipython3
# Randomness.
seed = 0
root_key = jax.random.PRNGKey(seed=seed)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

# A simple network.
class MyModel(nn.Module):
  num_neurons: int
  training: bool
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.num_neurons)(x)
    # Set the dropout layer with a rate of 50% .
    # When the `deterministic` flag is `True`, dropout is turned off.
    x = nn.Dropout(rate=0.5, deterministic=not self.training)(x)
    return x

# Instantiate `MyModel` (you don't need to set `training=True` to
# avoid performing the forward pass computation).
my_model = MyModel(num_neurons=3, training=False)

x = jax.random.uniform(key=main_key, shape=(3, 4, 4))

# Initialize with `flax.linen.init()`.
# The `params_key` is equivalent to a dictionary of PRNGs.
# (Here, you are providing only one PRNG key.) 
variables = my_model.init(params_key, x)

# Perform the forward pass with `flax.linen.apply()`.
y = my_model.apply(variables, x, rngs={'dropout': dropout_key})
```

Real-life examples:

* Applying word dropout to a batch of input IDs (in a [text classification](https://github.com/google/flax/blob/main/examples/sst2/models.py) context).
* Defining a prediction token in a decoder of a [sequence-to-sequence model](https://github.com/google/flax/blob/main/examples/seq2seq/models.py).
