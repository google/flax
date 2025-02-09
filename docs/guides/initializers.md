---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "5D-VCn-KtpXk"}

# Initializers

In this guide, we will demonstrate how to use initializers and some of the common pitfalls when using initializers in Flax.

+++ {"id": "6nsUREar3k1H"}

`Initializers` are initialization functions for the parameters of your neural network. The kernel initializer (`kernel_init`) and the bias initializer (`bias_init`) are examples of optional arguments that you can pass `Initializers` to. A full list of Flax initializers can be found [here](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers), and most of them are inherited from the [JAX initializers](https://jax.readthedocs.io/en/latest/jax.nn.initializers.html).

The default kernel initializer is [`flax.linen.initializers.lecun_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.lecun_normal.html) and the default bias initializer is [`flax.linen.initializers.zeros_init`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.zeros_init.html).

```{code-cell}
:id: NPv5hHES8rA9

import numpy as np

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.core import scope
from flax.linen.initializers import lecun_normal, zeros_init, uniform
```

```{code-cell}
:id: TZEdgD40-bDo

x = jnp.empty((1,4))
rng = jax.random.PRNGKey(0)
params = nn.Dense(features=5).init(rng, x)['params']

# Internally, Flax generates a new RNG key for each param by folding in the
# original key with an integer; the integer is derived from a counter which 
# starts at 1.
kernel_rng = scope.LazyRng.create(rng, 1).as_jax_rng()
bias_rng = scope.LazyRng.create(rng, 2).as_jax_rng()

# confirm the params initialized by the default initializers in nn.Dense are 
# the same as manually initializing the params using lecun_normal and zeros_init
np.testing.assert_allclose(params['kernel'], lecun_normal()(kernel_rng, (4, 5)))
np.testing.assert_allclose(params['bias'], zeros_init()(bias_rng, (5,)))
```

```{code-cell}
:id: 0L-DwRnAQmqi

# Since the zeros initializer is deterministic and doesn't actually depend on 
# RNG, we can use the uniform initializer to also confirm how the RNG key for 
# the bias initializer is derived internally by Flax
params = nn.Dense(features=5,bias_init=uniform()).init(rng, x)['params']

np.testing.assert_allclose(params['kernel'], lecun_normal()(kernel_rng, (4, 5)))
np.testing.assert_allclose(params['bias'], uniform()(bias_rng, (5,)))
```

+++ {"id": "jPW_E1ZqHdrQ"}

## `Initializer` function signature

+++ {"id": "3Kglqd9vuxTG"}

To maintain consistency, all `Initializer` functions that are passed to the `kernel_init` and `bias_init` arguments **must follow the function signature: `[PRNGKey, Shape, Dtype] -> Array`**. Most functions in the [Flax initializer list](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers) are **builder functions** and build an `Initializer` function that follows this function signature. Therefore, you would normally call these builder functions and pass them to arguments like `kernel_init=lecun_normal()` and `bias_init=zeros_init()`. 

The two exceptions are [`flax.linen.initializers.zeros`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.zeros.html) and [`flax.linen.initializers.ones`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.ones.html), which **are already `Initializer` functions that follow the function signature**. These two functions are inherited from [JAX](https://jax.readthedocs.io/en/latest/jax.nn.initializers.html), but Flax also implements thinly-wrapped builder functions [`zeros_init`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.zeros_init.html) and [`ones_init`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.ones_init.html) that return those respective `Initializers` when called. For consistency and to minimize confusion, it is recommended to use `zeros_init()` and `ones_init()` instead of `zeros` and `ones`.

+++ {"id": "sG5oygwB7UrW"}

Note that even if the `Initializer` is deterministic, **a PRNGKey must be passed**. Therefore functions like [`jax.numpy.zeros`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.zeros.html) and [`jax.numpy.ones`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ones.html) can't be used as initializers since they have an invalid function signature.

```{code-cell}
:id: IjLW1abzw2pM
:outputId: de2f26d9-df65-4ee0-cf7f-acac4bc290f7

# jnp.zeros has an invalid function signature
layer = nn.Dense(features=5, kernel_init=jnp.zeros)
try:
  # this will throw an error
  layer.init(jax.random.PRNGKey(42), jnp.empty((1,4)))['params']
except Exception as e:
  print(f'Caught error: {e}')
```

```{code-cell}
:id: RlOIOCj68p86
:outputId: 4ef30f68-078a-4c57-e8de-2c49ef266cae

# use flax.linen.initializers.zeros_init instead
layer = nn.Dense(features=5, kernel_init=zeros_init())
# successfully generate params
layer.init(jax.random.PRNGKey(42), jnp.empty((1,4)))['params']
```

+++ {"id": "S4X_xHHk-b4V"}

## `Initializer` restrictions for `bias_init`

+++ {"id": "jLUFaliMy_DT"}

The `Initializer` functions built by [`constant`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.constant.html), [`normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.normal.html), [`uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.uniform.html), [`zeros_init`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.zeros_init.html) and [`ones_init`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.ones_init.html) and the `Initializer` functions [`zeros`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.zeros.html) and [`ones`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.ones.html) can be used as valid arguments for `bias_init`.

We can't use the variance scaling `Initializer` functions built by [`glorot_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.glorot_normal.html), [`glorot_uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.glorot_uniform.html), [`he_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.he_normal.html), [`he_uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.he_uniform.html), [`kaiming_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.kaiming_normal.html), [`kaiming_uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.kaiming_uniform.html), [`lecun_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.lecun_normal.html), [`lecun_uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.lecun_uniform.html), [`variance_scaling`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.variance_scaling.html), [`xavier_normal`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.xavier_normal.html), [`xavier_uniform`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.initializers.xavier_uniform.html) as arguments for `bias_init`, since they fail for 1D arrays.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: UW2I1Bda8mXu
outputId: 459bfff8-8a14-4440-ba09-d01dfa72b5a3
---
layer = nn.Dense(features=5, bias_init=lecun_normal())
try:
  # this will throw an error
  layer.init(jax.random.PRNGKey(42), jnp.empty((1,4)))['params']
except Exception as e:
  print(f'Caught error: {e}')
```

+++ {"id": "6kKUpPmR4xS8"}

For more information, refer to this [issue](https://github.com/google/jax/issues/2075#issuecomment-578465814).
