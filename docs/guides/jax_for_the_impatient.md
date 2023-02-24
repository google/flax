---
jupytext:
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "SwtfSYdoHsc_"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/flax/blob/main/docs/guides/jax_for_the_impatient.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/guides/jax_for_the_impatient.ipynb)

# JAX for the Impatient
**JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.**

Here we will cover the basics of JAX so that you can get started with Flax, however we very much recommend that you go through JAX's documentation [here](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) after going over the basics here.

+++ {"id": "gF2oOT78zOIr"}

## NumPy API

Let's start by exploring the NumPy API coming from JAX and the main differences you should be aware of.

```{code-cell}
:id: 5csM8DZYEqk6

import jax
from jax import numpy as jnp, random

import numpy as np # We import the standard NumPy library
```

+++ {"id": "Z5BLL6v_JUSI"}

`jax.numpy` is the NumPy-like API that needs to be imported, and we will also use `jax.random` to generate some data to work on.

Let's start by generating some matrices, and then try matrix multiplication.

```{code-cell}
:id: L2HKiLTNJ4Eh
:outputId: c4297a1a-4e4b-4bdc-ca5d-3d33aca92b3b

m = jnp.ones((4,4)) # We're generating one 4 by 4 matrix filled with ones.
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]]) # An explicit 2 by 4 array
m
```

+++ {"id": "NKFtn4d_Nu07"}

Arrays in JAX are represented as DeviceArray instances and are agnostic to the place where the array lives (CPU, GPU, or TPU). This is why we're getting the warning that no GPU/TPU was found and JAX is falling back to a CPU (unless you're running it in an environment that has a GPU/TPU available).

We can obviously multiply matrices like we would do in NumPy.

```{code-cell}
:id: 9do-ZRGaRThn
:outputId: 9c4feb4d-3bd1-4921-97ce-c8087b37496f

jnp.dot(n, m).block_until_ready() # Note: yields the same result as np.dot(m)
```

+++ {"id": "Jkyt5xXpRidn"}

DeviceArray instances are actually futures ([more here](https://jax.readthedocs.io/en/latest/async_dispatch.html)) due to the **default asynchronous execution** in JAX. For that reason, the Python call might return before the computation actually ends, hence we're using the `block_until_ready()` method to ensure we return the end result.

JAX is fully compatible with NumPy, and can transparently process arrays from one library to the other.

```{code-cell}
:id: hFthGlHoRZ59
:outputId: 15892d6a-c06c-4f98-a7d4-ad432bdd1f57

x = np.random.normal(size=(4,4)) # Creating one standard NumPy array instance
jnp.dot(x,m)
```

+++ {"id": "AoaA-FS2XpsC"}

If you're using accelerators, using NumPy arrays directly will result in multiple transfers from CPU to GPU/TPU memory. You can save that transfer bandwidth, either by creating directly a DeviceArray or by using `jax.device_put` on the NumPy array. With DeviceArrays, computation is done on device so no additional data transfer is required, e.g. `jnp.dot(long_vector, long_vector)` will only transfer a single scalar (result of the computation) back from device to host.

```{code-cell}
:id: -VABtdIwTFfN
:outputId: 08965869-bdd7-44c8-ae46-207061b5112c

x = np.random.normal(size=(4,4))
x = jax.device_put(x)
x
```

+++ {"id": "y_2QavY1tR8j"}

Conversely, if you want to get back a Numpy array from a JAX array, you can simply do so by using it in the Numpy API.

```{code-cell}
:id: vEJ1mSvStjEC
:outputId: 00a8cc38-59a2-4cf9-ed23-eb5fbb708495

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
np.array(x)
```

+++ {"id": "CBHVd3GTpLKD"}

## (Im)mutability
JAX is functional by essence, one practical consequence being that JAX arrays are immutable. This means no in-place ops and sliced assignments. More generally, functions should not take input or produce output using a global state.

```{code-cell}
:id: -erZrgZXawFW
:outputId: c3c03081-6235-482f-a88c-cc180f661954

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
updated = x.at[0, 0].set(3.0) # whereas x[0,0] = 3.0 would fail
print("x: \n", x) # Note that x didn't change, no in-place mutation.
print("updated: \n", updated)
```

+++ {"id": "Sz_9b-XUTjjl"}

All jax ops are available with this syntax, including: `set`, `add`, `mul`, `min`, `max`.

+++ {"id": "o8QGdusyzbmP"}

## Managing randomness
In JAX, randomness is managed in a very specific way, and you can read more on JAX's docs [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers) and [here](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html) (we borrow content from there!). As the JAX team puts it:

*JAX implements an explicit PRNG where entropy production and consumption are handled by explicitly passing and iterating a PRNG state. JAX uses a modern Threefry counter-based PRNG that’s splittable. That is, its design allows us to fork the PRNG state into new PRNGs for use with parallel stochastic generation.*

In short, you need to explicitly manage the PRNGs (pseudo random number generators) and their states. In JAX's PRNGs, the state is represented as a pair of two unsigned-int32s that is called a key (there is no special meaning to the two unsigned int32s -- it's just a way of representing a uint64).

```{code-cell}
:id: 8iz9KGF4s7nN
:outputId: c5bb1581-090b-42ed-cc42-08436154bc14

key = random.PRNGKey(0)
key
```

+++ {"id": "1y622foIaYjL"}

If you use this key multiple times, you'll get the same "random" output each time. To generate further entries in the sequence, you'll need to split the PRNG and thus generate a new pair of keys.

```{code-cell}
---
vscode:
  languageId: python
---
for i in range(3):
    print("Printing the random number using key: ", key, " gives: ", random.normal(key,shape=(1,))) # Boringly not that random since we use the same key
```

```{code-cell}
:id: lOBv5CaB3dMa
:outputId: ac89afdc-a73e-4c31-d005-7e1e6ad551cd

print("old key", key, "--> normal", random.normal(key, shape=(1,)))
key, subkey = random.split(key)
print("    \---SPLIT --> new key   ", key, "--> normal", random.normal(key, shape=(1,)) )
print("             \--> new subkey", subkey, "--> normal", random.normal(subkey, shape=(1,)) )
```

+++ {"id": "QgCCZtyQ4EqA"}

You can also generate multiple subkeys at once if needed:

```{code-cell}
:id: G3zRojMs4Cce
:outputId: e48e1ed0-4f16-49cb-dc2b-cb51d3ec56b5

key, *subkeys = random.split(key, 4)
key, subkeys
```

+++ {"id": "20lC7np5YKDq"}

You can think about those PRNGs as trees of keys that match the structure of your models, which is important for reproducibility and soundness of the random behavior that you expect.

+++ {"id": "GC6-1gq1YsgZ"}

## Gradients and autodiff

For a full overview of JAX's automatic differentiation system, you can check the [Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).

Even though, theoretically, a VJP (Vector-Jacobian product - reverse autodiff) and a JVP (Jacobian-Vector product - forward-mode autodiff) are similar—they compute a product of a Jacobian and a vector—they differ by the computational complexity of the operation. In short, when you have a large number of parameters (hence a wide matrix), a JVP is less efficient computationally than a VJP, and, conversely, a JVP is more efficient when the Jacobian matrix is a tall matrix. You can read more in the JAX [cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff) [notebook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff) mentioned above.

+++ {"id": "CUFwVnn4011l"}

### Gradients

JAX provides first-class support for gradients and automatic differentiation in functions. This is also where the functional paradigm shines, since gradients on functions are essentially stateless operations. If we consider a simple function $f:\mathbb{R}^n\rightarrow\mathbb{R}$

$$f(x) = \frac{1}{2} x^T x$$

with the (known) gradient:

$$\nabla f(x) = x$$

```{code-cell}
:id: zDOydrLMcIzp
:outputId: 580c14ed-d1a3-4f92-c9b9-78d58c87bc76

key = random.PRNGKey(0)
def f(x):
  return jnp.dot(x.T,x)/2.0

v = jnp.ones((4,))
f(v)
```

+++ {"id": "zVaiZplShoBK"}

JAX computes the gradient as an operator acting on functions with `jax.grad`. Note that this only works for scalar valued functions.

Let's take the gradient of f and make sure it matches the identity map.

```{code-cell}
:id: ael3pVHmhhTs
:outputId: 4d0c5122-1ead-4a94-9153-7eb3b399dae2

v = random.normal(key,(4,))
print("Original v:")
print(v)
print("Gradient of f taken at point v")
print(jax.grad(f)(v)) # should be equal to v !
```

+++ {"id": "UHIMfchIiQMR"}

As previously mentioned, `jax.grad` only works for scalar-valued functions. JAX can also handle general vector valued functions. The most useful primitives are a Jacobian-Vector product - `jax.jvp` - and a Vector-Jacobian product - `jax.vjp`.

### Jacobian-Vector product

Let's consider a map $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$. As a reminder, the differential of f is the map $df:\mathbb{R}^n \rightarrow \mathcal{L}(\mathbb{R}^n,\mathbb{R}^m)$ where $\mathcal{L}(\mathbb{R}^n,\mathbb{R}^m)$ is the space of linear maps from $\mathbb{R}^n$ to $\mathbb{R}^m$ (hence $df(x)$ is often represented as a Jacobian matrix). The linear approximation of f at point $x$ reads:

$$f(x+v) = f(x) + df(x)\bullet v + o(v)$$

The $\bullet$ operator means you are applying the linear map $df(x)$ to the vector v.

Even though you are rarely interested in computing the full Jacobian matrix representing the linear map $df(x)$ in a standard basis, you are often interested in the quantity $df(x)\bullet v$. This is exactly what `jax.jvp` is for, and `jax.jvp(f, (x,), (v,))` returns the tuple:

$$(f(x), df(x)\bullet v)$$

+++ {"id": "F5nI_gbeqj2y"}

Let's use a simple function as an example: $f(x) = \frac{1}{2}({x_1}^2, {x_2}^2, \ldots, {x_n}^2)$ where we know that $df(x)\bullet h = (x_1h_1, x_2h_2,\ldots,x_nh_n)$. Hence using `jax.jvp` with $h= (1,1,\ldots,1)$ should return $x$ as an output.

```{code-cell}
:id: Q2ntaHBeh-5u
:outputId: 93591ad3-832f-4928-c1f8-073cc3b7aae7

def f(x):
  return jnp.multiply(x,x)/2.0

x = random.normal(key, (5,))
v = jnp.ones(5)
print("(x,f(x))")
print((x,f(x)))
print("jax.jvp(f, (x,),(v,))")
print(jax.jvp(f, (x,),(v,)))
```

+++ {"id": "gdm_TTDLal_X"}

### Vector-Jacobian product
Keeping our $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ it's often the case (for example, when you are working with a scalar loss function) that you are interested in the composition $x\rightarrow\phi\circ f(x)$ where $\phi :\mathbb{R}^m\rightarrow\mathbb{R}$. In that case, the gradient reads:

$$\nabla(\phi\circ f)(x) = J_f(x)^T\nabla\phi(f(x))$$

Where $J_f(x)$ is the Jacobian matrix of f evaluated at x, meaning that $df(x)\bullet v = J_f(x)v$.

`jax.vjp(f,x)` returns the tuple:

$$(f(x),v\rightarrow v^TJ_f(x))$$

Keeping the same example as previously, using $v=(1,\ldots,1)$, applying the VJP function returned by JAX should return the $x$ value:

```{code-cell}
:id: _1VTl9zXqsFl
:outputId: f3f143a9-b1f1-4a4d-e4b1-c24a0fa114b8

(val, jvp_fun) = jax.vjp(f,x)
print("x = ", x)
print("v^T Jf(x) = ", jvp_fun(jnp.ones((5,)))[0])
```

+++ {"id": "2v1Uq_XlzRZS"}

## Accelerating code with jit & ops vectorization
We borrow the following example from the [JAX quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

---

+++ {"id": "kF04t9L71dhH"}

### Jit

JAX uses the XLA compiler under the hood, and enables you to jit compile your code to make it faster and more efficient. This is the purpose of the @jit annotation.

```{code-cell}
:id: D6p_wQ9xeIiu
:outputId: af7ea5af-5ee1-4aa5-d8d7-8f6a20da2b0e

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

v = random.normal(key, (1000000,))
%timeit selu(v).block_until_ready()
```

+++ {"id": "Nk9LVX580j6M"}

Now using the jit annotation (or function here) to speed things up:

```{code-cell}
:id: us5pWySG0jWL
:outputId: e8ff3b7b-3917-40fc-8f29-eb9e6df262e5

selu_jit = jax.jit(selu)
%timeit selu_jit(v).block_until_ready()
```

+++ {"id": "6kQyCgo407oF"}

jit compilation can be used along with autodiff in the code transparently.

---
### Vectorization

Finally, JAX enables you to write code that applies to a single example, and then vectorize it to manage transparently batching dimensions.

```{code-cell}
:id: j-E6MsKF0tmZ
:outputId: bfa377e8-92ee-4473-abd4-8d52338e2cc5

mat = random.normal(key, (15, 10))
batched_x = random.normal(key, (5, 10)) # Batch size on axis 0
single = random.normal(key, (10,))

def apply_matrix(v):
  return jnp.dot(mat, v)

print("Single apply shape: ", apply_matrix(single).shape)
print("Batched example shape: ", jax.vmap(apply_matrix)(batched_x).shape)
```

+++ {"id": "S2BcA8wm2_FW"}

## Full example: linear regression

Let's implement one of the simplest models using everything we have seen so far: a linear regression. From a set of data points $\{(x_i,y_i), i\in \{1,\ldots, k\}, x_i\in\mathbb{R}^n,y_i\in\mathbb{R}^m\}$, we try to find a set of parameters $W\in \mathcal{M}_{m,n}(\mathbb{R}), b\in\mathbb{R}^m$ such that the function $f_{W,b}(x)=Wx+b$ minimizes the mean squared error:

$$\mathcal{L}(W,b)\rightarrow\frac{1}{k}\sum_{i=1}^{k} \frac{1}{2}\|y_i-f_{W,b}(x_i)\|^2_2$$

(Note: depending on how you cast the regression problem you might end up with different setups. Theoretically we should be minimizing the expectation of the loss wrt to the data distribution, however for the sake of simplicity here we consider only the sampled loss).

```{code-cell}
:id: 5W9p_zVe2Cj-

# Linear feed-forward.
def predict(W, b, x):
  return jnp.dot(x, W) + b

# Loss function: Mean squared error.
def mse(W, b, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x, y):
    y_pred = predict(W, b, x)
    return jnp.inner(y-y_pred, y-y_pred) / 2.0
  # We vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
```

```{code-cell}
:id: qMkIxjjsduPY

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = predict(W, b, x_samples) + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)
```

```{code-cell}
:id: 5L2np6wve_xp
:outputId: 9db5c834-d7da-4291-d1ec-d4c39008d5ed

# Initialize estimated W and b with zeros.
W_hat = jnp.zeros_like(W)
b_hat = jnp.zeros_like(b)

# Ensure we jit the largest-possible jittable block.
@jax.jit
def update_params(W, b, x, y, lr):
  W, b = W - lr * jax.grad(mse, 0)(W, b, x, y), b - lr * jax.grad(mse, 1)(W, b, x, y)
  return W, b

learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse(W, b, x_samples, y_samples))
for i in range(101):
  # Perform one gradient update.
  W_hat, b_hat = update_params(W_hat, b_hat, x_samples, y_samples, learning_rate)
  if (i % 5 == 0):
    print(f"Loss step {i}: ", mse(W_hat, b_hat, x_samples, y_samples))
```

+++ {"id": "bJGKunxNzrxa"}

This is obviously an approximate solution to the linear regression problem (solving it would require a bit more work!), but here you have all the tools you would need if you wanted to do it the proper way.

+++ {"id": "bQXmL86aUS9x"}

## Refining a bit with pytrees

Here we're going to elaborate on our previous example using JAX pytree data structure.

+++ {"id": "zZMUvyCgUzby"}

### Pytrees basics

The JAX ecosystem uses pytrees everywhere and we do as well in Flax (the previous FrozenDict example is one, we'll get back to this). For a complete overview, we suggest that you take a look at the [pytree page](https://jax.readthedocs.io/en/latest/pytrees.html) from JAX's doc:

*In JAX, a pytree is a container of leaf elements and/or more pytrees. Containers include lists, tuples, and dicts (JAX can be extended to consider other container types as pytrees, see Extending pytrees below). A leaf element is anything that’s not a pytree, e.g. an array. In other words, a pytree is just a possibly-nested standard or user-registered Python container. If nested, note that the container types do not need to match. A single “leaf”, i.e. a non-container object, is also considered a pytree.*

```python
[1, "a", object()] # 3 leaves: 1, "a" and object()

(1, (2, 3), ()) # 3 leaves: 1, 2 and 3

[1, {"k1": 2, "k2": (3, 4)}, 5] # 5 leaves: 1, 2, 3, 4, 5
```

JAX provides a few utilities to work with pytrees that live in the `tree_util` package.

```{code-cell}
:id: 9SNY5eA1UdkJ

from jax import tree_util

t = [1, {"k1": 2, "k2": (3, 4)}, 5]
```

+++ {"id": "LujWjwVQUeea"}

You will often come across `tree_map` function that maps a function f to a tree and its leaves. We used it in the previous section to display the shapes of the model's parameters.

```{code-cell}
:id: szDhssVBUjTa
:outputId: 9ae4ebf1-a3c4-4ecb-b3df-67c8450310f8

tree_util.tree_map(lambda x: x*x, t)
```

+++ {"id": "3s167WGKUlZ9"}

Instead of applying a standalone function to each of the tree leaves, you can also provide a tuple of additional trees with similar shape to the input tree that will provide per leaf arguments to the function.

```{code-cell}
:id: bNOYK_E7UnOh
:outputId: d211bf85-5993-488c-9fec-aeaf375df007

t2 = tree_util.tree_map(lambda x: x*x, t)
tree_util.tree_map(lambda x,y: x+y, t, t2)
```

+++ {"id": "HnE75pvlVDO5"}

### Linear regression with Pytrees

Whereas our previous example was perfectly fine, we can see that when things get more complicated (as they will with neural networks), it will be harder to manage parameters of the models as we did.

Here we show an alternative based on pytrees, using the same data from the previous example.
Now, our `params` is a pytree containing both the `W` and `b` entries.

```{code-cell}
:id: 8v8gNkvUVZnl

# Linear feed-forward that takes a params pytree.
def predict_pytree(params, x):
  return jnp.dot(x, params['W']) + params['b']

# Loss function: Mean squared error.
def mse_pytree(params, x_batched,y_batched):
  # Define the squared loss for a single pair (x,y)
  def squared_error(x,y):
    y_pred = predict_pytree(params, x)
    return jnp.inner(y-y_pred, y-y_pred) / 2.0
  # We vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

# Initialize estimated W and b with zeros. Store in a pytree.
params = {'W': jnp.zeros_like(W), 'b': jnp.zeros_like(b)}
```

+++ {"id": "rKP0X8rnWAiA"}

The great thing is that JAX is able to handle differentiation with respect to pytree parameters:

```{code-cell}
:id: 8zc7cMaiWSny
:outputId: a69605cb-1eed-4f81-fc2e-93646c9694dd

jax.grad(mse_pytree)(params, x_samples, y_samples)
```

+++ {"id": "nW1IKnjqXFdN"}

Now using our tree of params, we can write the gradient descent in a simpler way using `jax.tree_util.tree_map`:

```{code-cell}
:id: jEntdcDBXBCj
:outputId: f309aff7-2aad-453f-ad88-019d967d4289

# Always remember to jit!
@jax.jit
def update_params_pytree(params, learning_rate, x_samples, y_samples):
  params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params,
        jax.grad(mse_pytree)(params, x_samples, y_samples))
  return params

learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse_pytree({'W': W, 'b': b}, x_samples, y_samples))
for i in range(101):
  # Perform one gradient update.
  params = update_params_pytree(params, learning_rate, x_samples, y_samples)
  if (i % 5 == 0):
    print(f"Loss step {i}: ", mse_pytree(params, x_samples, y_samples))
```

Besides `jax.grad()`, another useful function is `jax.value_and_grad()`, which returns the value of the input function and of its gradient.

To switch from `jax.grad()` to `jax.value_and_grad()`, replace the training loop above with the following:

```{code-cell}
---
vscode:
  languageId: python
---
# Using jax.value_and_grad instead:
loss_grad_fn = jax.value_and_grad(mse_pytree)
for i in range(101):
  # Note that here the loss is computed before the param update.
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads)
    if (i % 5 == 0):
        print(f"Loss step {i}: ", loss_val)
```

+++ {"id": "Xh-oo8jFUPNQ"}

That's all you needed to know to get started with Flax! To dive deeper, we very much recommend checking the JAX [docs](https://jax.readthedocs.io/en/latest/index.html).
