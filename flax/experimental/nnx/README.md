[![codecov](https://codecov.io/gh/cgarciae/nnx/branch/main/graph/badge.svg?token=VqJjL474Z7)](https://codecov.io/gh/cgarciae/nnx)

# NNX

_**N**eural **N**etworks for JA**X**_ - | [docs](https://flax.readthedocs.io/en/latest/experimental/nnx/index.html) |

NNX is a JAX-based neural network library designed for simplicity and power. Its modular approach follows standard Python conventions, making it both intuitive and compatible with the broader JAX ecosystem.

* **Pythonic**: Modules are standard Python classes, promoting ease of use and a more familiar
  development experience.
* **Compatible**: Effortlessly convert between Modules and pytrees using the Functional API for maximum flexibility.
* **Control**: Manage a Module's state with precision using typed Variable collections, enabling fine-grained control
  on JAX transformations.
* **User-friendly**: NNX prioritizes simplicity for common use cases, building upon lessons learned from Linen
  to provide a streamlined experience.

> [!NOTE]
> NNX is currently in an experimental state and is subject to change. Linen is still the
   recommended option for large-scale projects. Feedback and contributions are welcome!


## Installation

To get started with `nnx`, install Flax from GitHub:
```
pip install git+https://github.com/google/flax.git
```

## What does NNX look like?

We provide three examples using the NNX API: a simple multi-layer perceptron, a CNN and an auto-encoder.

To learn more about the `Module` abstraction, check out our [NNX Basics](https://flax.readthedocs.io/en/latest/experimental/nnx/nnx_basics.html#) guide.

```python
import jax
import jax.numpy as jnp

from flax.experimental import nnx


class MLP(nnx.Module):
  def __init__(self, features: list[int], *, rngs: nnx.Rngs):
    self.layers = [
      nnx.Linear(din, dout, rngs=rngs)
      for din, dout in zip(features[:-1], features[1:])
    ]

  def __call__(self, x: jax.Array) -> jax.Array:
    for layer in self.layers[:-1]:
      x = nnx.relu(layer(x))
    x = self.layers[-1](x)
    return x


model = MLP([784, 64, 32, 10], rngs=nnx.Rngs(0))
y = model(jnp.ones((1, 784)))
```

```python
class CNN(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(64, 32, kernel_size=(3, 3), rngs=rngs)
    self.linear1 = nnx.Linear(7 * 7 * 32, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.conv1(x))
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nnx.relu(self.conv2(x))
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nnx.relu(self.linear1(x))
    logits = self.linear2(x)
    return logits


model = CNN(rngs=nnx.Rngs(0))
x = jnp.ones((1, 28, 28, 1))  # (N, H, W, C) format
logits = model(x)
```

```python
class AutoEncoder(nnx.Module):
  def __init__(
    self,
    input_features: int,
    encoder_features: list[int],
    decoder_features: list[int],
    *,
    rngs: nnx.Rngs,
  ):
    self.encoder = MLP([input_features, *encoder_features], rngs=rngs)
    self.decoder = MLP([*decoder_features, input_features], rngs=rngs)

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z):
    return nnx.sigmoid(self.decoder(z))


model = AutoEncoder(
  input_features=784,
  encoder_features=[64, 32],
  decoder_features=[32, 64],
  rngs=nnx.Rngs(0),
)
x = jnp.ones((1, 784))
z = model.encode(x)
y = model.decode(z)
```

### Interacting with JAX

To interact with JAX NNX provides the [Functional API](https://flax.readthedocs.io/en/latest/experimental/nnx/nnx_basics.html#the-functional-api) which consists of 3 simple methods: `split`, `merge`, and `update`. Using these methods any Module can be lifted to be used in JAX transformations. Here is a simple jitted `forward` function as an example:

```python
state, static = model.split()

@jax.jit
def forward(static: nnx.GraphDef, state: nnx.State, x: jax.Array):
  model = static.merge(state)
  y = model(x)
  state, _ = model.split()
  return y, state

x = jnp.ones((2, 4))
y, state = forward(static, state, x)

model.update(state)
```

### Examples

* [LM1B](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/lm1b): A language model trained on the 1 Billion Word Benchmark dataset.

#### Toy Examples
* [Using the Functional API](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/01_functional_api.py): Shows how to train a simple model using the functional API.
* [Using Lifted Transforms](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/02_lifted_transforms.py): Shows how to train a simple model using lifted transforms.
* [Using TrainState](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/03_train_state.py): Shows how to train a simple model using the functional API with the help of `TrainState`.
* [Training a VAE](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/05_vae.py): Shows how to train a VAE on the binarized MNIST dataset, uses the functional API, `TrainState`, and shows how to use capture intermediate values to retrieve `kl_loss`.
* [Scan over layers](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/06_scan_over_layers.py): An contrived example that implements scan over layers with dropout and a share BatcNorm layer to showcase how lifted transforms can be implemented. It uses the functional API along with `jax.vmap` and `jax.lax.scan`.
* [Creating a Transformer](https://github.com/google/flax/tree/main/flax/experimental/nnx/examples/toy_examples/07_transformer.py): Shows how to create a Transformer with an auto-regressive decoder that uses scan over layers and a kv-cache for fast inference. Credits to @levskaya.
