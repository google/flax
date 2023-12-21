# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset

from flax.experimental import nnx

np.random.seed(42)
image_shape: tp.Sequence[int] = (28, 28)
steps_per_epoch: int = 200
batch_size: int = 64
epochs: int = 20


@jax.custom_vjp
def diff_round(x) -> jax.Array:
  y = jnp.round(x)
  return y


def diff_round_fwd(x):
  return diff_round(x), None


def diff_round_bwd(_, g):
  return (g,)


diff_round.defvjp(diff_round_fwd, diff_round_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def diff_clip(x, low, high) -> jax.Array:
  return jnp.clip(x, low, high)


def diff_clip_fwd(x, low, high):
  return diff_clip(x, low, high), None


def diff_clip_bwd(_, _1, _2, dy):
  return (dy,)


diff_clip.defvjp(diff_clip_fwd, diff_clip_bwd)


# %%
def f(x):
  return diff_clip(diff_round(x * 128) + 128, 0, 255)


df = jax.vmap(jax.grad(f))

x = jnp.linspace(-1.5, 1.5, 100)
dx = df(x)

plt.plot(x, dx)

# %%
dataset = load_dataset('mnist')
X_train = np.array(np.stack(dataset['train']['image']), dtype=np.float32)
Y_train = np.array(dataset['train']['label'], dtype=np.int32)
X_test = np.array(np.stack(dataset['test']['image']), dtype=np.float32)
Y_test = np.array(dataset['test']['label'], dtype=np.int32)
# normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


print('X_train:', X_train.shape, X_train.dtype)
print('X_test:', X_test.shape, X_test.dtype)


# %%
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.reshape((x.shape[0], -1))
    x = self.linear1(x)
    x = jax.nn.gelu(x)
    x = self.linear2(x)
    return x


params, static = MLP(
  din=np.prod(image_shape), dmid=256, dout=10, rngs=nnx.Rngs(0)
).split(nnx.Param)

state = nnx.TrainState(
  static,
  params=params,
  tx=optax.adam(1e-3),
)


# %%
@jax.jit
def train_step(
  state: nnx.TrainState[MLP],
  inputs: jax.Array,
  labels: jax.Array,
):
  def loss_fn(params: nnx.State):
    logits, _ = state.apply(params)(inputs)
    loss = jnp.mean(
      optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )
    return loss

  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)

  return state, loss


@jax.jit
def eval_step(state: nnx.TrainState[MLP], inputs: jax.Array, labels: jax.Array):
  logits, _ = state.apply('params')(inputs)
  loss = jnp.mean(
    optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  )
  acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  return {'loss': loss, 'accuracy': acc}


@partial(jax.jit, donate_argnums=(0,))
def forward(state: nnx.TrainState[MLP], inputs: jax.Array) -> jax.Array:
  y_pred = state.apply('params')(inputs)[0]
  return jnp.argmax(y_pred, axis=-1)


# %%
key = jax.random.key(0)

for epoch in range(epochs):
  for step in range(steps_per_epoch):
    idxs = np.random.randint(0, len(X_train), size=(batch_size,))
    x_batch = X_train[idxs]
    y_batch = Y_train[idxs]

    state, loss = train_step(state, x_batch, y_batch)

  metrics = eval_step(state, X_test, Y_test)
  metrics = jax.tree_map(lambda x: x.item(), metrics)
  print(f'Epoch {epoch} - {metrics}')

# %%
# get random samples
idxs = np.random.randint(0, len(X_test), size=(10,))
x_sample = X_test[idxs]
y_sample = Y_test[idxs]

# get predictions
y_pred = forward(state, x_sample)

# plot predictions
figure = plt.figure(figsize=(3 * 5, 3 * 2))

for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(x_sample[i].reshape(image_shape), cmap='gray')
  plt.title(f'{y_pred[i]}')

plt.show()

model = state.graphdef.merge(state.params)
# %%
# Quantization

A = tp.TypeVar('A')


class QParam(nnx.Variable[A]):
  pass


class QHParam(nnx.Variable[A]):
  pass


class QLinear(nnx.Module):
  def __init__(self, din: int, dout: int):
    self.scale = QHParam(jnp.array(0.5))
    self.zero_point = QHParam(jnp.array(0.5))
    self.qkernel = QParam(jnp.zeros((din, dout)))
    self.qbias = QParam(jnp.zeros((dout,)))

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.quantize(x, 8, jnp.uint8)
    print(x.shape, self.qkernel.shape, self.qbias.shape)
    x = jnp.dot(x, self.qkernel, preferred_element_type=jnp.uint16)
    x = (x + self.qbias).astype(jnp.uint32)
    x = self.dequantize(x)
    return x

  def quantize(self, x: jax.Array, b: int, dtype: jnp.dtype) -> jax.Array:
    return jnp.clip(
      diff_round(x / self.scale) + self.zero_point, 0, 2**b - 1
    ).astype(dtype)

  def dequantize(self, x: jax.Array) -> jax.Array:
    return (x - self.zero_point) * self.scale

  def optimize(
    self,
    pretrained: nnx.Linear,
    x: jax.Array,
    *,
    num_steps: int = 100,
    debug: bool = False,
  ):
    q_hparams, rest, static = self.split(QHParam, ...)
    tx = optax.adam(1e-3)
    opt_state = tx.init(q_hparams)

    print(jax.tree_map(lambda x: x.shape, q_hparams))

    @jax.jit
    def optimization_step(
      q_hparams: nnx.State,
      rest: nnx.State,
      opt_state: optax.OptState,
      x: jax.Array,
    ):
      print('JITTING')

      def loss_fn(q_hparams: nnx.State):
        model = static.merge(q_hparams, rest)
        model.qkernel = model.quantize(pretrained.kernel, 8, jnp.uint8)
        assert pretrained.bias is not None
        model.qbias = model.quantize(pretrained.bias, 16, jnp.uint16)

        y_quant = model(x)
        y_unquant = pretrained(x)
        loss = jnp.mean((y_unquant - y_quant) ** 2)
        return loss

      loss, grads = jax.value_and_grad(loss_fn)(q_hparams)

      updates, opt_state = tx.update(grads, opt_state, q_hparams)
      q_hparams = optax.apply_updates(q_hparams, updates)  # type: ignore

      return q_hparams, opt_state, loss

    for step in range(num_steps):
      q_hparams, opt_state, loss = optimization_step(
        q_hparams, rest, opt_state, x
      )
      if debug and step % (num_steps / 10) == 0:
        print(f'Step {step} - loss: {loss}')

    self.update(q_hparams)

    self.qkernel = self.quantize(pretrained.kernel, 8, jnp.uint8)
    assert pretrained.bias is not None
    self.qbias = self.quantize(pretrained.bias, 16, jnp.uint16)


def optimize2(
  self,
  pretrained: nnx.Linear,
  X: jax.Array,
):
  W = pretrained.kernel
  b = pretrained.bias
  assert b is not None

  # X
  alpha_X = jnp.min(X)
  beta_X = jnp.max(X)
  s_X, z_X = generate_quantization_int8_constants(alpha=alpha_X, beta=beta_X)
  X_q = quantization_int8(x=X, s=s_X, z=z_X)
  X_q_dq = dequantization(x_q=X_q, s=s_X, z=z_X)

  # W
  alpha_W = jnp.min(W)
  beta_W = jnp.max(W)
  s_W, z_W = generate_quantization_int8_constants(alpha=alpha_W, beta=beta_W)
  W_q = quantization_int8(x=W, s=s_W, z=z_W)
  W_q_dq = dequantization(x_q=W_q, s=s_W, z=z_W)

  # b
  alpha_b = jnp.min(b)
  beta_b = jnp.max(b)
  s_b, z_b = generate_quantization_int8_constants(alpha=alpha_b, beta=beta_b)
  b_q = quantization_int8(x=b, s=s_b, z=z_b)
  b_q_dq = dequantization(x_q=b_q, s=s_b, z=z_b)

  # Y
  Y = jnp.matmul(X, W) + b
  alpha_Y = jnp.min(Y)
  beta_Y = jnp.max(Y)
  s_Y, z_Y = generate_quantization_int8_constants(alpha=alpha_Y, beta=beta_Y)
  Y_q = quantization_int8(x=Y, s=s_Y, z=z_Y)

  Y_prime = jnp.matmul(X_q_dq, W_q_dq) + b_q_dq
  Y_prime_q = quantization_int8(x=Y_prime, s=s_Y, z=z_Y)
  Y_prime_q_dq = dequantization(x_q=Y_prime_q, s=s_Y, z=z_Y)

  print('Expected FP32 Y:')
  print(Y)
  print('Expected FP32 Y Quantized:')
  print(Y_q)

  Y_q_simulated = quantization_matrix_multiplication_int8(
    X_q=X_q,
    W_q=W_q,
    b_q=b_q,
    s_X=s_X,
    z_X=z_X,
    s_W=s_W,
    z_W=z_W,
    s_b=s_b,
    z_b=z_b,
    s_Y=s_Y,
    z_Y=z_Y,
  )
  Y_simulated = dequantization(x_q=Y_q_simulated, s=s_Y, z=z_Y)

  print('Expected Quantized Y_q from Quantized Matrix Multiplication:')
  print(Y_q_simulated)
  print(
    'Expected Quantized Y_q from Quantized Matrix Multiplication Dequantized:'
  )
  print(Y_simulated)

  # Ensure the algorithm implementation is correct
  assert jnp.array_equal(Y_simulated, Y_prime_q_dq)
  assert jnp.array_equal(Y_q_simulated, Y_prime_q)


def quantization(x, s, z, alpha_q, beta_q):
  x_q = jnp.round(1 / s * x + z, decimals=0)
  x_q = jnp.clip(x_q, a_min=alpha_q, a_max=beta_q)

  return x_q


def quantization_int8(x, s, z):
  x_q = quantization(x, s, z, alpha_q=-128, beta_q=127)
  x_q = x_q.astype(jnp.int8)

  return x_q


def dequantization(x_q, s, z):
  # x_q - z might go outside the quantization range.
  x_q = x_q.astype(jnp.int32)
  x = s * (x_q - z)
  x = x.astype(jnp.float32)

  return x


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):
  # Affine quantization mapping
  s = (beta - alpha) / (beta_q - alpha_q)
  z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

  return s, z


def generate_quantization_int8_constants(alpha, beta):
  b = 8
  alpha_q = -(2 ** (b - 1))
  beta_q = 2 ** (b - 1) - 1

  s, z = generate_quantization_constants(
    alpha=alpha, beta=beta, alpha_q=alpha_q, beta_q=beta_q
  )

  return s, z


def quantization_matrix_multiplication_int8(
  X_q, W_q, b_q, s_X, z_X, s_W, z_W, s_b, z_b, s_Y, z_Y
):
  p = W_q.shape[0]

  # Y_q_simulated is FP32
  Y_q_simulated = (
    z_Y
    + (s_b / s_Y * (b_q.astype(jnp.int32) - z_b))
    + (
      (s_X * s_W / s_Y)
      * (
        jnp.matmul(X_q.astype(jnp.int32), W_q.astype(jnp.int32))
        - z_W * jnp.sum(X_q.astype(jnp.int32), axis=1, keepdims=True)
        - z_X * jnp.sum(W_q.astype(jnp.int32), axis=0, keepdims=True)
        + p * z_X * z_W
      )
    )
  )

  Y_q_simulated = jnp.round(Y_q_simulated, decimals=0)
  Y_q_simulated = jnp.clip(Y_q_simulated, a_min=-128, a_max=127)
  Y_q_simulated = Y_q_simulated.astype(jnp.int8)

  return Y_q_simulated


# %%
qlinear1 = QLinear(din=np.prod(image_shape), dout=256)
# qlinear2 = QLinear(din=256, dout=10)

idxs = np.random.randint(0, len(X_test), size=(100,))
x_optimize = jnp.asarray(X_test[idxs], dtype=jnp.float32)
x_optimize = x_optimize.reshape((x_optimize.shape[0], -1))
print(x_optimize.shape)
qlinear1.optimize(model.linear1, x_optimize, num_steps=1000, debug=True)

# %%
