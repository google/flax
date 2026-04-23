From PyTorch to Jax and Flax
============================

This guide is written for those who are familiar with PyTorch and covers the essentials of Jax and Flax.

When moving from PyTorch to Jax and Flax, there is a paradigm shift to take into account:

- PyTorch is inherently eager, object-oriented and mutable.

- Jax is functional, immutable and relies on Just-In-Time (JIT) compilation.

- Flax NNX allows pythonic, mutable objects like in PyTorch but provides mechanisms like ``nnx.split`` and ``nnx.merge`` to safely cross Jax's functional boundaries.


Quickstart with Jax Arrays
--------------------------

This part provides a quick recap on Jax Arrays manipulation compared to PyTorch tensors.
Let's first create a Jax Array from data:

.. tab-set::

  .. tab-item:: Jax
    :sync: Jax

    .. code-block:: python

      import jax
      import jax.numpy as jnp

      data = [[1, 2, 3], [3, 4, 5]]
      x = jnp.array(data)
      assert isinstance(x, jax.Array)
      print(x, x.shape, x.dtype, x.device)
      # [[1 2 3]
      #  [3 4 5]] (2, 3) int32 cuda:0

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      import torch

      data = [[1, 2, 3], [3, 4, 5]]
      x = torch.tensor(data)
      assert isinstance(x, torch.Tensor)
      print(x, x.shape, x.dtype, x.device)
      # tensor([[1, 2, 3],
      #         [3, 4, 5]]) torch.Size([2, 3]) torch.int64 cpu

We can see that Jax Array is allocated on the default device e.g. GPU if available vs CPU for PyTorch tensor.
Jax Array of integer data has int32 dtype vs int64 for PyTorch tensor.

We can initialize arrays with constants or random values:

.. tab-set::

  .. tab-item:: Jax
    :sync: Jax

    .. code-block:: python

      shape = (2, 3)
      ones_array = jnp.ones(shape)
      zeros_array = jnp.zeros(shape)
      rand_array = jax.random.uniform(jax.random.key(123), shape)

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      shape = (2, 3)
      ones_tensor = torch.ones(shape)
      zeros_tensor = torch.zeros(shape)
      rand_tensor = torch.rand(shape)

Jax avoids implicit global random state and instead tracks state
explicitly via a random key. If we create two random arrays using
the same key we will obtain two identical random arrays. We can also
split the random key into multiple keys to create two different
random arrays:

.. code-block:: python

  key = jax.random.key(124)
  rand_tensor1 = jax.random.uniform(key, (2, 3))
  rand_tensor2 = jax.random.uniform(key, (2, 3))
  assert (rand_tensor1 == rand_tensor2).all()

  k1, k2 = jax.random.split(key, num=2)
  rand_tensor1 = jax.random.uniform(k1, (2, 3))
  rand_tensor2 = jax.random.uniform(k2, (2, 3))
  assert (rand_tensor1 != rand_tensor2).all()


For further discussion on random numbers in NumPy and Jax check `this tutorial <https://docs.jax.dev/en/latest/random-numbers.html>`__.

Finally, we can initialize a Jax Array from a PyTorch tensor:

.. code-block:: python

  import torch

  x_torch = torch.rand(3, 4)

  # Create Jax Array as a copy of x_torch tensor
  x_jax = jnp.asarray(x_torch)
  assert isinstance(x_jax, jax.Array)
  print(x_jax, x_jax.shape, x_jax.dtype)

  # Use dlpack to create Jax Array without copying
  x_jax = jax.dlpack.from_dlpack(x_torch.to(device="cuda"), copy=False)
  print(x_jax, x_jax.shape, x_jax.dtype, x_jax.device)

There are some notable differences between PyTorch tensors and Jax Arrays:

- Jax Arrays are immutable
- The default integer and float dtypes are int32 and float32
- The default device corresponds to the available accelerator, e.g. cuda:0 if one or multiple GPUs are available.

As Jax Arrays are immutable, to write an equivalent of in-place expression
is possible using ``at`` property:

.. code-block:: python

  x_jax = x_jax.at[0].set(123)

For more examples of Jax's alternative to in-place mutations,
refer to `this Jax documentation <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html>`__


Devices and accelerators
^^^^^^^^^^^^^^^^^^^^^^^^

In Jax, we can check available devices as follows:

.. code-block:: python

  print(f"Available devices given a backend (gpu or tpu or cpu): {jax.devices()}")
  # Define CPU and CUDA devices
  cpu_device = jax.devices("cpu")[0]
  cuda_device = jax.devices("cuda")[0]
  print(cpu_device, cuda_device)

and create arrays on a specific device:

.. code-block:: python

  # create an array on CPU
  x_cpu = jnp.ones((3, 4), device=cpu_device)
  # create an array on GPU
  x_gpu = jnp.ones((3, 4), device=cuda_device)

  # Or using a context manager:
  print(x_cpu.device, x_cpu.committed)
  with jax.default_device("cpu"):
    x_cpu = jnp.ones((3, 4))

  with jax.default_device("gpu"):
    x_gpu = jnp.ones((3, 4))


In PyTorch tensor device placement is always being explicit.
Jax can operate this way via explicit device placement as above,
but unless the device is specified the array will remain uncommitted:
i.e. it will be stored on the default device, but allow implicit movement
to other devices when necessary:

.. code-block:: python

  x_cpu = jnp.ones((3, 4), device=cpu_device)
  print(x_cpu.device, x_cpu.committed)
  # TFRT_CPU_0 True
  x = jnp.ones((3, 4))
  print(x.device, x.committed, (x_cpu + x).device)
  # cuda:0 False TFRT_CPU_0

However, if we make a computation with two arrays with
explicitly specified devices, e.g. CPU and CUDA, similarly to PyTorch,
an error will be raised:

.. code-block:: python

  x_cpu = jnp.ones((3, 4), device=cpu_device)
  x_gpu = jnp.ones((3, 4), device=cuda_device)
  x_cpu + x_gpu  # Raises an error

Finally, to move from one device to another, we can use ``jax.device_put``
function:

.. code-block:: python

  x = jnp.ones((3, 4))
  x_cpu = jax.device_put(x, device=jax.devices("cpu")[0])
  x_cuda = jax.device_put(x_cpu, device=jax.devices("cuda")[0])
  print(f"{x.device} -> {x_cpu.device} -> {x_cuda.device}")


Operations on Jax Arrays and JIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a large list of operations (arithmetics, linear algebra, matrix manipulation, etc) that can be directly performed on Jax Arrays.
Jax API contains important modules:

- ``jax.numpy`` provides NumPy-like functions
- ``jax.scipy`` provides SciPy-like functions
- ``jax.nn`` provides common functions for neural networks: activations, softmax, one-hot encoding etc
- ``jax.lax`` provides low-level XLA APIs
- ...

More details on available ops can be found in the `API reference <https://docs.jax.dev/en/latest/jax.html>`__.

Jax relies on ``jax.jit`` transformation to provide the most efficient computation performance.
It performs JIT compilation of a Python function for efficient execution in XLA.
Behind the scenes, ``jax.jit`` wraps the input into tracers and is tracing the function to record all Jax operations.
By default, Jax JIT is compiling the function on the first call and reusing the cached compiled XLA code on subsequent
calls.

PyTorch users may think of ``jax.jit`` as roughly equivalent to TorchScript introduced to optimize and serialize PyTorch
models by capturing the execution graph into TorchScript programs, which can then be run independently from Python,
e.g. as a C++ program.

However, Jax requires all output arrays and intermediate arrays to have static shape: that is,
the shape cannot depend on values within other arrays. It will trigger recompilation if shapes have changed.
One can easily track the recompilation using ``jax.config.update("jax_log_compiles", True)``.



Building Neural Networks
------------------------

In this section we will cover the translation from ``torch.nn.Module`` to ``nnx.Module``.
The Flax NNX module is very similar to PyTorch ``torch.nn`` module and we can map the
following modules between PyTorch and Flax NNX:

- ``nn.Sequential`` and ``nn.ModuleList`` ⇒ ``nnx.Sequential``
- ``nn.Linear`` ⇒ ``nnx.Linear``
- ``nn.Conv2d`` ⇒ ``nnx.Conv``
- ``nn.BatchNorm2d`` ⇒ ``nnx.BatchNorm``
- Activations like ``nn.ReLU`` ⇒ ``nnx.relu`` function
- Pooling layers like ``nn.AvgPool2d(...)`` ⇒ ``lambda x: nnx.avg_pool(x, ...)`` function
  - ``nn.AdaptiveAvgPool2d(1)`` ⇒ ``lambda x: nnx.avg_pool(x, (x.shape[1], x.shape[2]))``, ``x`` is in NHWC format
- ``nn.Flatten()`` ⇒ ``lambda x: x.reshape(x.shape[0], -1)``

See :ref:`the next section <pytorch-vs-nnx>` for a detailed comparison between PyTorch and NNX layers.

Here is an example of a simple neural network implementation in NNX vs PyTorch:

.. tab-set::

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      import flax.nnx as nnx


      class Model(nnx.Module):
        def __init__(self, n: int, m: int, h: int, rngs: nnx.Rngs):
          self.linear1 = nnx.Linear(n, h, rngs=rngs)
          self.act = nnx.relu
          self.linear2 = nnx.Linear(h, m, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
          x = self.linear1(x)
          x = self.act(x)
          return self.linear2(x)


      model = Model(10, 10, 12, rngs=nnx.Rngs(12))
      out = model(jnp.ones((2, 10)))

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      import torch.nn as nn


      class Model(nn.Module):
        def __init__(self, n: int, m: int, h: int):
          super().__init__()
          self.linear1 = nn.Linear(n, h)
          self.act = nn.ReLU
          self.linear2 = nn.Linear(h, m)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
          x = self.linear1(x)
          x = self.act(x)
          return self.linear2(x)

      model = Model(10, 10, 12)
      out = model(torch.ones(2, 10))


If the PyTorch model defines a learnable parameter and a buffer,
an equivalent code in Flax would be

.. tab-set::

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      class Buffer(nnx.Variable):
        pass


      class Model(nnx.Module):
        def __init__(self, ...):
            ...
            self.p = nnx.Param(jnp.ones((10,)))
            self.b = Buffer(jnp.ones(5))

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      class Model(nn.Module):
        def __init__(self, ...):
            ...
            self.p = nn.Parameter(torch.ones(10))
            self.register_buffer("b", torch.ones(5))



.. _pytorch-vs-nnx:

.. _nnx_layers_comparison:

PyTorch layers vs NNX layers
----------------------------


.. testsetup::

  import numpy as np
  import jax
  from jax import random, numpy as jnp
  from flax import nnx

  import torch

In this part we will consider differences between PyTorch and NNX neural network layers.
We will cover convolutions, Fully-Connected (FC) layers, Batch Norm, and average pooling.

Fully-Connected Layers
^^^^^^^^^^^^^^^^^^^^^^

Let's start with Fully-Connected (FC) layers. The only thing to be aware of here is that the PyTorch kernel has shape [outC, inC]
and the Flax kernel has shape [inC, outC]. Transposing the kernel will do the trick.

.. testcode::

    t_fc = torch.nn.Linear(in_features=3, out_features=4)

    kernel = t_fc.weight.detach().cpu().numpy()
    bias = t_fc.bias.detach().cpu().numpy()

    # [outC, inC] -> [inC, outC]
    kernel = jnp.transpose(kernel, (1, 0))

    key = random.key(0)
    x = random.normal(key, (1, 3))

    j_fc = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
    j_fc.kernel.value = kernel
    j_fc.bias.value = jnp.array(bias)
    j_out = j_fc(x)

    t_x = torch.from_numpy(np.array(x))
    t_out = t_fc(t_x)
    t_out = t_out.detach().cpu().numpy()

    np.testing.assert_almost_equal(j_out, t_out, decimal=6)


Convolutions
^^^^^^^^^^^^

Let's now look at 2D convolutions. PyTorch uses the NCHW format and Flax uses NHWC.
Consequently, the kernels will have different shapes. The kernel in PyTorch has shape [outC, inC, kH, kW]
and the Flax kernel has shape [kH, kW, inC, outC]. Transposing the kernel will do the trick.

.. testcode::

    t_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding='valid')

    kernel = t_conv.weight.detach().cpu().numpy()
    bias = t_conv.bias.detach().cpu().numpy()

    # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
    kernel = jnp.transpose(kernel, (2, 3, 1, 0))

    key = random.key(0)
    x = random.normal(key, (1, 6, 6, 3))

    j_conv = nnx.Conv(3, 4, kernel_size=(2, 2), padding='valid', rngs=nnx.Rngs(0))
    j_conv.kernel.value = kernel
    j_conv.bias.value = bias
    j_out = j_conv(x)

    # [N, H, W, C] -> [N, C, H, W]
    t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
    t_out = t_conv(t_x)
    # [N, C, H, W] -> [N, H, W, C]
    t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))

    np.testing.assert_almost_equal(j_out, t_out, decimal=6)



Convolutions and FC Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^

We have to be careful, when we have a model that uses convolutions followed by fc layers (ResNet, VGG, etc).
In PyTorch, the activations will have shape [N, C, H, W] after the convolutions and are then
reshaped to [N, C * H * W] before being fed to the fc layers.
When we port our weights from PyTorch to Flax, the activations after the convolutions will be of shape [N, H, W, C] in Flax.
Before we reshape the activations for the fc layers, we have to transpose them to [N, C, H, W].

Consider this PyTorch model:

.. testcode::

  class TModel(torch.nn.Module):

    def __init__(self):
      super(TModel, self).__init__()
      self.conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding='valid')
      self.fc = torch.nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
      x = self.conv(x)
      x = x.reshape(x.shape[0], -1)
      x = self.fc(x)
      return x


  t_model = TModel()



Now, if you want to use the weights from this model in Flax, the corresponding Flax model has to look like this:


.. testcode::

    class JModel(nnx.Module):
        def __init__(self, rngs):
            self.conv = nnx.Conv(3, 4, kernel_size=(2, 2), padding='valid', rngs=rngs)
            self.linear = nnx.Linear(100, 2, rngs=rngs)

        def __call__(self, x):
            x = self.conv(x)
            # [N, H, W, C] -> [N, C, H, W]
            x = jnp.transpose(x, (0, 3, 1, 2))
            x = jnp.reshape(x, (x.shape[0], -1))
            x = self.linear(x)
            return x

    j_model = JModel(nnx.Rngs(0))



The model looks very similar to the PyTorch model, except that we included a transpose operation before
reshaping our activations for the fc layer.
We can omit the transpose operation if we apply pooling before reshaping such that the spatial dimensions are 1x1.

Other than the transpose operation before reshaping, we can convert the weights the same way as we did before:


.. testcode::

  conv_kernel = t_model.state_dict()['conv.weight'].detach().cpu().numpy()
  conv_bias = t_model.state_dict()['conv.bias'].detach().cpu().numpy()
  fc_kernel = t_model.state_dict()['fc.weight'].detach().cpu().numpy()
  fc_bias = t_model.state_dict()['fc.bias'].detach().cpu().numpy()

  # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
  conv_kernel = jnp.transpose(conv_kernel, (2, 3, 1, 0))

  # [outC, inC] -> [inC, outC]
  fc_kernel = jnp.transpose(fc_kernel, (1, 0))

  j_model.conv.kernel.value = conv_kernel
  j_model.conv.bias.value = conv_bias
  j_model.linear.kernel.value = fc_kernel
  j_model.linear.bias.value = fc_bias

  key = random.key(0)
  x = random.normal(key, (1, 6, 6, 3))
  j_out = j_model(x)

  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_model(t_x)
  t_out = t_out.detach().cpu().numpy()

  np.testing.assert_almost_equal(j_out, t_out, decimal=6)



Batch Norm
^^^^^^^^^^

``torch.nn.BatchNorm2d`` uses ``0.1`` as the default value for the ``momentum`` parameter while
|nnx.BatchNorm|_ uses ``0.9``. However, this corresponds to the same computation, because PyTorch multiplies
the estimated statistic with ``(1 - momentum)`` and the new observed value with ``momentum``,
while Flax multiplies the estimated statistic with ``momentum`` and the new observed value with ``(1 - momentum)``.

.. |nnx.BatchNorm| replace:: ``nnx.BatchNorm``
.. _nnx.BatchNorm: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm

.. testcode::

  t_bn = torch.nn.BatchNorm2d(num_features=3, momentum=0.1)
  t_bn.eval()

  key = random.key(0)
  x = random.normal(key, (1, 6, 6, 3))

  j_bn = nnx.BatchNorm(num_features=3, momentum=0.9, use_running_average=True, rngs=nnx.Rngs(0))

  j_out = j_bn(x)

  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_bn(t_x)
  # [N, C, H, W] -> [N, H, W, C]
  t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))

  np.testing.assert_almost_equal(j_out, t_out, decimal=6)


Average Pooling
^^^^^^^^^^^^^^^

``torch.nn.AvgPool2d`` and |nnx.avg_pool()|_ are compatible when using default parameters.
However, ``torch.nn.AvgPool2d`` has a parameter ``count_include_pad``. When ``count_include_pad=False``,
the zero-padding will not be considered for the average calculation. There does not exist a similar
parameter for |nnx.avg_pool()|_. However, we can easily implement a wrapper around the pooling
operation. ``nnx.pool()`` is the core function behind |nnx.avg_pool()|_ and |nnx.max_pool()|_.

.. |nnx.avg_pool()| replace:: ``nnx.avg_pool()``
.. _nnx.avg_pool(): https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool

.. |nnx.max_pool()| replace:: ``nnx.max_pool()``
.. _nnx.max_pool(): https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool


.. testcode::

  def avg_pool(inputs, window_shape, strides=None, padding='VALID'):
    """
    Pools the input by taking the average over a window.
    In comparison to nnx.avg_pool(), this pooling operation does not
    consider the padded zero's for the average computation.
    """
    assert len(window_shape) == 2

    y = nnx.pool(inputs, 0., jax.lax.add, window_shape, strides, padding)
    counts = nnx.pool(jnp.ones_like(inputs), 0., jax.lax.add, window_shape, strides, padding)
    y = y / counts
    return y


  key = random.key(0)
  x = random.normal(key, (1, 6, 6, 3))

  j_out = avg_pool(x, window_shape=(2, 2), strides=(1, 1), padding=((1, 1), (1, 1)))
  t_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=1, count_include_pad=False)

  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_pool(t_x)
  # [N, C, H, W] -> [N, H, W, C]
  t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))

  np.testing.assert_almost_equal(j_out, t_out, decimal=6)



Transposed Convolutions
^^^^^^^^^^^^^^^^^^^^^^^

``torch.nn.ConvTranspose2d`` and |nnx.ConvTranspose|_ are not compatible.
``nnx.ConvTranspose``_ is a wrapper around ``jax.lax.conv_transpose``_ which computes a fractionally strided convolution,
while ``torch.nn.ConvTranspose2d`` computes a gradient based transposed convolution. Currently, there is no
implementation of a gradient based transposed convolution in ``Jax``. However, there is a pending `pull request`_
that contains an implementation.

To load ``torch.nn.ConvTranspose2d`` parameters into Flax, we need to use the ``transpose_kernel`` arg in Flax's
``nnx.ConvTranspose`` layer.

.. testcode::

  t_conv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=4, kernel_size=2, padding=0)

  kernel = t_conv.weight.detach().cpu().numpy()
  bias = t_conv.bias.detach().cpu().numpy()

  # [inC, outC, kH, kW] -> [kH, kW, outC, inC]
  kernel = jnp.transpose(kernel, (2, 3, 1, 0))

  key = random.key(0)
  x = random.normal(key, (1, 6, 6, 3))

  # ConvTranspose expects the kernel to be [kH, kW, inC, outC],
  # but with `transpose_kernel=True`, it expects [kH, kW, outC, inC] instead
  j_conv = nnx.ConvTranspose(3, 4, kernel_size=(2, 2), padding='VALID', transpose_kernel=True, rngs=nnx.Rngs(0))
  j_conv.kernel.value = kernel
  j_conv.bias.value = bias
  j_out = j_conv(x)

  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_conv(t_x)
  # [N, C, H, W] -> [N, H, W, C]
  t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))
  np.testing.assert_almost_equal(j_out, t_out, decimal=6)

.. _`pull request`: https://github.com/jax-ml/jax/pull/5772

.. |nnx.ConvTranspose| replace:: ``nnx.ConvTranspose``
.. _nnx.ConvTranspose: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.ConvTranspose

.. |jax.lax.conv_transpose| replace:: ``jax.lax.conv_transpose``
.. _jax.lax.conv_transpose: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_transpose.html


Training Neural Networks
------------------------

Jax and PyTorch perform differently gradients computation:

- PyTorch sets `requires_grad=True` flag to tensors to record the computation graph and perform the automatic differentiation.

- In Jax, the automatic differentiation is a functional operation, i.e., there is no need to mark arrays with a flag to enable gradient tracking.

In Jax/flax we can easily compute gradients of a function with respect to a set of parameters using ``nnx.grad``.

Here is how the supervised learning training step can be implemented in Flax vs PyTorch:

.. tab-set::

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      import optax


      def loss_function(
          params: nnx.State,
          graphdef: nnx.GraphDef,
          nondiff: nnx.State,
          x: jax.Array,
          y: jax.Array,
          rngs: nnx.Rngs
      ):
        model = nnx.merge(graphdef, params, nondiff)
        logits = model(x, rngs=rngs)
        # For example, we compute MSE
        return ((logits - y) ** 2).mean()

      @nnx.jit(donate_argnames=("model", "optimizer"))
      def train_step(
          model: nnx.Module,
          optimizer: nnx.Optimizer,
          rngs: nnx.Rngs,
          batch: tuple[jax.Array, jax.Array],
      ) -> jax.Array:
        x, y = batch

        graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

        grad_fn = nnx.value_and_grad(loss_function)
        loss, grads = grad_fn(params, graphdef, nondiff, x, y, rngs.fork())
        optimizer.update(model, grads)

        return loss

      optimizer = nnx.Optimizer(model, tx=optax.adamw(0.001), wrt=nnx.Param)

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      import torch.optim as optim

      def loss_function(
          y_pred: torch.Tensor,
          y_true: torch.Tensor,
      ):
        # For example, we compute MSE
        return ((y_pred - y_true) ** 2).mean()

      def train_step(
          model: nn.Module,
          optimizer: optim.Optimizer,
          batch: tuple[torch.Tensor, torch.Tensor],
      ) -> float:
        model.train()
        x, y = batch
        logits = model(x)
        loss = loss_function(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

      optimizer = optim.AdamW(model.parameters(), lr=0.001)


In case if we train all model's parameters and the model does not have any other non-differentiable arrays,
the training step can be simplified and looks similar to PyTorch code
(no more ``nnx.split`` and ``nnx.merge`` of the model):

.. code-block:: python

  def loss_function(
      model: nnx.Module,
      x: jax.Array,
      y: jax.Array,
      rngs: nnx.Rngs
  ):
    logits = model(x, rngs=rngs)
    # For example, we compute MSE
    return ((logits - y) ** 2).mean()

  @nnx.jit(donate_argnames=("model", "optimizer"))
  def train_step(
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      rngs: nnx.Rngs,
      batch: tuple[jax.Array, jax.Array],
  ) -> jax.Array:
    x, y = batch

    grad_fn = nnx.value_and_grad(loss_function)
    loss, grads = grad_fn(model, x, y, rngs.fork())
    optimizer.update(model, grads)

    return loss

Please note that the training step function is jitted in NNX version and we have consider the following implications:

- the first call of the training step is slow as it starts the compilation
- if the input batch changes its array shape, then the training step will be recompiled (i.e. performance degradation).
- the argument ``donate_argnames=("model", "optimizer")`` in ``nnx.jit`` is important for efficient device memory usage.
- direct printing of the loss value per iteration will result into synchronous execution (i.e. performance degradation).


It is common to run the model evaluation during the training loop. In the evaluation phase,
we usually disable the stochastic layers (e.g. ``Dropout``) and stop batch statistics accumulation
in normalization layers.

.. tab-set::

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      model = nnx.view(model, deterministic=False, use_running_average=False)
      eval_model = nnx.view(model, deterministic=False, use_running_average=False)

      for epoch in range(num_epochs):
        # run model single epoch training
        train_single_epoch(model, ...)

        # run model evaluation
        metrics = evaluate(eval_model)

  .. tab-item:: PyTorch
    :sync: PyTorch

    .. code-block:: python

      for epoch in range(num_epochs):
        # run model single epoch training
        model.train()
        train_single_epoch(model, ...)

        # run model evaluation
        model.eval()
        metrics = evaluate(model)


The function ``nnx.view`` creates a new instance of the model with shared parameters.
The arguments passed to ``nnx.view`` function (e.g. ``deterministic``, ``use_running_average``
in above example) are defined by the layers of the model: ``deterministic`` for ``nnx.Dropout``,
``use_running_average`` for ``nnx.BatchNorm``. If model does not have stochastic nor normalization
layers, then we do not need to create a new view of the model with ``nnx.view``.


We recommend to check the following resources for the best practices on the topic:

- :doc:`Flax Examples </examples/index>`
- `Jax Training Cookbook <https://docs.jax.dev/en/latest/the-training-cookbook.html>`__


Porting PyTorch weights to NNX
------------------------------

This section explains how to port pretrained weights from a PyTorch model to a Flax NNX model.

First, let us see how we can inspect an NNX model's parameters using the ``nnx.state()`` function, which is similar to PyTorch's
``model.state_dict()`` method. The function returns a dict-like ``nnx.State`` object with keys defining the path
to each parameter and ``jax.Array`` values:

.. code-block:: python

  import flax.nnx as nnx
  import jax.numpy as jnp

  model = nnx.Sequential(
      nnx.Linear(4, 6, rngs=nnx.Rngs(0)),
      nnx.Linear(6, 8, rngs=nnx.Rngs(0)),
  )
  state = nnx.state(model)
  print(jax.tree.map(lambda p: p.shape, state))
  # State({
  #   'layers': {
  #     0: {
  #       'bias': Param(
  #         value=(6,)
  #       ),
  #       'kernel': Param(
  #         value=(4, 6)
  #       )
  #     },
  #     1: {
  #       'bias': Param(
  #         value=(8,)
  #       ),
  #       'kernel': Param(
  #         value=(6, 8)
  #       )
  #     }
  #   }
  # })

The keys in the state dictionary represent the path to each parameter using the attribute hierarchy of the model.
For example, ``layers.0.kernel`` refers to the ``kernel`` attribute of the first ``Linear`` layer in the ``Sequential`` container.

Now, let us update the state and load the updated weights to the model using ``nnx.update``
(similar to PyTorch ``model.load_state_dict()`` method):

.. code-block:: python

  new_state = jax.tree.map(lambda p: jnp.ones_like(p), state)
  nnx.update(model, new_state)

Thus, we need to create a state of ``jax.Array`` initialized using PyTorch weights and update NNX model.
In order to avoid multiple memory allocations on the device e.g. NNX model random weights allocation, we can use
``nnx.eval_shape``. This function helps to create a state of the model with abstract arrays (similar to tensors on ``device=meta`` in PyTorch)
such that there is no memory allocation on the device and we replace all the abstract array with real arrays.

.. code-block:: python

  model = nnx.eval_shape(lambda: MyModel())
  graph_def, abs_state = nnx.split(model)

  # State({
  #   'layers': {
  #     0: {
  #       'bias': Param( # 6 (24 B)
  #         value=ShapeDtypeStruct(shape=(6,), dtype=float32)
  #       ),
  #       'kernel': Param( # 24 (96 B)
  #         value=ShapeDtypeStruct(shape=(4, 6), dtype=float32)
  #       )
  #     },
  #     1: {
  #       'bias': Param( # 8 (32 B)
  #         value=ShapeDtypeStruct(shape=(8,), dtype=float32)
  #       ),
  #       'kernel': Param( # 48 (192 B)
  #         value=ShapeDtypeStruct(shape=(6, 8), dtype=float32)
  #       )
  #     }
  #   }
  # })

Note that if model contains non-parameters arrays, abstract state will still replace them
with ``ShapeDtypeStruct`` and we should manually replace them with appropriate ``jax.Array``
to avoid errors.


In the section below we provide all the details on how to port PyTorch weights to NNX model.

Example: loading PyTorch Weights for ResNet50
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider an example of porting ResNet50 weights downloaded from HuggingFace hub
and converted to NNX state and finally update our NNX ResNet50 model.

1. Download weights from the HuggingFace hub and load the tensors

.. code-block:: python

  import os
  from huggingface_hub import snapshot_download
  from safetensors import safe_open


  checkpoint_path = snapshot_download(repo_id="microsoft/resnet-50", allow_patterns="*.safetensors")
  checkpoint_sft = os.path.join(checkpoint_path, "model.safetensors")
  state_dict = {}
  with safe_open(checkpoint_sft, framework="pt") as f:
    for key in f.keys():
      state_dict[key] = f.get_tensor(key)
  print(len(state_dict), list(state_dict.keys())[:3], type(next(iter(state_dict.values()))))
  # 320 ['classifier.1.bias', 'classifier.1.weight', 'resnet.embedder.embedder.convolution.weight'] <class 'torch.Tensor'>

Note: we can also use ``framework="flax"`` to get directly a dictionary of ``jax.Array``,
but to expose the full conversion pipeline, we start with ``torch.Tensor`` as input.

2. Convert PyTorch key names to match the NNX state dictionary structure.

Let us first provide an NNX implementation of ResNet50:

.. code-block:: python

  import flax.nnx as nnx


  class ResNet50(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
      self.stem = Stem(rngs=rngs)

      self.layer0 = BlockGroup(64, 64, 3, stride=1, rngs=rngs)
      self.layer1 = BlockGroup(256, 128, 4, stride=2, rngs=rngs)
      self.layer2 = BlockGroup(512, 256, 6, stride=2, rngs=rngs)
      self.layer3 = BlockGroup(1024, 512, 3, stride=2, rngs=rngs)
      self.pool = lambda x: nnx.avg_pool(x, (x.shape[1], x.shape[2]))
      self.fc = nnx.Linear(2048, 1000, rngs=rngs)

    def __call__(self, x):
      x = self.stem(x)
      x = self.layer0(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.pool(x)
      x = x.reshape((x.shape[0], x.shape[-1]))
      return self.fc(x)


  class Stem(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
      self.conv = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False, rngs=rngs)
      self.bn = nnx.BatchNorm(64, use_running_average=True, rngs=rngs)
      self.pool = lambda x: nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
      x = nnx.relu(self.bn(self.conv(x)))
      x = self.pool(x)
      return x


  class BlockGroup(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks, stride: int, *, rngs: nnx.Rngs):
      self.blocks = nnx.List()

      downsample = None
      if stride != 1 or in_channels != out_channels * 4:
        downsample = Downsample(in_channels, out_channels * 4, stride, rngs=rngs)

      self.blocks.append(Bottleneck(in_channels, out_channels, stride, downsample, rngs=rngs))
      for _ in range(1, blocks):
        self.blocks.append(Bottleneck(out_channels * 4, out_channels, stride=1, downsample=None, rngs=rngs))

    def __call__(self, x):
      for block in self.blocks:
        x = block(x)
      return x

  class Bottleneck(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None, *, rngs: nnx.Rngs):
      self.conv0 = nnx.Conv(
          in_channels, out_channels, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
      )
      self.bn0 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

      self.conv1 = nnx.Conv(
          out_channels, out_channels, kernel_size=(3, 3), strides=stride, padding=1, use_bias=False, rngs=rngs
      )
      self.bn1 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

      self.conv2 = nnx.Conv(
          out_channels, out_channels * 4, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
      )
      self.bn2 = nnx.BatchNorm(out_channels * 4, use_running_average=True, rngs=rngs)

      self.downsample = downsample

    def __call__(self, x):
      identity = x
      x = nnx.relu(self.bn0(self.conv0(x)))
      x = nnx.relu(self.bn1(self.conv1(x)))
      x = self.bn2(self.conv2(x))
      if self.downsample is not None:
        identity = self.downsample(identity)
      return nnx.relu(x + identity)


  class Downsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, *, rngs: nnx.Rngs):
      self.conv = nnx.Conv(
        in_channels, out_channels, kernel_size=(1, 1), strides=stride, padding=0, use_bias=False, rngs=rngs
      )
      self.bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

    def __call__(self, x):
      return self.bn(self.conv(x))


  model = nnx.eval_shape(lambda : ResNet50(rngs=nnx.Rngs(0)))
  graph_def, abs_state = nnx.split(model)
  jax_state = nnx.to_pure_dict(abs_state)
  print(len(jax_state), list(jax_state.paths)[:3], type(next(iter(jax_state.leaves))))
  # 6 ['fc', 'layer0', 'layer1'] <class 'dict'>

We have to manually create a mapping between pretrained weights keys and our NNX model state keys.
Depending on implementation differences between pretrained model and NNX model, the mapping code can be bulky.
In addition to key mapping, we should also provide data permutation rules for linear and convolution layers.
Here is an example of the weights mapping (inspired by `Bonsai project <https://github.com/jax-ml/bonsai>`_)

.. code-block:: python

  def get_key_and_transform_mapping():
    from enum import Enum

    class Transform(Enum):
      BIAS = None
      LINEAR = ((1, 0), None, False)
      CONV2D = ((2, 3, 1, 0), None, False)
      DEFAULT = None

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
    return {
      r"^resnet\.embedder\.embedder\.convolution\.weight$": ("stem.conv.kernel", Transform.CONV2D),
      r"^resnet\.embedder\.embedder\.normalization\.weight$": ("stem.bn.scale", Transform.DEFAULT),
      r"^resnet\.embedder\.embedder\.normalization\.bias$": ("stem.bn.bias", Transform.BIAS),
      r"^resnet\.embedder\.embedder\.normalization\.running_mean$": ("stem.bn.mean", Transform.DEFAULT),
      r"^resnet\.embedder\.embedder\.normalization\.running_var$": ("stem.bn.var", Transform.DEFAULT),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.convolution\.weight$": (
        r"layer\1.blocks.\2.conv\3.kernel",
        Transform.CONV2D,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.weight$": (
        r"layer\1.blocks.\2.bn\3.scale",
        Transform.DEFAULT,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.bias$": (
        r"layer\1.blocks.\2.bn\3.bias",
        Transform.BIAS,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_mean$": (
        r"layer\1.blocks.\2.bn\3.mean",
        Transform.DEFAULT,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_var$": (
        r"layer\1.blocks.\2.bn\3.var",
        Transform.DEFAULT,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.convolution\.weight$": (
        r"layer\1.blocks.\2.downsample.conv.kernel",
        Transform.CONV2D,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.weight$": (
        r"layer\1.blocks.\2.downsample.bn.scale",
        Transform.DEFAULT,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.bias$": (
        r"layer\1.blocks.\2.downsample.bn.bias",
        Transform.BIAS,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_mean$": (
        r"layer\1.blocks.\2.downsample.bn.mean",
        Transform.DEFAULT,
      ),
      r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_var$": (
        r"layer\1.blocks.\2.downsample.bn.var",
        Transform.DEFAULT,
      ),
      r"^classifier\.1\.weight$": ("fc.kernel", Transform.LINEAR),
      r"^classifier\.1\.bias$": ("fc.bias", Transform.BIAS),
    }

  def map_to_nnx_key(mapping, source_key):
    import re

    subs = [
      (re.sub(pat, repl, source_key), transform)
      for pat, (repl, transform) in mapping.items()
      if re.match(pat, source_key)
    ]
    if not subs:
      if "num_batches_tracked" not in source_key:
        print(f"No mapping found for key: {source_key!r}")
      return None, None
    if len(subs) > 1:
      keys = [s for s, _ in subs]
      raise ValueError(f"Multiple mappings found for {source_key!r}: {keys}")
    return subs[0]

  def stoi(s: str) -> int | str:
    try:
      return int(s)
    except ValueError:
      return s

  def assign_weights_from_eval_shape(
    keys: list[str], tensor: jnp.ndarray, state_dict: dict, st_key: str, transform
  ):
    key, *rest = keys
    if not rest:
      if isinstance(tensor, torch.Tensor):
        tensor = jnp.asarray(tensor)
      if transform is not None:
        permute, reshape, reshape_first = transform
        if reshape_first and reshape is not None:
          tensor = tensor.reshape(reshape)
        if permute:
          tensor = tensor.transpose(permute)
        if not reshape_first and reshape is not None:
          tensor = tensor.reshape(reshape)
      if tensor.shape != state_dict[key].shape:
        raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")

      tensor = tensor.astype(state_dict[key].dtype)
      if hasattr(state_dict[key], "sharding") and state_dict[key].sharding is not None:
        tensor = jax.device_put(tensor, state_dict[key].sharding.spec)
      state_dict[key] = tensor
    else:
      assign_weights_from_eval_shape(rest, tensor, state_dict[key], st_key, transform)


3. Update the NNX model

Finally we convert the ``state_dict`` with downloaded weights into ``jax_state`` and update
the NNX model with the new state:

.. code-block:: python

  mapping = get_key_and_transform_mapping()
  for st_key, tensor in state_dict.items():
    jax_key, transform = map_to_nnx_key(mapping, st_key)
    if jax_key is None:
      continue
    keys = [stoi(k) for k in jax_key.split(".")]
    assign_weights_from_eval_shape(keys, tensor, jax_state, st_key, transform.value)

  model = nnx.merge(graph_def, jax_state)

4. Verify the ported model

After porting the weights, it is important to verify that the NNX model produces the same
outputs as the PyTorch model:

.. code-block:: python

  import numpy as np
  import torch
  from transformers import ResNetForImageClassification

  x = jax.random.uniform(jax.random.key(0), (2, 224, 224, 3))
  output = model(x)

  torch_model = ResNetForImageClassification.from_pretrained(checkpoint_path)
  baseline_inputs = {
    "pixel_values": torch.tensor(np.asarray(x)).to(torch.float32).permute(0, 3, 1, 2)
  }
  with torch.no_grad():
    expected = torch_model(**baseline_inputs).logits.cpu().detach().numpy()

  np.testing.assert_allclose(output, expected, rtol=1e-5)

