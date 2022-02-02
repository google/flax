Convert PyTorch Models to Flax
==============================

.. testsetup::

  import numpy as np
  import jax
  from jax import random, numpy as jnp
  import flax

  from flax import linen as nn
  from flax.core import freeze

  import torch

We will show how to convert PyTorch models to Flax. We will cover convolutions, fc layers, batch norm, and average pooling.


FC Layers
--------------------------------

Let's start with fc layers. The only thing to be aware of here is that the PyTorch kernel has shape [outC, inC]
and the Flax kernel has shape [inC, outC]. Transposing the kernel will do the trick.

.. testcode::

  t_fc = torch.nn.Linear(in_features=3, out_features=4)
  
  kernel = t_fc.weight.detach().cpu().numpy()
  bias = t_fc.bias.detach().cpu().numpy()
  
  # [outC, inC] -> [inC, outC]
  kernel = jnp.transpose(kernel, (1, 0))
  
  key = random.PRNGKey(0)
  x = random.normal(key, (1, 3))
  
  params = freeze({'params': {'kernel': kernel, 'bias': bias}})
  j_fc = nn.Dense(features=4)
  j_out = j_fc.apply(params, x)
  
  t_x = torch.from_numpy(np.array(x))
  t_out = t_fc(t_x)
  t_out = t_out.detach().cpu().numpy()
  
  assert np.all(np.abs(j_out - t_out) < 1e-06)



Convolutions
--------------------------------

Let's now look at 2D convolutions. PyTorch uses the NCHW format and Flax uses NHWC.
Consequently, the kernels will have different shapes. The kernel in PyTorch has shape [outC, inC, kH, kW]
and the Flax kernel has shape [kH, kW, inC, outC]. Transposing the kernel will do the trick.

.. testcode::

  t_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding='valid')
  
  kernel = t_conv.weight.detach().cpu().numpy()
  bias = t_conv.bias.detach().cpu().numpy()
  
  # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
  kernel = jnp.transpose(kernel, (2, 3, 1, 0))
  
  key = random.PRNGKey(0)
  x = random.normal(key, (1, 6, 6, 3))
  
  params = freeze({'params': {'kernel': kernel, 'bias': bias}})
  j_conv = nn.Conv(features=4, kernel_size=(2, 2), padding='valid')
  j_out = j_conv.apply(params, x)
  
  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_conv(t_x)
  # [N, C, H, W] -> [N, H, W, C]
  t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))
  
  assert np.all(np.abs(j_out - t_out) < 1e-06)


   
Convolutions and FC Layers
--------------------------------

We have to be careful, when we have a model that uses convolutions followed by fc layers (ResNet, VGG, etc).
In PyTorch, the activations will have shape [N, C, H, W] after the convolutions and are then
reshaped to [N, C * H * W] before being fed to the fc layers.
When we port our weights from PyToch to Flax, the activations after the convolutions will be of shape [N, H, W, C] in Flax.
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
      x = x.view(x.shape[0], -1)
      x = self.fc(x)
      return x


  t_model = TModel()



Now, if you want to use the weights from this model in Flax, the corresponding Flax model has to look like this:


.. testcode::

  class JModel(nn.Module):
      
    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=4, kernel_size=(2, 2), padding='valid', name='conv')(x)
      # [N, H, W, C] -> [N, C, H, W]
      x = jnp.transpose(x, (0, 3, 1, 2))
      x = jnp.reshape(x, (x.shape[0], -1))
      x = nn.Dense(features=2, name='fc')(x)
      return x


  j_model = JModel()



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
  
  params = freeze({'params': {'conv': {'kernel': conv_kernel, 'bias': conv_bias},
                              'fc': {'kernel': fc_kernel, 'bias': fc_bias}}})

  key = random.PRNGKey(0)
  x = random.normal(key, (1, 6, 6, 3))

  j_out = j_model.apply(params, x)
  
  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_model(t_x)
  t_out = t_out.detach().cpu().numpy()
  
  assert np.all(np.abs(j_out - t_out) < 1e-06)




Batch Norm
--------------------------------

``torch.nn.BatchNorm2d`` uses ``0.1`` as the default value for the momentum parameter while
|flax.linen.BatchNorm|_ uses ``0.9``. However, this corresponds to the same computation, because PyTorch multiplies
the estimated statistic with ``(1 − momentum)`` and the new observed value with ``momentum``,
while Flax multiplies the estimated statistic with momentum and the new observed value with ``(1 − momentum)``.

.. |flax.linen.BatchNorm| replace:: ``flax.linen.BatchNorm``
.. _flax.linen.BatchNorm: https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.BatchNorm.html

.. testcode::

  t_bn = torch.nn.BatchNorm2d(num_features=3, momentum=0.1)
  t_bn.eval()
  
  scale = t_bn.weight.detach().cpu().numpy()
  bias = t_bn.bias.detach().cpu().numpy()
  mean = t_bn.running_mean.detach().cpu().numpy()
  var = t_bn.running_var.detach().cpu().numpy()
  
  params = freeze({'params': {'scale': scale, 'bias': bias},
                   'batch_stats': {'mean': mean, 'var': var}})
  
  key = random.PRNGKey(0)
  x = random.normal(key, (1, 6, 6, 3))
  
  j_bn = nn.BatchNorm(momentum=0.9, use_running_average=True)
  
  j_out = j_bn.apply(params, x)
  
  # [N, H, W, C] -> [N, C, H, W]
  t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
  t_out = t_bn(t_x)
  # [N, C, H, W] -> [N, H, W, C]
  t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))
  
  assert np.all(np.abs(j_out - t_out) < 1e-06)



Average Pooling
--------------------------------

``torch.nn.AvgPool2d`` and |flax.linen.avg_pool()|_ are compatible when using default parameters.
However, ``torch.nn.AvgPool2d`` has a parameter ``count_include_pad``. When ``count_include_pad=False``,
the zero-padding will not be considered for the average calculation. There does not exist a similar
parameter for |flax.linen.avg_pool()|_. However, we can easily implement a wrapper around the pooling
operation.

.. |flax.linen.avg_pool()| replace:: ``flax.linen.avg_pool()``
.. _flax.linen.avg_pool(): https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.avg_pool.html

.. testcode::

   def avg_pool(inputs, window_shape, strides=None, padding='VALID'):
     """
     Pools the input by taking the average over a window.
     In comparison to flax.linen.avg_pool, this pooling operation does not
     consider the padded zero's for the average computation.
     Args:
       inputs: input data with dimensions (batch, window dims..., features).
       window_shape: a shape tuple defining the window to reduce over.
       strides: a sequence of `n` integers, representing the inter-window
           strides (default: `(1, ..., 1)`).
       padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
         of `n` `(low, high)` integer pairs that give the padding to apply before
         and after each spatial dimension (default: `'VALID'`).
     Returns:
       The average for each window slice.
     """
     assert inputs.ndim == 4
     assert len(window_shape) == 2
     
     # from https://github.com/google/flax/blob/main/flax/linen/pooling.py
     y = pool(inputs, 0., jax.lax.add, window_shape, strides, padding)

     ones = jnp.ones(shape=(1, inputs.shape[1], inputs.shape[2], 1)).astype(inputs.dtype)
     counts = jax.lax.conv_general_dilated(ones,
                                           jnp.expand_dims(jnp.ones(window_shape).astype(inputs.dtype), axis=(-2, -1)),
                                           window_strides=(1, 1),
                                           padding=((1, 1), (1, 1)),
                                           dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
                                           feature_group_count=1)
     y = y / counts 
     return y




Transposed Convolutions
--------------------------------

``torch.nn.ConvTranspose2d`` and |flax.linen.ConvTranspose|_ are not compatible.
|flax.linen.ConvTranspose|_ is a wrapper around |jax.lax.conv_transpose|_ which computes a fractionally strided convolution,
while ``torch.nn.ConvTranspose2d`` computes a gradient based transposed convolution.


.. |flax.linen.ConvTranspose| replace:: ``flax.linen.ConvTranspose``
.. _flax.linen.ConvTranspose: https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.ConvTranspose.html

.. |jax.lax.conv_transpose| replace:: ``jax.lax.conv_transpose``
.. _jax.lax.conv_transpose: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_transpose.html
