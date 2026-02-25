# Copyright 2024 The Flax Authors.
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

from functools import partial
import typing as tp

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
from jax.lax import Precision
import numpy as np

from flax import linen
from flax import nnx
from flax.typing import Dtype, PrecisionLike, Shape


class TestLinearGeneral(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_basic(
    self,
    dtype,
    param_dtype,
    precision,
    preferred_element_type,
  ):
    module = nnx.LinearGeneral(
      2,
      3,
      rngs=nnx.Rngs(0),
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
    )
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)
    if preferred_element_type is not None:
      assert y.dtype == preferred_element_type
    assert module.kernel.shape == (2, 3)
    assert module.kernel.dtype == param_dtype
    assert module.bias is not None
    assert module.bias.shape == (3,)

  def test_basic_multi_features(self):
    module = nnx.LinearGeneral(2, (3, 4), rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3, 4)
    assert module.kernel.shape == (2, 3, 4)
    assert module.bias is not None
    assert module.bias.shape == (3, 4)


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_nnx_linear_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    preferred_element_type: tp.Optional[Dtype],
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    OUT_FEATURES = 64

    x = jax.numpy.ones((1, IN_FEATURES))
    model_nnx = nnx.eval_shape(
      lambda rngs: nnx.Linear(
        IN_FEATURES,
        OUT_FEATURES,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        preferred_element_type=preferred_element_type,
        rngs=rngs,
      ),
      rngs,
    )
    if preferred_element_type is not None:
      dot_general = partial(
        jax.lax.dot_general,
        preferred_element_type=preferred_element_type,
      )
    else:
      dot_general = None
    model = linen.Dense(
      OUT_FEATURES,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      dot_general=dot_general,
    )
    variables = model.init(key, x)
    model_nnx.kernel.set_value(variables['params']['kernel'])
    if use_bias:
      model_nnx.bias.set_value(variables['params']['bias'])

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  @parameterized.product(
    einsum_str=['defab,bcef->adefc', 'd...ab,bc...->ad...c'],
    bias_shape=[None, (6, 7, 5)],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_nnx_einsum_equivalence(
    self,
    einsum_str,
    bias_shape: tp.Optional[Shape],
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    preferred_element_type: tp.Optional[Dtype],
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    INPUT_SHAPE = (8, 6, 7, 3, 4)
    KERNEL_SHAPE = (4, 5, 6, 7)

    x = jax.random.normal(key, INPUT_SHAPE)
    model_nnx = nnx.Einsum(
      einsum_str,
      KERNEL_SHAPE,
      bias_shape,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
      rngs=rngs,
    )
    model = linen.Einsum(
      KERNEL_SHAPE,
      einsum_str,
      use_bias=True if bias_shape is not None else False,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
    )

    variables = model.init(key, x)
    variables['params']['kernel'] = model_nnx.kernel[...]
    if bias_shape is not None:
      assert model_nnx.bias is not None
      variables['params']['bias'] = model_nnx.bias[...]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key, x)
    model_nnx.kernel.set_value(variables['params']['kernel'])
    if bias_shape is not None:
      assert model_nnx.bias is not None
      model_nnx.bias.set_value(variables['params']['bias'])
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  def test_einsum_op(self):
    def custom_einsum(*args, **kwargs):
      out = jnp.einsum(*args, **kwargs)
      return out.reshape((1, *out.shape))
    model = nnx.Einsum('ab,bc->ac', (3, 4), einsum_op=custom_einsum,
                       rngs=nnx.Rngs(42))
    y = model(jnp.ones((2, 3)))
    assert y.shape == (1, 2, 4)


class TestLayersSameGraph(parameterized.TestCase):

  @parameterized.product(
      module_args_kwargs_initargs=[
          (nnx.LinearGeneral, (2, (3, 4)), ("kernel_init", "bias_init")),
          (nnx.Linear, (2, 4), ("kernel_init", "bias_init")),
          (nnx.Einsum, ("ik,kj->ij", (5, 4), 5), ("kernel_init", "bias_init")),
          (nnx.Conv, (2, 4, 3), ("kernel_init", "bias_init")),
          (nnx.ConvTranspose, (2, 4, 3), ("kernel_init", "bias_init")),
          (nnx.Embed, (2, 4), ("embedding_init",)),
          (
              nnx.MultiHeadAttention,
              (8, 5, 16),
              ("kernel_init", "out_kernel_init", "bias_init", "out_bias_init"),
          ),
          (nnx.BatchNorm, (3,), ("scale_init", "bias_init")),
          (nnx.LayerNorm, (3,), ("scale_init", "bias_init")),
          (nnx.RMSNorm, (3,), ("scale_init",)),
          (nnx.GroupNorm, (6, 3), ("scale_init", "bias_init")),
          (nnx.InstanceNorm, (6,), ("scale_init", "bias_init")),
          (
              nnx.LSTMCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.OptimizedLSTMCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.SimpleCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.GRUCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
      ],
  )
  def test(self, module_args_kwargs_initargs):
    module_cls, args, init_argnames = module_args_kwargs_initargs
    kwargs = {"rngs": nnx.Rngs(0)}
    init_zeros = nnx.initializers.zeros
    init_ones = nnx.initializers.ones
    init1_kwargs = {k: init_zeros for k in init_argnames}
    init2_kwargs = {k: init_ones for k in init_argnames}
    mod1 = module_cls(*args, **init1_kwargs, **kwargs)
    mod2 = module_cls(*args, **init2_kwargs, **kwargs)
    g1, g2 = nnx.graphdef(mod1), nnx.graphdef(mod2)
    assert g1 == g2


class TestLayersParamsMetadata(parameterized.TestCase):

  @parameterized.product(
      module_args_kwargs_initargs=[
          (nnx.LinearGeneral, (2, (3, 4)), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Linear, (2, 4), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Einsum, ("ik,kj->ij", (5, 4), 5), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Conv, (2, 4, 3), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.ConvTranspose, (2, 4, 3), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Embed, (2, 4), (("embedding", 2, ()), )),
          (
              partial(nnx.MultiHeadAttention, normalize_qk=True),
              (8, 5, 16),
              (
                ("kernel", 2, (("query", "kernel"), ("key", "kernel"), ("value", "kernel"))),
                ("out_kernel", 2, (("out", "kernel"), )),
                ("bias", 1, (("query", "bias"), ("key", "bias"), ("value", "bias"))),
                ("out_bias", 1, (("out", "bias"), )),
                ("query_ln_scale", 1, (("query_ln", "scale"), )),
                ("key_ln_scale", 1, (("key_ln", "scale"), )),
              ),
          ),
          (nnx.BatchNorm, (3,), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.LayerNorm, (3,), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.RMSNorm, (3,), (("scale", 1, ()), )),
          (nnx.GroupNorm, (6, 3), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.InstanceNorm, (6,), (("scale", 1, ()), ("bias", 1, ()))),
          (
            nnx.LoRA,
            (3, 2, 4),
            (
              ("a", 2, ((None, "lora_a"), )),
              ("b", 2, ((None, "lora_b"), )),
            )
          ),
          (
            partial(nnx.LoRALinear, lora_rank=4),
            (3, 2),
            (
              ("a", 2, (("lora", "lora_a"), )),
              ("b", 2, (("lora", "lora_b"), )),
            )
          ),
          (
              nnx.LSTMCell,
              (4, 5),
              (
                (
                  "kernel",
                  2,
                  ((name, "kernel") for name in ["ii", "if_", "ig", "io"])
                ),
                (
                  "recurrent_kernel",
                  2,
                  ((name, "kernel") for name in ["hi", "hf", "hg", "ho"])
                ),
                (
                  "bias",
                  1,
                  ((name, "bias") for name in ["hi", "hf", "hg", "ho"])
                ),
              ),
          ),
          (
              nnx.OptimizedLSTMCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
                ("bias", 1, (("dense_h", "bias"), )),
              )
          ),
          (
              nnx.SimpleCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("bias", 1, (("dense_i", "bias"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
              )
          ),
          (
              nnx.GRUCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("bias", 1, (("dense_i", "bias"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
              )
          ),
      ],
  )
  def test(self, module_args_kwargs_initargs):
    module_cls, args, metadata_argnames = module_args_kwargs_initargs
    kwargs = {"rngs": nnx.Rngs(0)}
    out_sharding = ("din", "dout")
    metadata_kwargs = {
      f"{key}_metadata": {"out_sharding": out_sharding[:le]}
      for key, le, _ in metadata_argnames
    }

    mesh = jax.make_mesh(
      (1, 1),
      out_sharding,
      axis_types=(jax.sharding.AxisType.Auto,) * len(out_sharding),
    )
    with jax.set_mesh(mesh):
      module = module_cls(*args, **metadata_kwargs, **kwargs)

    for key, le, attrs in metadata_argnames:
      attrs = attrs if attrs else ((None, key), )
      for attr_name, param_name in attrs:
        attr = getattr(module, attr_name) if attr_name is not None else module
        param = getattr(attr, param_name)
        self.assertIsNotNone(param.out_sharding)
        self.assertEqual(param.out_sharding, out_sharding[:le])


if __name__ == '__main__':
  absltest.main()
