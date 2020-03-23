import jax
import jax.numpy as jnp
import flax
from jax import random


class Layer(flax.nn.Module):
    def apply(self, x, a_fixed=False, a_init=jax.nn.initializers.ones):
        a = self.param('a', (1, ), a_init)
        if a_fixed:
            _default_key = random.PRNGKey(0)
            a = a_init(_default_key, (1, ))
        return a * x


class MyModel(flax.nn.Module):
    def apply(self, x, **kwargs):
        x = Layer(x, **kwargs.get('layer_kwargs', {}), name='layer')
        return x


free_kwargs = {'layer_kwargs': {'a_fixed': False}}
fixed_kwargs = {'layer_kwargs': {'a_fixed': True}}

# use partial to fix the initial functions
free_model_def = MyModel.partial(**free_kwargs)
fixed_model_def = MyModel.partial(**fixed_kwargs)

def create_model(model_def, key, input_specs):
    x, init_params = model_def.init_by_shape(key, input_specs)
    return flax.nn.Model(model_def, init_params)


rng = random.PRNGKey(0)
input_shape_and_dtype = [((5, 1), jnp.float32), ]
free_model = create_model(free_model_def, rng, input_shape_and_dtype)
fixed_model = create_model(fixed_model_def, rng, input_shape_and_dtype)

new_fixed_kwargs = {
    'layer_kwargs':
        {'a_fixed': True, 'a_init': lambda key, shape: 3.14*jnp.ones([1])}}


def loss_fn(model):
    x = jnp.ones(*input_shape_and_dtype[0])
    y = model(x)
    return jnp.mean(y ** 2)


free_model_grad = jax.grad(loss_fn)(free_model)
fixed_model_grad = jax.grad(loss_fn)(fixed_model)

assert(loss_fn(free_model) == loss_fn(fixed_model))
assert(free_model_grad.params['layer']['a'] == 2.)
assert(fixed_model_grad.params['layer']['a'] == 0.)

x = jnp.ones(*input_shape_and_dtype[0])
new_fixed_model = fixed_model.replace(params={'layer': {'a': 3.14*jnp.ones([1])}})
assert(loss_fn(new_fixed_model) != loss_fn(fixed_model))