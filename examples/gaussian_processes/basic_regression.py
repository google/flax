from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from flax import nn
from jax import random

import scipy as oscipy

from jax.tree_util import tree_flatten, tree_unflatten

import kernels
import distributions

FLAGS = flags.FLAGS


class KernelProvider(nn.Module):
    """ Provides a Kernel -- to be used inside more functional Modules

    The role of a kernel provider is to handle initialisation, and
    parameter storage of a particular kernel function. Allowing
    kernels to be slotted into more complex models build using the Flax
    functional api.
    """
    def apply(self, x,
              amplitude_init=jax.nn.initializers.ones,
              lengthscale_init=jax.nn.initializers.ones):
        amplitude = jax.nn.softplus(
            self.param('amplitude',
                       (1,),
                       amplitude_init)) + jnp.finfo(float).tiny

        lengthscale = jax.nn.softplus(
            self.param('lengthscale',
                       (x.shape[-1],),
                       lengthscale_init)) + jnp.finfo(float).tiny

        return kernels.Kernel(
            lambda x, y: kernels.rbf_kernel_fun(x, y, amplitude, lengthscale))


class ObservationModel(nn.Module):
    """ p(y|x, {hyper par}) = ∫p(y, f | x),
    where f(x) ~ GP() """
    def apply(self, gp):
        """
        """
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale',
                       (), jax.nn.initializers.ones))

        # get p(f|x, hyper-params)
        pf = gp.marginal()

        mean = pf.mean
        diag = (obs_noise_scale ** 2
                + gp.jitter) * jnp.ones(mean.shape[-1])
        covariance = pf.scale @ pf.scale.T + jnp.diag(diag)

        return distributions.MultivariateNormalTriL(
            mean, jnp.linalg.cholesky(covariance))


class MyModel(nn.Module):
    """ GP Regression with observation noise. """
    def apply(self, x):
        kern_fn = KernelProvider(x, name='rbf_kernel')
        mean_fn = lambda x: jnp.zeros(x.shape[:-1], x.dtype)

        gp = distributions.GaussianProcess(x,
                                           mean_fn,
                                           kern_fn,
                                           1.0e-6)

        # marginalise over distribution of gp at index_points
        # to produce p(y|x, {hyper_pars})
        py = ObservationModel(gp, name='obs_model')
        return py


def build_par_pack_and_unpack(model):
    value_flat, value_tree = tree_flatten(model.params)
    section_shapes = [item.shape for item in value_flat]
    section_sizes = jnp.cumsum(jnp.array([item.size for item in value_flat]))

    def par_from_array(arr):
        value_flat = jnp.split(arr, section_sizes)
        value_flat = [x.reshape(s)
                      for x, s in zip(value_flat, section_shapes)]

        params = tree_unflatten(value_tree, value_flat)
        return params

    def array_from_par(params):
        value_flat, value_tree = tree_flatten(params)
        return jnp.concatenate([item.ravel() for item in value_flat])

    return par_from_array, array_from_par


def get_datasets():
    rng = random.PRNGKey(0)
    index_points = jnp.linspace(-3., 3., 5)[..., jnp.newaxis]
    y = (jnp.sin(index_points[:, 0])
         + .1 * random.normal(rng, index_points.shape[:-1]))
    train_ds = {'index_points': index_points,
                'y': y}
    return train_ds


def train(train_ds):
    rng = random.PRNGKey(0)

    # initialise the model
    py, params = MyModel.init(rng, train_ds['index_points'])
    model = nn.Model(MyModel, params)

    # utility functions for packing and unpacking param dicts
    par_from_array, array_from_par = build_par_pack_and_unpack(model)

    def loss_fun(model, params):
        py = model.module.call(params, train_ds['index_points'])
        return -py.log_prob(train_ds['y'])

    # wrap loss fun for scipy.optimize
    def loss_fun_closure(arr):
        params = par_from_array(arr)
        return loss_fun(model, params)

    @jax.jit
    def loss_and_grads(x):
        return jax.value_and_grad(loss_fun_closure)(x)

    init_value = array_from_par(params)
    res = oscipy.optimize.minimize(loss_and_grads, init_value, jac=True)
    logging.info('Optimisation message: {}'.format(res.message))

    trained_model = model.replace(params=par_from_array(res.x))
    return trained_model


def main(_):
    # GPs need higher prec. for cholesky decomps.
    import matplotlib.pyplot as plt
    from jax.config import config
    config.update("jax_enable_x64", True)

    train_ds = get_datasets()
    trained_model = train(train_ds)


if __name__ == '__main__':
    app.run(main)