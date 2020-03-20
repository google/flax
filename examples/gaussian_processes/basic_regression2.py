from absl import app
from absl import flags
from absl import logging

from flax import nn

import jax
from jax import random, ops
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import scipy as oscipy
import kernels
import distributions

from jax.config import config
config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'plot', default=True,
    help=('Plot the results.', ))


def _diag_shift(mat, val):
    """ Shifts the diagonal of mat by val. """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)


class MeanShiftDistribution(nn.Module):
    """ Shifts the mean of a distribution with `mean` field. """
    def apply(self, p, shift):
        """ Shift the mean of `p` by `shift`.

        Args:
            p: `dataclass` with field `mean`
            shift: `jnp.ndarray` shift vector, should broadcast with
              p.mean

        Returns:
            pnew: A new object of the same type as `p` with
              `pnew.mean = p.mean + shift`.
        """
        try:
            return p.replace(mean=p.mean + shift)
        except AttributeError:
            AttributeError('{} must have a `mean` field.'.format(p))


class GaussianProcessLayer(nn.Module):
    """ Given index points the role is to provide
    finite dimensional distributions.

    This implementation handles very little with regards to
    parameterisations.
    """
    def apply(self,
              index_points,
              kernel_fun,
              mean_fun=None,
              jitter=1e-4):
        """

        Args:
            index_points:
            kernel_fun:
            mean_fun: Default: jnp.zeros

        Returns:
            p: `distributions.MultivariateNormalTriL` object.
        """
        if mean_fun is None:
            mean_fun = lambda x: jnp.zeros(x.shape[:-1], dtype=index_points.dtype)

        mean = mean_fun(index_points)
        cov = kernel_fun(index_points)
        cov = _diag_shift(cov, jitter)

        return distributions.MultivariateNormalTriL(
            mean, jnp.linalg.cholesky(cov))


class RBFKernelProvider(nn.Module):
    """ Provides an RBF Kernel to be used inside more functional Modules

    The role of a kernel provider is to handle initialisation, and
    parameter storage of a particular kernel function. Allowing
    functionally defined kernels to be slotted into more complex models
    built using the Flax functional api.
    """
    def apply(self, x,
              amplitude_init=jax.nn.initializers.ones,
              lengthscale_init=jax.nn.initializers.ones):
        """

        Args:
            x:
            amplitude_init:
            lengthscale_init:

        Returns:

        """
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


class MarginalObservationModel(nn.Module):
    """ p(y|x, {hyper par}) = ∫p(y, f | x),
    where f(x) ~ GP() """
    def apply(self, pf):
        """ Applys the marginal observation model of the conditional
        y | f ~ prod N(yi | fi, obs_noise_scale**2) when f ~ pf

        Args:
            pf: `distribution.MultivariateNormal` object.

        Returns:
            py: `distributions.MultivariateNormal` object, the
              marginalised distribution of f.
        """
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale',
                       (), jax.nn.initializers.ones))

        covariance = pf.scale @ pf.scale.T
        covariance = _diag_shift(covariance, obs_noise_scale**2)

        return distributions.MultivariateNormalFull(
            pf.mean, covariance)


class GPModel(nn.Module):
    """ Model for i.i.d noise observations from a GP with
    RBF kernel. """
    def apply(self, x, dtype=jnp.float64):
        """

        Args:
            x: Index points of the observations.
            dtype:

        Returns:
            py_x: Distribution of the observations at the index points.
        """
        kern_fun = RBFKernelProvider(x, name='kernel_fun')
        pf_x = GaussianProcessLayer(x, kern_fun, name='gp_layer')
        #linear_mean = nn.Dense(x, features=1, name='linear_mean',
        #                       dtype=dtype)
        #pf_x = MeanShiftDistribution(
        #    pf_x, linear_mean[..., 0], name='mean_shift')

        py_x = MarginalObservationModel(pf_x, name='observation_model')
        return py_x


def build_par_pack_and_unpack(model):
    """ Build utility functions to pack and unpack paramater pytrees
    for the scipy optimizers. """
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


def get_datasets(sim_key):
    """ Generate the datasets. """
    index_points = jnp.linspace(-3., 3., 15)[..., jnp.newaxis]
    linear_trend = 0. #0.33 + .1 * index_points[:, 0]
    y = (linear_trend
         + jnp.sin(index_points[:, 0]*2)
         + .5 * random.normal(sim_key, index_points.shape[:-1]))
    train_ds = {'index_points': index_points,
                'y': y}
    return train_ds


def train(train_ds):
    """ Complete training of the GP-Model.

    Args:
        train_ds: Python `dict` with entries `index_points` and `y`.

    Returns:
        trained_model: A `GPModel` instance with trained hyper-parameters.

    """
    rng = random.PRNGKey(0)

    # initialise the model
    py, params = GPModel.init(rng, train_ds['index_points'])
    model = nn.Model(GPModel, params)

    # utility functions for packing and unpacking param dicts
    par_from_array, array_from_par = build_par_pack_and_unpack(model)

    def loss_fun(model, params):
        py = model.module.call(params, train_ds['index_points'])
        return -py.log_prob(train_ds['y'])

    # wrap loss fun for scipy.optimize
    def wrapped_loss_fun(arr):
        params = par_from_array(arr)
        return loss_fun(model, params)

    @jax.jit
    def loss_and_grads(x):
        return jax.value_and_grad(wrapped_loss_fun)(x)

    res = oscipy.optimize.minimize(
        loss_and_grads,
        x0=array_from_par(params),
        jac=True,
        method='BFGS')

    print(res)

    logging.info('Optimisation message: {}'.format(res.message))

    trained_model = model.replace(params=par_from_array(res.x))
    return trained_model


def main(_):
    train_ds = get_datasets(random.PRNGKey(123))
    trained_model = train(train_ds)
    print(trained_model.params)

    if FLAGS.plot:

        obs_noise_scale = jax.nn.softplus(
            trained_model.params['observation_model']['observation_noise_scale'])

        def learned_kernel_fn(x1, x2):
            return RBFKernelProvider.call(
                trained_model.params['kernel_fun'], x1)(x1, x2)

        def learned_mean_fn(x):
            return jnp.zeros(x.shape[:-1])
            return nn.Dense.call(
                trained_model.params['linear_mean'], x, features=1)[:, 0]

        xx_new = jnp.linspace(-3., 3., 100)[:, None]

        # prior GP model at learned model parameters
        fitted_gp = distributions.GaussianProcess(
            train_ds['index_points'],
            learned_mean_fn,
            learned_kernel_fn, 1e-4
        )
        posterior_gp = fitted_gp.posterior_gp(
                train_ds['y'],
                xx_new,
                obs_noise_scale**2)

        pred_f_mean = posterior_gp.mean_function(xx_new)
        pred_f_var = jnp.diag(posterior_gp.kernel_function(xx_new, xx_new))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.fill_between(xx_new[:, 0],
                        pred_f_mean - 2*jnp.sqrt(pred_f_var),
                        pred_f_mean + 2*jnp.sqrt(pred_f_var), alpha=0.5)
        ax.plot(xx_new, posterior_gp.mean_function(xx_new), '-')
        ax.plot(train_ds['index_points'], train_ds['y'], 'ks')

        plt.show()


if __name__ == '__main__':
    app.run(main)
