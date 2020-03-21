from jax.config import config;

config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from flax import nn, optim
from jax import random, ops
from typing import Callable, Union

import kernels
import distributions
import inducing_variables
import gaussian_processes
import likelihoods

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.001,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'num_epochs', default=1000,
    help=('Number of training epochs.'))

flags.DEFINE_bool(
    'plot', default=True,
    help=('Plot the results.',))


def _diag_shift(mat: jnp.ndarray,
                val: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """ Shifts the diagonal of mat by val. """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)


class LikelihoodProvider(nn.Module):
    def apply(self,
              vgp: gaussian_processes.VariationalGaussianProcess) -> likelihoods.GaussianLogLik:
        """

        Args:
            vgp: variational Gaussian process regression model q(f).

        Returns:
            ll: log-likelihood model with method `variational_expectations` to
              compute ∫ log p(y|f) q(f) df
        """
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale',
                       (),
                       jax.nn.initializers.ones))
        variational_distribution = vgp.marginal()
        return likelihoods.GaussianLogLik(
            variational_distribution.mean,
            variational_distribution.scale, obs_noise_scale)


class SVGPProvider(nn.Module):
    def apply(self,
              index_points,
              mean_fn,
              kernel_fn,
              inducing_var,
              jitter=1e-4):
        """

        Args:
            index_points: the nd-array of index points of the GP model.
            mean_fn: callable mean function of the GP model.
            kernel_fn: callable kernel function.
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`.
            jitter: float `jitter` term to add to the diagonal of the covariance
              function before computing Cholesky decompositions.

        Returns:
            svgp: A sparse Variational GP model.
        """
        qu = inducing_var.variational_distribution
        z = inducing_var.locations

        var_kern = kernels.VariationalKernel(
            kernel_fn, z, qu.scale)

        def var_mean(x_):
            kzz_chol = jnp.linalg.cholesky(
                _diag_shift(kernel_fn(z, z), jitter))

            kxz = kernel_fn(x_, z)
            dev = (qu.mean - mean_fn(z))[..., None]
            return (mean_fn(x_)[..., None]
                    + kxz @ jscipy.linalg.cho_solve(
                        (kzz_chol, True), dev))[..., 0]

        return gaussian_processes.VariationalGaussianProcess(
            index_points, var_mean, var_kern, jitter, inducing_var)


class SVGPModel(nn.Module):
    def apply(self, x, inducing_locations_init, **kwargs):
        """

        Args:
            x: the nd-array of index points of the GP model
            inducing_locations_init: initializer function for the inducing
              variable locations.

        Returns:
            ell: variational likelihood object.
            vgp: the variational GP q(f) = ∫p(f|u)q(u)du where
              `q(u) == inducing_var.variational_distribution`.
        """
        kern_fun = kernels.RBFKernelProvider(
            x, name='kernel_fun', **kwargs.get('kernel_fun_kwargs', {}))
        inducing_var = inducing_variables.InducingPointsProvider(
            x,
            kern_fun,
            inducing_locations_init=inducing_locations_init,
            name='inducing_var')

        vgp = gaussian_processes.SVGPProvider(x,
                                              lambda x_: jnp.zeros(x_.shape[:-1]),
                                              kern_fun,
                                              inducing_var, name='vgp')

        ell = LikelihoodProvider(vgp, name='ell')

        return ell, vgp


def create_model(key, input_shape):
    def inducing_loc_init(key, shape):
        return random.uniform(key, shape, minval=-3., maxval=3.)

    # pass initializers as kwargs
    kernel_fun_kwargs = {
        'amplitude_init': lambda key, shape: jnp.ones(shape),
        'length_scale_init': lambda key, shape: .5 * jnp.ones(shape)}
    kwargs = {'kernel_fun_kwargs': kernel_fun_kwargs}

    _, params = SVGPModel.init_by_shape(
        key,
        [(input_shape, jnp.float64), ],
        inducing_locations_init=inducing_loc_init,
        **kwargs)

    return nn.Model(SVGPModel, params)


def create_optimizer(model, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(model)
    return optimizer


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""

    def inducing_loc_init(key, shape):
        return random.uniform(key, shape, minval=-3., maxval=3.)

    def loss_fn(model):
        ell, vgp = model(batch['index_points'], inducing_loc_init)
        return (-ell.variational_expectation(batch['y'])
                + vgp.prior_kl())

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = {'loss': loss}
    # metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics


def train_epoch(optimizer, train_ds, epoch):
    """Train for a single epoch."""
    optimizer, batch_metrics = train_step(optimizer, train_ds)
    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = batch_metrics_np
    # epoch_metrics_np = {
    #    k: onp.mean([metrics[k] for metrics in batch_metrics_np])
    #    for k in batch_metrics_np[0]}

    logging.info('train epoch: %d, loss: %.4f',
                 epoch,
                 epoch_metrics_np['loss'])

    return optimizer, epoch_metrics_np


def train(train_ds):
    rng = random.PRNGKey(0)

    num_epochs = FLAGS.num_epochs

    model = create_model(rng, (15, 1))
    optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)

    for epoch in range(1, num_epochs + 1):
        optimizer, metrics = train_epoch(
            optimizer, train_ds, epoch)

    return optimizer


def main(_):
    jnp.set_printoptions(precision=3, suppress=True)

    shape = (15, 1)
    index_points = jnp.linspace(-3., 3., shape[0])[:, None]

    rng = random.PRNGKey(123)

    y = (jnp.sin(index_points)[:, 0]
         + 0.33 * random.normal(rng, (15,)))

    train_ds = {'index_points': index_points, 'y': y}

    optimizer = train(train_ds)

    if FLAGS.plot:
        import matplotlib.pyplot as plt
        model = optimizer.target

        def inducing_loc_init(key, shape):
            return random.uniform(key, shape, minval=-3., maxval=3.)

        xx_pred = jnp.linspace(-3., 5.)[:, None]

        _, vgp = model(xx_pred, inducing_loc_init)

        pred_m = vgp.mean_function(xx_pred)
        pred_v = jnp.diag(vgp.kernel_function(xx_pred, xx_pred))

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.fill_between(
            xx_pred[:, 0],
            pred_m - 2 * jnp.sqrt(pred_v),
            pred_m + 2 * jnp.sqrt(pred_v), alpha=0.5)
        ax.plot(xx_pred[:, 0], pred_m, '-',
                label=r'$\mathbb{E}_{f \sim q(f)}[f(x)]$')
        ax.plot(model.params['inducing_var']['locations'][:, 0],
                model.params['inducing_var']['mean'], '+',
                label=r'$E_{u \sim q(u)}[u]$')
        ax.plot(train_ds['index_points'][:, 0], train_ds['y'], 'ks', label='observations')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    app.run(main)
