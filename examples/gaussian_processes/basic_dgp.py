from jax.config import config; config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from flax import nn, optim
from jax import random, ops

import kernels
import likelihoods
import gaussian_processes
import inducing_variables

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.0001,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_integer(
    'num_epochs', default=10000,
    help=('Number of training epochs.'))

flags.DEFINE_bool(
    'plot', default=True,
    help=('Plot the results.',))


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
        obs_noise_scale = 0.1
        variational_distribution = vgp.marginal()
        return likelihoods.GaussianLogLik(
            variational_distribution.mean,
            variational_distribution.scale, obs_noise_scale)


class DeepGPModel(nn.Module):
    def apply(self, x, **kwargs):

        vgps = {}

        for layer in range(1, 3):

            if layer == 1:
                mean_fn = lambda x_: jnp.zeros(x_.shape[:-1])
            else:
                mean_fn = lambda x_: x_[..., 0]

            kern_fn = kernels.RBFKernelProvider(
                x, name='kernel_fun_{}'.format(layer),
                **kwargs.get('kernel_fun_{}_kwargs'.format(layer), {}))

            inducing_var = inducing_variables.InducingPointsProvider(
                x,
                kern_fn,
                name='inducing_var_{}'.format(layer),
                num_inducing_points=7,
                **kwargs.get('inducing_var_{}_kwargs'.format(layer), {}))

            vgp = gaussian_processes.SVGPProvider(
                x, mean_fn, kern_fn, inducing_var, name='vgp_{}'.format(layer))

            #pz_zprev = vgp.marginal()
            #x = pz_zprev.sample(nn.make_rng())[:, None]
            x = vgp.mean_function(x)[..., None]

            vgps[layer] = vgp

        ell = LikelihoodProvider(vgp, name='ell')

        return ell, vgps


def create_model(key, input_shape):

    def inducing_loc_init(key, shape):
        return jnp.linspace(-1.5, 1.5, 7)[:, None]

    kwargs = {}
    for i in range(1, 3):
        kwargs['kernel_fun_{}_kwargs'.format(i)] = {
            'amplitude_init': lambda key, shape: jnp.ones(shape),
            'length_scale_init': lambda key, shape: .1 * jnp.ones(shape)}
        kwargs['inducing_var_{}_kwargs'.format(i)] = {
            #'num_inducing_points': 7,
            'fixed_locations': False,
            'inducing_locations_init': inducing_loc_init}


    with nn.stochastic(key):
        _, params = DeepGPModel.init_by_shape(
            key,
            [(input_shape, jnp.float64), ],
            **kwargs)

        return nn.Model(DeepGPModel, params)


def create_optimizer(model, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(model)
    return optimizer


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""

    def loss_fn(model):
        ell, vgps = model(batch['index_points'])
        prior_kl = jnp.sum([item.prior_kl() for _, item in vgps.items()])
        return -ell.variational_expectation(batch['y']) + prior_kl

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

    with nn.stochastic(rng):
        model = create_model(rng, (15, 1))
        optimizer = create_optimizer(model, FLAGS.learning_rate, 0.33)

        for epoch in range(1, num_epochs + 1):
            optimizer, metrics = train_epoch(
                optimizer, train_ds, epoch)

    return optimizer


def step_fun(x):
    if x <= 0.:
        return -1.
    else:
        return 1.


def get_datasets():
    rng = random.PRNGKey(123)
    index_points = jnp.linspace(-1.5, 1.5, 15)
    y = (jnp.array([step_fun(x) for x in index_points])
         + 0.1*random.normal(rng, index_points.shape))
    train_ds = {'index_points': index_points[..., None], 'y': y}
    return train_ds


def main(_):

    train_ds = get_datasets()
    optimizer = train(train_ds)

    if FLAGS.plot:
        import matplotlib.pyplot as plt

        model = optimizer.target

        xx_pred = jnp.linspace(-1.5, 1.5)[:, None]

        fig, ax = plt.subplots()

        with nn.stochastic(random.PRNGKey(123)):
            y, vgps = model(xx_pred)
        vgp = vgps[2]

        pred_m = vgp.mean_function(xx_pred)
        pred_v = jnp.diag(vgp.kernel_function(xx_pred, xx_pred))

        ax.plot(xx_pred[:, 0], pred_m, '-')

        ax.step(xx_pred, [step_fun(x) for x in xx_pred], 'k--')
        ax.plot(train_ds['index_points'][:, 0], train_ds['y'], 'ks')
        plt.show()


if __name__ == '__main__':
    app.run(main)