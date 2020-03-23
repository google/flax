from jax.config import config; config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from flax import nn, optim
from jax import random

import kernels
import distributions
import gaussian_processes
import inducing_variables

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.01,
    help=('The learning rate for the adam optimizer.'))

flags.DEFINE_float(
    'beta1', default=0.9,
    help=('The beta1 parameter of the adam optimizer.'))

flags.DEFINE_integer(
    'num_epochs', default=1000,
    help=('Number of training epochs.'))

flags.DEFINE_integer(
    'num_samples', default=10,
    help=('Number of samples to approximate the ELBO.'))

flags.DEFINE_bool(
    'plot', default=False,
    help=('Plot the results.',))

flags.DEFINE_integer(
    'num_inducing_points', default=10,
    help=('Number of inducing points epochs.'))


class LikelihoodProvider(nn.Module):
    def apply(self,
              x: jnp.ndarray) -> distributions.MultivariateNormal:
        """

        Args:
            x: nd-array

        Returns:
            ll:

        """
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale',
                       (1, ),
                       lambda key, shape: 1.0e-1*jnp.ones([1])))
        return distributions.MultivariateNormalDiag(
            mean=x[..., 0], scale_diag=jnp.ones(x.shape[:-1])*obs_noise_scale)


class DeepGPModel(nn.Module):
    def apply(self, x, sample_key, **kwargs):
        """

        Args:
            x: nd-array input index points for the Deep GP model.
            sample_key: random number generator for stochastic inference.
            **kwargs: additional kwargs passed to layers.

        Returns:
            loglik: The output observation model.
            vgps: Intermediate variational GP outputs for each layer
        """
        vgps = {}

        mf = lambda x_: jnp.zeros(x_.shape[:-1])  # initial mean_fun
        for layer in range(1, 3):
            kf = kernels.RBFKernelProvider(
                x, name='kernel_fn_{}'.format(layer),
                **kwargs.get('kernel_fn_{}_kwargs'.format(layer), {}))

            inducing_var = inducing_variables.InducingPointsProvider(
                x,
                kf,
                name='inducing_var_{}'.format(layer),
                num_inducing_points=FLAGS.num_inducing_points,
                **kwargs.get('inducing_var_{}_kwargs'.format(layer), {}))

            vgp = gaussian_processes.SVGPProvider(
                x, mf, kf,
                inducing_var,
                name='vgp_{}'.format(layer))

            x = vgp.marginal().sample(sample_key)[..., jnp.newaxis]
            vgps[layer] = vgp

            mf = lambda x_: x_[..., 0]  # identity mean_fn for later layers.

        loglik = LikelihoodProvider(x, name='loglik')

        return loglik, vgps


def create_model(key, input_shape):

    def inducing_loc_init(key, shape):
        return jnp.linspace(-1.5, 1.5, FLAGS.num_inducing_points)[:, jnp.newaxis]

    kwargs = {}
    for i in range(1, 3):
        kwargs['kernel_fn_{}_kwargs'.format(i)] = {
            'amplitude_init': lambda key, shape: jnp.ones(shape),
            'length_scale_init': lambda key, shape: jnp.ones(shape)}
        kwargs['inducing_var_{}_kwargs'.format(i)] = {
            'fixed_locations': False,
            'whiten': True,
            'inducing_locations_init': inducing_loc_init}

    model_def = DeepGPModel.partial(**kwargs)

    with nn.stochastic(key):
        _, params = model_def.init_by_shape(
            key,
            [(input_shape, jnp.float64), ],
            nn.make_rng(),
            **kwargs)

        return nn.Model(model_def, params)


def create_optimizer(model, learning_rate, beta1):
    optimizer_def = optim.Adam(learning_rate=learning_rate, beta1=beta1)
    optimizer = optimizer_def.create(model)
    return optimizer


@jax.jit
def multi_sample_train_step(optimizer, batch, sample_keys):

    def loss_fn(model):
        def single_sample_loss(key):
            ell, vgps = model(batch['index_points'], key)
            return -ell.log_prob(batch['y']) + jnp.sum([vgp.prior_kl() for _, vgp in vgps.items()])
        return jnp.mean(jax.vmap(single_sample_loss)(sample_keys))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(optimizer.target)

    optimizer = optimizer.apply_gradient(grad)
    metrics = {'loss': loss}
    return optimizer, metrics


def train_epoch(optimizer, train_ds, epoch, sample_key):
    """Train for a single epoch."""
    optimizer, epoch_metrics = multi_sample_train_step(optimizer, train_ds, sample_key)
    epoch_metrics_np = jax.device_get(epoch_metrics)

    logging.info('train epoch: %d, loss: %.4f',
                 epoch,
                 epoch_metrics_np['loss'])

    return optimizer, epoch_metrics_np


def train(train_ds):
    rng = random.PRNGKey(0)

    with nn.stochastic(rng):
        model = create_model(rng, train_ds['index_points'].shape)
        optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.beta1)

        key = nn.make_rng()

        for epoch in range(1, FLAGS.num_epochs + 1):
            key = random.split(key, FLAGS.num_samples + 1)
            key, sample_key = (key[0], key[1:])
            optimizer, metrics = train_epoch(
                optimizer, train_ds, epoch, sample_key)

    return optimizer


def step_fn(x):
    if x <= 0.:
        return -1.
    else:
        return 1.


def get_datasets():
    rng = random.PRNGKey(123)
    index_points = jnp.linspace(-1.5, 1.5, 25)
    y = (jnp.array([step_fn(x) for x in index_points])
         + 0.1*random.normal(rng, index_points.shape))
    train_ds = {'index_points': index_points[..., jnp.newaxis], 'y': y}
    return train_ds


def main(_):

    train_ds = get_datasets()
    optimizer = train(train_ds)

    if FLAGS.plot:
        import matplotlib.pyplot as plt

        xx_pred = jnp.linspace(-1.5, 1.5)[:, jnp.newaxis]

        num_samples = 50
        subkeys = random.split(random.PRNGKey(0), num=num_samples)

        def sample(skey):
            ll, vgps = optimizer.target(xx_pred, skey)
            return ll.mean

        samples = jax.vmap(sample)(subkeys)
        pred_m = jnp.mean(samples, axis=0)
        pred_sd = jnp.std(samples, axis=0)

        fig, ax = plt.subplots()

        ax.plot(xx_pred[:, 0], pred_m, 'C0-',
                label=r'$\mathbb{E}_{f \sim q(f)}[f(x)]$')
        ax.fill_between(xx_pred[:, 0],
                        pred_m - 2 * pred_sd,
                        pred_m + 2 * pred_sd, alpha=0.5, label='true function')

        ax.step(xx_pred[:, 0], [step_fn(x) for x in xx_pred], 'k--', alpha=0.7)
        ax.plot(train_ds['index_points'][:, 0], train_ds['y'],
                'ks', label='observations')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    app.run(main)
