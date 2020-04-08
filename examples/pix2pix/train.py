import functools

import jax

from examples.pix2pix.input_pipeline import BATCH_SIZE, IMG_HEIGHT, get_dataset
from examples.pix2pix.models import Generator, Discriminator
import flax

import jax.numpy as jnp

LAMBDA = 100


# create model
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, batch_size, image_size, model_def):
  input_shape = (batch_size, image_size, image_size, 3)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(key, [
        (input_shape, jnp.float32)])
      model = flax.nn.Model(model_def, initial_params)
  return model, init_state


def create_optimizer(model, learning_rate, beta):
  optimizer_def = flax.optim.Adam(learning_rate=learning_rate,
                                  beta1=beta)
  optimizer = optimizer_def.create(model)
  optimizer = flax.jax_utils.replicate(optimizer)
  return optimizer


@jax.vmap
def binary_cross_entropy_loss(x, y):
  max_val = jnp.clip(x, 0, None)
  loss = x - x * y + max_val + jnp.log(
    jnp.exp(-max_val) + jnp.exp((-x - max_val)))
  return loss.mean()


@jax.vmap
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = binary_cross_entropy_loss(jnp.ones_like(disc_generated_output),
                                       disc_generated_output)

  l1_loss = jnp.mean(jnp.absolute(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  # think about negative
  return total_gen_loss, gan_loss, l1_loss


@jax.vmap
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = binary_cross_entropy_loss(jnp.ones_like(disc_real_output),
                                        disc_real_output)
  generated_loss = binary_cross_entropy_loss(
    jnp.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


@jax.jit
def train_step(generator_opt, discriminator_opt, input_image, target_image):
  """Perform a single training step."""

  def loss_fn(gen_model, disc_model):
    """loss function used for training."""
    # with flax.nn.stateful(state) as new_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      gen_output = gen_model(input_image)

      disc_real_output = disc_model(
        jnp.concatenate((input_image, target_image)))
      disc_generated_output = disc_model(
        jnp.concatenate((input_image, gen_output)))

    gen_total_loss, _, _ = generator_loss(disc_generated_output, gen_output,
                                          target_image)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_total_loss, disc_loss

  step = generator_opt.state.step
  gen_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  disc_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  _, gen_grad = gen_grad_fn(generator_opt.target, discriminator_opt.target)
  (gen_total_loss, disc_loss), disc_grad = disc_grad_fn(generator_opt.target,
                                                        discriminator_opt.target)

  new_gen_opt = generator_opt.apply_gradient(gen_grad)
  new_disc_opt = discriminator_opt.apply_gradient(disc_grad)

  return new_gen_opt, new_disc_opt


def train():
  train_dataset, test_dataset = get_dataset()

  key = jax.random.PRNGKey(0)
  generator_model, generator_state = create_model(key, BATCH_SIZE, IMG_HEIGHT,
                                                  Generator)
  discriminatogr_model, discriminator_state = create_model(key, BATCH_SIZE,
                                                           IMG_HEIGHT,
                                                           Discriminator)

  generator_optimizer = create_optimizer(generator_model, 2e-4, 0.5)
  discriminator_optimizer = create_optimizer(discriminator_model, 2e-4, 0.5)
  p_train_step = jax.pmap(functools.partial(train_step), axis_name='batch')
