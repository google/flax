import argparse
import math
import numpy
import pickle

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

IMG_LENGTH = 28
LEARNING_RATE = 0.001

KL_WARM_UP_STEPS = 1000
KL_LOSS_SCALING = 10e-3

KEY = jax.random.PRNGKey(0)

@jax.vmap
def draw_sample(z_mean, z_log_var):
    epsilon = jax.random.normal(KEY, z_mean.shape)
    sample = z_mean + (jax.numpy.exp(0.5 * z_log_var) * epsilon)
    return sample


class Encoder(flax.nn.Module):
    def apply(self, x):
        x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
        x = jax.nn.relu(x)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
        x = flax.nn.relu(x)
        x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = flax.nn.Dense(x, features=16)
        x = flax.nn.relu(x)

        # z_mean = flax.nn.Dense(x, features=2, name="z_mean")
        z_log_var = flax.nn.Dense(x, features=2, name="z_log_var") # only learning the variance as per https://openreview.net/pdf?id=r1xaVLUYuE
        z_mean = jnp.zeros(z_log_var.shape)
        sample = draw_sample(z_mean, z_log_var)

        return z_mean, z_log_var, sample


class Decoder(flax.nn.Module):
    """
    Decoder adapted from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
    """

    def apply(self, x):
        x = flax.nn.Dense(x, features=7 * 7 * 64, name="decoder_input")
        x = flax.nn.relu(x)
        x = x.reshape((-1, 7, 7, 64))
        x = flax.nn.Conv(x, features=64, kernel_size=(2, 2), lhs_dilation=(2, 2), padding=[(1, 1), (1, 1)])
        x = flax.nn.relu(x)
        x = flax.nn.Conv(x, features=32, kernel_size=(2, 2), lhs_dilation=(2, 2), padding=[(1, 1), (1, 1)])
        x = flax.nn.relu(x)
        x = flax.nn.Conv(x, features=1, kernel_size=(3, 3), name="generated_image")

        generated_image = flax.nn.sigmoid(x)

        return generated_image


class VAE(flax.nn.Module):
    def apply(self, x):
        z_mean, z_log_var, sample = Encoder(x, name="encoder")
        decoded = Decoder(sample, name="decoder")
        return z_mean, z_log_var, decoded


@jax.vmap
def mse(decoded, original_image):
    return ((decoded - original_image) ** 2).mean(axis=(0, 1))


@jax.vmap
def kl_loss(z_mean, z_log_var):
    loss = 1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)
    loss = jnp.sum(loss, axis=-1)
    loss *= -0.5

    return loss


def compute_loss(step, decoded, original_image, z_mean, z_log_var):
    error = mse(decoded, original_image)
    kl = kl_loss(z_mean, z_log_var) * KL_LOSS_SCALING

    # specify the warmup condition as a lax function so that jit() still works
    kl = kl * jnp.clip(step - KL_WARM_UP_STEPS, 0, 1)

    return jnp.mean(error + kl)


@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        z_mean, z_log_var, decoded = model(batch["image"])
        loss = compute_loss(optimizer.state.step, decoded, batch['image'], z_mean, z_log_var)
        return loss, decoded

    optimizer, _, _ = optimizer.optimize(loss_fn)
    return optimizer


@jax.jit
def eval(model, eval_ds):
    z_mean, z_log_var, decoded = model(eval_ds)
    error = mse(decoded, eval_ds)
    kl = kl_loss(z_mean, z_log_var)
    return {'total_loss': jnp.mean(error + kl), "kl_loss": jnp.mean(kl), "mse": jnp.mean(error)}


def print_metrics(epoch, metrics):
    print('eval epoch: %d, total_loss: %.10f, kullback-leibler: %.10f, mse: %.6f'
          % (epoch, metrics['total_loss'], metrics['kl_loss'], metrics['mse']))


def train(num_epochs=10):
    train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
    train_ds = train_ds.cache().shuffle(1000).batch(128)
    validation_ds = tfds.as_numpy(tfds.load(
        'mnist', split=tfds.Split.TEST, batch_size=-1))

    _, model = VAE.create_by_shape(
        KEY,
        [((1, IMG_LENGTH, IMG_LENGTH, 1), jnp.float32)])

    optimizer = flax.optim.Adam(
        learning_rate=LEARNING_RATE).create(model)

    for epoch in range(num_epochs):
        for batch in tfds.as_numpy(train_ds):
            batch['image'] = batch['image'] / 255.0
            optimizer = train_step(optimizer, batch)

        metrics = eval(optimizer.target, validation_ds['image'] / 255.0)
        print_metrics(epoch, metrics)

    return optimizer


def visualize_latent_space(samples, labels):
    plt.figure(figsize=(10, 10))
    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x, y, c=labels, label=labels)
    plt.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A variational auto-encoder for the MNIST dataset ')
    parser.add_argument("--model_path", default="vae.pickle", help="Where the model is (de)serialized to/from")
    parser.add_argument("--train", default=False, action="store_true", help="Train the model")
    parser.add_argument("--test", default=False, action="store_true", help="")
    parser.add_argument("--epochs", default=10, type=int, help="How many epochs to train")
    parser.add_argument("--sample_output_path", default="output", help="Where to store the sample outputs")
    parser.add_argument("--num_samples", default=10, type=int, help="How often to sample the decoder")

    args = parser.parse_args()

    if args.train:
        optimizer = train(args.epochs)
        model = optimizer.target
        state_dict = flax.serialization.to_state_dict(model)
        with open(args.model_path, 'wb') as handle:
            print("Pickling to %s" % args.model_path)
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.model_path, 'rb') as handle:
            print("Loading model from %s" % args.model_path)
            state_dict = pickle.load(handle)
            _, model = VAE.create_by_shape(
                jax.random.PRNGKey(0),
                [((1, IMG_LENGTH, IMG_LENGTH, 1), jnp.float32)])
            model = flax.serialization.from_state_dict(model, state_dict)

    test_ds = tfds.as_numpy(tfds.load(
        'mnist', split=tfds.Split.TEST, batch_size=-1))
    test_ds['image'] = test_ds['image'] / 255.0

    if args.test:
        test_metrics = eval(model, test_ds['image'])
        print_metrics(-1, test_metrics)

    _, _, encoded_samples = Encoder.call(model.params['encoder'], test_ds['image'])
    visualize_latent_space(encoded_samples, test_ds['label'])

    grid_length = math.ceil(math.sqrt(args.num_samples))
    x_interval = jnp.linspace(jnp.amin(encoded_samples[:, 0]), jnp.amax(encoded_samples[:, 0]), grid_length)
    y_interval = jnp.linspace(jnp.amin(encoded_samples[:, 1]), jnp.amax(encoded_samples[:, 1]), grid_length)
    gen_image = Decoder.call(model.params['decoder'], encoded_samples)

    image_grid = numpy.zeros((grid_length * IMG_LENGTH, grid_length * IMG_LENGTH))
    for i in range(0, grid_length):
        for j in range(0, grid_length):
            gen_image_idx = i * grid_length + j
            if gen_image_idx > len(gen_image):
                break

            fig_row_start = i * IMG_LENGTH
            fig_row_end = (i + 1) * IMG_LENGTH
            fig_col_start = j * IMG_LENGTH
            fig_col_end = (j + 1) * IMG_LENGTH
            img = gen_image[gen_image_idx].reshape(IMG_LENGTH, IMG_LENGTH)
            image_grid[fig_row_start:fig_row_end, fig_col_start:fig_col_end] = img

    image_grid *= 255.0

    plt.figure(figsize=(10, 10))
    plt.imshow(image_grid)
    plt.xticks(plt.xticks()[0], x_interval)
    plt.yticks(plt.yticks()[0], y_interval)
    plt.show()
