# Annotated full end-to-end MNIST example

```py
import jax
import flax
import tensorflow as tf
```

Load vanilla NumPy for use on host.

```py
import numpy as onp
```

JAX has a re-implemented NumPy that runs on GPU and TPU

```py
import jax.numpy as jnp
```

Flax can use any data loading pipeline. We use TF datasets.

```py
import tensorflow_datasets as tfds
```

A Flax "module" lets you write a normal function, which
defines learnable parameters in-line. In this case,
we define a simple convolutional neural network.

Each call to `flax.nn.Conv` defines a learnable kernel.

```py
class CNN(flax.nn.Module):
  def apply(self, x):
    x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
    x = flax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
    x = flax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, features=256)
    x = flax.nn.relu(x)
    x = flax.nn.Dense(x, features=10)
    x = flax.nn.log_softmax(x)
    return x
```

`jax.vmap` allows us to define the `cross_entropy_loss`
function as if it acts on a single sample. `jax.vmap`
automatically vectorizes code efficiently to run on entire
batches.

```py
@jax.vmap
def cross_entropy_loss(logits, label):
  return -logits[label]
```

Compute loss and accuracy. We use `jnp` (`jax.numpy`) which can run on
device (GPU or TPU).

```py
def compute_metrics(logits, labels):
  loss = jnp.mean(cross_entropy_loss(logits, labels))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return {'loss': loss, 'accuracy': accuracy}
```

`jax.jit` traces the `train_step` function and compiles into fused device
operations that run on GPU or TPU.

```py
@jax.jit
def train_step(optimizer, batch):
  def loss_fn(model):
    logits = model(batch['image'])
    loss = jnp.mean(cross_entropy_loss(
        logits, batch['label']))
    return loss
  grad = jax.grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer
```

Making model predictions is as simple as calling `model(input)`:

```py
@jax.jit
def eval(model, eval_ds):
  logits = model(eval_ds['image'] / 255.0)
  return compute_metrics(logits, eval_ds['label'])
```

## Main train loop

```py
def train():
```

Load, convert dtypes, and shuffle MNIST.

```py
  train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
  train_ds = train_ds.map(lambda x: {'image':tf.cast(x['image'], tf.float32),
                                     'label':tf.cast(x['label'], tf.int32)})
  train_ds = train_ds.cache().shuffle(1000).batch(128)
  test_ds = tfds.as_numpy(tfds.load(
      'mnist', split=tfds.Split.TEST, batch_size=-1))
  test_ds = {'image': test_ds['image'].astype(jnp.float32),
             'label': test_ds['label'].astype(jnp.int32)}
```

Create a new model, running all necessary initializers.

The parameters are stored as nested dicts on `model.params`.

```py
  _, initial_params = CNN.init_by_shape(
  jax.random.PRNGKey(0),
   [((1, 28, 28, 1), jnp.float32)])
  model = flax.nn.Model(CNN, initial_params)
```

Define an optimizer. At any particular optimzation step,
`optimizer.target` contains the model.

```py
  optimizer = flax.optim.Momentum(
      learning_rate=0.1, beta=0.9).create(model)
```

Run an optimization step for each batch of training

```py
  for epoch in range(10):
    for batch in tfds.as_numpy(train_ds):
      batch['image'] = batch['image'] / 255.0
      optimizer = train_step(optimizer, batch)
```

Once an epoch, evaluate on the test set.

```py
    metrics = eval(optimizer.target, test_ds)
```

`metrics` are only retrieved from device when needed on host
(like in this `print` statement)

```py
    print('eval epoch: %d, loss: %.4f, accuracy: %.2f'
         % (epoch+1,
          metrics['loss'], metrics['accuracy'] * 100))
```

