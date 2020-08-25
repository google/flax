from flax.training import checkpoints
from input_pipeline import prepare_data
from jax import numpy as jnp
from model import create_retinanet
from typing import Any, Dict, Iterable, Mapping, Tuple

import flax
import jax
import math

_EPSILON =  1e-8


@flax.struct.dataclass
class State:
  """A dataclass which stores the state of the training loop.
  """
  # The state variable of the model
  model_state: flax.nn.Collection
  # The optimizer, which also holds the model
  optimizer: flax.optim.Optimizer
  # The global state of this checkpoint
  step: int = 0


def create_scheduled_decay_fn(learning_rate: float,
                              training_steps: int,
                              warmup_steps: int,
                              division_factor: float = 10.0,
                              division_schedule: list = None):
  """Creates a scheduled division based learning rate decay function.

  More specifically, produces a function which takes in a single parameter,
  the current step in the training process, and yields its respective learning
  rate based on a scheduled division based decay, which divides the previous
  learning rate by `division_factor` at the steps specified in the 
  `division_schedule`.

  Args:
    learning_rate: the base learning rate
    training_steps: the number of training steps 
    warmup_steps: the number of warmup steps 
    division_factor: the factor by which the learning rate is divided at the 
      training steps indicated by `division_schedule`
    division_schedule: a list which indicates the iterations at which the 
      learning rate should be divided by the `division_factor`. Note that
      the values in 

  Returns:
    A function, which takes in a single parameter, the global step in 
    the training process, and yields the current learning rate.
  """
  assert training_steps > 0, "training_steps must be greater than 0"
  assert warmup_steps >= 0, "warmup_steps must be greater than 0"
  assert division_factor > 0.0, "division_factor must be positive"

  # Get the default values for learning rate decay
  if division_schedule is None:
    division_schedule = [int(training_steps * .66), int(training_steps * .88)]

  # Adjust the schedule to not consider the warmup steps
  division_schedule = jnp.sort(jnp.unique(division_schedule)) + warmup_steps

  # Define the decay function
  def decay_fn(step: int) -> float:
    lr = learning_rate / division_factor**jnp.sum(division_schedule < step)

    # Linearly increase the learning rate during warmup
    return lr * jnp.minimum(1., step / warmup_steps)

  return decay_fn


def create_model(rng: jnp.ndarray,
                 depth: int = 50,
                 classes: int = 1000,
                 shape: Iterable[int] = (224, 224, 3),
                 dtype: jnp.dtype = jnp.float32) -> flax.nn.Model:
  """Creates a RetinaNet model.

  Args:
    rng: the Jax PRNG, which is used to instantiate the model weights
    depth: the depth of the basckbone network
    classes: the number of classes in the object detection task
    shape: the shape of the image inputs, with the format (H, W, C)
    dtype: the data type of the model

  Returns:
    The RetinaNet instance, and its state object
  """
  # The number of classes is increased by 1 since we add the background
  partial_module = create_retinanet(depth, classes=classes + 1, dtype=dtype)

  # Since the BatchNorm has state, we'll need to use stateful here
  with flax.nn.stateful() as init_state:
    _, params = partial_module.init(rng, jnp.zeros((1,) + shape))

  return flax.nn.Model(partial_module, params), init_state


def create_optimizer(model: flax.nn.Model,
                     optimizer: str = "momentum",
                     **optimizer_params) -> flax.optim.Optimizer:
  """Create either an Adam or Momentum optimizer.

  Args:
    model: a flax.nn.Model object, which encapsulates the neural network
    optimizer: this selects the optimizer, either `momentum` or `adam`
    **optimizer_params: extra kwargs for the created optimizer

  Returns:
    An optimizer which wraps the model
  """
  assert optimizer in ["momentum", "adam"], "The optimizer is not supported"

  if optimizer == "adam":
    optimizer_def = flax.optim.Adam(**optimizer_params)
  else:
    optimizer_def = flax.optim.Momentum(**optimizer_params)

  return optimizer_def.create(model)


@jax.vmap
def focal_loss(logits: jnp.array,
               label: int,
               anchor_type: int,
               alpha: float = 0.25,
               gamma: float = 2.0) -> float:
  """Implements the Focal Loss.

  Args:
    logits: an array of logits, with as many entries as candidate classes
    label: the ground truth label
    anchor_type: an integer which identifies the type of the anchor: 
      ignored (-1), background (0), foreground (1). If -1, the loss will be 0
    alpha: the value of the alpha constant in the Focal Loss
    gamma: the value of the gamma constant in the Focal Loss

  Returns:
    The value of the Focal Loss for this anchor.
  """
  # Only consider foreground (1) and background (0) anchors for this loss
  c = jnp.minimum(anchor_type + 1, 1)
  logit = jnp.maximum(_EPSILON, jnp.minimum(1.0 - _EPSILON, logits[label]))
  return c * -alpha * ((1 - logit)**gamma) * jnp.log(logit)


@jax.vmap
def smooth_l1(regressions: jnp.array, targets: jnp.array,
              anchor_type: int) -> float:
  """Implements the Smooth-L1 loss. 

  Args:
    regressions: an array of 4 elements containing the predicted regressions
    targets: an array of 4 elements containing the target regressions
    anchor_type: the type of the anchor whose predictions are being evaluated
  
  Returns:
    The value of the Smooth-L1 loss for this anchor.
  """
  # Only consider foreground (1) anchors for this loss
  c = jnp.maximum(anchor_type, 0)
  deltas = regressions - targets
  return c * jnp.sum(
      jnp.where(
          jnp.absolute(deltas) < 1.0, 0.5 * deltas**2.0,
          jnp.absolute(deltas) - 0.5))


@jax.vmap
def retinanet_loss(classifications: jnp.array,
                   regressions: jnp.array,
                   anchor_types: jnp.array,
                   classification_targets: jnp.array,
                   regression_targets: jnp.array,
                   reg_weight: float = 1.0) -> float:
  """Implements the loss for the RetinaNet: Focal Loss and Smooth-L1

  Args:
    predictions: a matrix of size (|A|, K), where |A| is the number of anchors
      and K is the number of classes, that cotains the model's predictions 
    regressions: a matrix of size (|A|, 4) containing the predictions of 
      the model for anchor location offsets
    anchor_types: an array of size |A| indicating the type of each anchor: 
      ignored (-1), background (0), foreground (1)
    classification_targets: an array of |A| elements, containing the ground 
      truth labels
    regression_targets: a matrix of (|A|, 4) elements, containing the ground
      truth regressions
    reg_weight: a scalar, which indicates the weight of the Smooth-L1 
      regularization term

  Returns:
    The image loss given by the RetinaNet loss function.
  """
  valid_anchors = jnp.maximum(1, jnp.sum(anchor_types > 0))
  fl = focal_loss(classifications, classification_targets, anchor_types)
  sl1 = smooth_l1(regressions, regression_targets, anchor_types)
  return jnp.sum(fl + sl1 * reg_weight) / valid_anchors


def compute_metrics(classifications: jnp.array, regressions: jnp.array,
                    bboxes: jnp.array,
                    data: Mapping[str, jnp.array]) -> Dict[str, float]:
  """Returns the accuracy and the cross entropy.

  Args:
    classifications: the classifications predicted by the model
    regressions: the regression matrix predicted by the model
    bboxes: the bboxes generated by the model, by applying the regressions to 
      the anchors
    data: a dictionary which maps from the name of the label to the list storing
      the ground truths

  Returns:
    A dictionary containins the metrics
  """
  metrics = {
      "retinanet_loss":
          retinanet_loss(classifications, regressions, data['anchor_type'],
                         data['classification_labels'],
                         data['regression_targets'])
  }
  return metrics


def eval(data: jnp.array, meta_state: State) -> Dict[str, float]:
  """Evaluates the model using the RetinaNet loss.

  Args:
    data: the test data
    model: an instance of the `State` class

  Returns:
    The accuracy and the Log-loss aggregated across multiple workers.
  """
  with flax.nn.stateful(meta_state.model_state, mutable=False):
    regressions, classifications, bboxes = meta_state.optimizer.target(
        data['image'], train=False)
  return compute_metrics(regressions, classifications, bboxes, data)


def aggregate_evals(eval_array):
  accumulator = {key: 0.0 for key in eval_array[0]}
  for d in eval_array:
    for k, v in d.items():
      accumulator[k] += v

  count = len(eval_array)
  for k in accumulator:
    accumulator[k] /= count

  return accumulator


def checkpoint_state(meta_state: State,
                     checkpoint_step: int,
                     checkpoint_dir: str = "checkpoints") -> None:
  """Checkpoints the training state.

  Args:
    meta_state: a `State` object, which contains the state of
      the current training step
    checkpoint_step: a checkpoint step, used for versioning the checkpoint
    checkpoint_dir: the directory where the checkpoint is stored
  """
  checkpoints.save_checkpoint(checkpoint_dir, meta_state, checkpoint_step)


def restore_checkpoint(meta_state: State,
                       checkpoint_dir: str = "checkpoints") -> State:
  """Restores the latest checkpoint.

  More specifically, either returns the latest checkpoint from the
  `checkpoint_dir` or returns the `meta_state` object, if no such checkpoint
  exists.

  Args:
    meta_state: a `State` object, used as last resort if no checkpoint
      exists
    checkpoint_dir: the directory where the checkpoints are searched for

  Returns:
    Either the latest checkpoint, if it exists, or the `meta_state` object.
  """
  return checkpoints.restore_checkpoint(checkpoint_dir, meta_state)


def create_step_fn(lr_function):
  """Creates a step function with a custom LR scheduler.

  Args:
    lr_function: function which takes in a single argument, the current step
      in the training process, and yields the learning rate

  Returns:
    A function responsible with carrying out a training step. The function takes
    in two arguments: the batch, and a `State` object, which
    stores the current training state.
  """

  def take_step(data: Mapping[str, jnp.array],
                meta_state: State) -> Tuple[State, Any]:
    """Trains the model on a batch and returns the updated model.

    Args:
      data: the batch on which the pass is performed
      meta_state: a `State` object, which holds the current model

    Returns:
      The updated model as a `State` object and the batch's loss
    """

    def _loss_fn(model: flax.nn.Model, state: flax.nn.Collection):
      with flax.nn.stateful(state) as new_state:
        classifications, regressions, _ = model(data['image'])
      loss = jnp.mean(
          retinanet_loss(classifications, regressions, data['anchor_type'],
                         data['classification_labels'],
                         data['regression_targets']))

      return loss, new_state

    # flax.struct.dataclass is immutable, so unwrap it
    step = meta_state.step + 1

    # Compute the gradients
    aux, grads = jax.value_and_grad(
        _loss_fn, has_aux=True)(meta_state.optimizer.target,
                                meta_state.model_state)
    loss, new_model_state = aux

    # Apply the gradients to the model
    updated_optimizer = meta_state.optimizer.apply_gradient(
        grads, learning_rate=lr_function(step))

    # Update the meta_state
    meta_state = meta_state.replace(
        step=step, model_state=new_model_state, optimizer=updated_optimizer)

    return meta_state, {"retinanet_loss": loss}

  return take_step


def train_retinanet_model(rng: jnp.array,
                          train_data: jnp.array,
                          test_data: jnp.array,
                          shape: list,
                          classes: int = 80,
                          depth: int = 50,
                          learning_rate: float = 0.1,
                          batch_size: int = 64,
                          training_steps: int = 90000,
                          warmup_steps: int = 30000,
                          try_restore: bool = True,
                          half_precision: bool = False,
                          checkpoint_period: int = 20000) -> State:
  """This method trains a RetinaNet instance.

  Args:
    train_data: a generator yielding batches of `batch_size`
    test_data: a generator yielding batches of `batch_size`
    shape: the shape of the padded images
    classes: the number of classes in the object detection task
    depth: the number of layers in the RetinaNet backbone
    learning_rate: the base learning rate for the training process
    batch_size: the batch size
    training_steps: the number of training steps
    warmup_steps: the number of warmup steps
    try_restore: indicates if the latest checkpoint should be restored
    half_precision: indicates if half-precision floating points should be used
    checkpoint_period: the frequency in steps for checkpointing the model

  Returns:
    A `State` object containing the trained model. 
  """
  # Set the correct dtype based on the platform being used
  dtype = jnp.float32
  if half_precision:
    if jax.local_devices()[0].platform == 'tpu':
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float16

  # Create the training entities, and replicate the state
  rng, rng_input = jax.random.split(rng)
  model, model_state = create_model(
      rng_input, shape=shape, classes=classes, depth=depth, dtype=dtype)
  optimizer = create_optimizer(model, beta=0.9, weight_decay=0.0001)
  meta_state = State(optimizer=optimizer, model_state=model_state)
  del model, model_state, optimizer  # Remove duplicate data

  # Try to restore the state of a previous run
  meta_state = restore_checkpoint(meta_state) if try_restore else meta_state
  start_step = meta_state.step

  # Prepare the LR scheduler
  learning_rate *= batch_size / 256
  learning_rate_fn = create_scheduled_decay_fn(learning_rate, training_steps,
                                               warmup_steps)

  # Create the step function
  step_fn = create_step_fn(learning_rate_fn)

  # Run the training loop
  train_iter = iter(train_data)
  print(f"Starting training loop at step {start_step}.")
  for step in range(start_step, warmup_steps + training_steps):
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
    print(f"(Training Step #{step}) Getting input batch.")
    meta_state, loss = step_fn(batch, meta_state)
    print(f"(Train Step #{step}) RetinaNet Loss: {loss}")

    # if step % 10 == 0 and step != 0:
    #   checkpoints.save_checkpoint(
    #       "singl_thread_checkpoints", meta_state, meta_state.step, keep=10)

    continue  # For now, skip evaluation

    # Evaluate and checkpoint the model
    if step % checkpoint_period == 0 and step != 0:
      epoch = step // checkpoint_period
      # checkpoint_state(meta_state, epoch)

      eval_results = []
      test_iter = iter(test_data)
      for _ in range(100):
        batch = jax.tree_map(lambda x: x._numpy(), next(test_iter))  # pylint: disable=protected-access
        results = eval(batch, meta_state)
        eval_results.append(results)
      eval_results = aggregate_evals(eval_results)
      print(f"(Epoch #{epoch}) Evaluation results: {eval_results}\n")

  return meta_state
