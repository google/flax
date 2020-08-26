import math
import time

from absl import logging
from clu import hooks
from coco_eval import CocoEvaluator
import ml_collections
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints, common_utils
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from typing import Any, Iterable, Mapping, Tuple

import input_pipeline
from model import create_retinanet

_EPSILON = 1e-9


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
                 shape: Iterable[int] = (224, 224, 3)) -> flax.nn.Model:
  """Creates a RetinaNet model.

  Args:
    rng: the Jax PRNG, which is used to instantiate the model weights
    depth: the depth of the basckbone network
    classes: the number of classes in the object detection task
    shape: the shape of the image inputs, with the format (N, H, W, C)

  Returns:
    The RetinaNet instance, and its state object
  """
  # The number of classes is increased by 1 since we add the background
  partial_module = create_retinanet(depth, classes=classes + 1)

  # Since the BatchNorm has state, we'll need to use stateful here
  with flax.nn.stateful() as init_state:
    _, params = partial_module.init_by_shape(
        rng, input_specs=[(shape, jnp.float32)])

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
  c = jnp.minimum(anchor_type + 1.0, 1.0)
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
  c = jnp.maximum(anchor_type, 0.0)
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


def coco_eval_step(bboxes, scores, img_ids, scales, evaluator):
  """Adds a set of batch inferences to the COCO evaluation object

  Args:
    pred: the output of the model in inference mode
    img_ids: an array of length `N`, containing the id of each of image
    scales: an array of length `N`, containing the scale of each image
    evaluator: a CocoEvaluator object, used to process the inferences
  """
  # Reshape the data such that it can be processed
  bboxes_shape = bboxes.shape
  bboxes = bboxes.reshape(-1, bboxes_shape[-2], bboxes_shape[-1])

  scores_shape = scores.shape
  scores = scores.reshape(-1, scores_shape[-2], scores_shape[-1])

  img_ids = img_ids.reshape(-1)
  scales = scales.reshape(-1)

  # Convert to numpy to avoid incompatibility issues
  bboxes = np.array(bboxes)
  scores = np.array(scores)

  # Add the annotations and return the evaluator
  evaluator.add_annotations(bboxes, scores, img_ids, scales)


def sync_results(coco_evaluator):
  """Synchronize the CocoEvaluator across hosts, and produce the COCO metrics.

  Args:
    coco_evaluator: the local CocoEvaluator object

  Returns:
    The results synchronized across hosts
  """
  # Get the local annotations, and clear the evaluator
  annotations, ids = coco_evaluator.get_annotations_and_ids()
  coco_evaluator.clear_annotations()

  def _inner(x):
    i_annotations = jax.lax.all_gather(annotations, 'batch')
    i_ids = jax.lax.all_gather(ids, 'batch')

    return i_annotations, i_ids
  inner = jax.pmap(_inner, 'batch')


  # Compute the results this is host 0
  inner()
  if jax.host_id() == 0:
    coco_evaluator.set_annotations_and_ids(annotations, ids) 

  return jax.tree_util.build_tree(tree_def, results[0])


def infer(data, meta_state):
  """Infers on data.

  Args:
    data: the data for inference
    model: an instance of the State class

  Returns:
    The inference on the data, i.e. the tuple consisting of: classifications, 
    regressions, bboxes 
  """
  with flax.nn.stateful(meta_state.model_state, mutable=False):
    pred = meta_state.optimizer.target(
        data['image'], img_shape=data['size'], train=False)
  return pred


def checkpoint_state(meta_state: State, checkpoint_dir: str = "checkpoints"):
  """
  Checkpoints the training state.

  Args:
    meta_state: a `State` object, which contains the state of
      the current training step
    checkpoint_step: a checkpoint step, used for versioning the checkpoint
    checkpoint_dir: the directory where the checkpoint is stored
  """
  if jax.host_id() == 0:
    meta_state = jax.device_get(jax.tree_map(lambda x: x[0], meta_state))
    checkpoints.save_checkpoint(
        checkpoint_dir, meta_state, meta_state.step, keep=3)


def restore_checkpoint(meta_state: State,
                       checkpoint_dir: str = "checkpoints") -> State:
  """Restores the latest checkpoint.

  More specifically, either return the latest checkpoint from the
  `checkpoint_dir` or returns the `meta_state` object, if no such checkpoint
  exists.

  Args:
    meta_state: a `State` object, used as last resort if no checkpoint
      does exist
    checkpoint_dir: the directory where the checkpoints are searched for

  Returns:
    Either the latest checkpoint, if it exists, or the `meta_state` object
  """
  return checkpoints.restore_checkpoint(checkpoint_dir, meta_state)


def sync_model_state(meta_state: State) -> State:
  """Synchronizes the model_state across devices.

  Args:
    meta_state: a `State` object to be used towards synchronization

  Returns:
    A new State object with an updated `model_state` field.
  """
  sync_state = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
  return meta_state.replace(model_state=sync_state(meta_state.model_state))


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

    # Compute the gradients and the loss, then average them across devices
    aux, grads = jax.value_and_grad(
        _loss_fn, has_aux=True)(meta_state.optimizer.target,
                                meta_state.model_state)
    loss, new_model_state = aux

    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')

    # Apply the gradients to the model
    updated_optimizer = meta_state.optimizer.apply_gradient(
        grads, learning_rate=lr_function(step))

    # Update the meta_state
    meta_state = meta_state.replace(
        step=step, model_state=new_model_state, optimizer=updated_optimizer)

    return meta_state, {"retinanet_loss": loss}

  return take_step


def eval_to_tensorboard(writer, evals, step, train=True, aggregate=True):
  if aggregate:
    evals = common_utils.get_metrics(evals)
    summary = jax.tree_map(lambda x: x.mean(), evals)
    logging.info("(Training Step #%d) Aggregated Metrics: %s", step, summary)

  for key, vals in evals.items():
    tag = f'{"train" if train else "eval"}_{key}'

    if not isinstance(vals, (list, np.ndarray)):
      vals = [vals]

    for i, val in enumerate(vals):
      writer.scalar(tag, val, step - len(vals) + i + 1)
  writer.flush()


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> State:
  """Runs a training and evaluation loop.

  Args:
    config: a `ConfigDict` object, which holds all the information necessary
      for configuring the training process
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint, training will be resumed from the latest checkpoint.
  """
  tf.io.gfile.makedirs(workdir)
  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)

  # Deterministic training, see go/deterministic training.
  rng = jax.random.PRNGKey(config.seed)

  # Set up the data pipeline
  rng, data_rng = jax.random.split(rng)
  ds_info, train_data, val_data = input_pipeline.create_datasets(
      config, data_rng)
  num_classes = ds_info.features["objects"]["label"].num_classes

  logging.info("Training data shapes: %s", train_data.element_spec)
  input_shape = list(train_data.element_spec["image"].shape)[1:]

  # Create the training entities, and replicate the state
  rng, model_rng = jax.random.split(rng)

  model, model_state = create_model(
      model_rng,
      shape=input_shape,
      classes=num_classes,
      depth=config.depth)
  optimizer = create_optimizer(model, beta=0.9, weight_decay=0.0001)
  meta_state = State(optimizer=optimizer, model_state=model_state)
  del model, model_state, optimizer

  # Try to restore the state of a previous run
  if config.try_restore:
    meta_state = restore_checkpoint(meta_state)
  start_step = meta_state.step

  # Replicate the state across devices
  meta_state = flax.jax_utils.replicate(meta_state)

  # Prepare the LR scheduler
  learning_rate = config.learning_rate * config.per_device_batch_size / 256
  learning_rate_fn = create_scheduled_decay_fn(learning_rate,
                                               config.num_train_steps,
                                               config.warmup_steps)

  # Prepare the training loop for distributed runs
  step_fn = create_step_fn(learning_rate_fn)
  p_step_fn = jax.pmap(step_fn, axis_name="batch")
  p_infer_fn = jax.pmap(infer, axis_name="batch")

  # Note that the CocoEvaluator is a singleton object
  coco_evaluator = CocoEvaluator(config.eval_annotations_path,
                                 config.eval_remove_background,
                                 config.eval_threshold)

  # Run the training loop
  running_metrics = []
  train_iter = iter(train_data)
  report_progress = hooks.ReportProgress(num_train_steps=config.num_train_steps)
  logging.info("Starting training loop at step %d.", start_step)
  for step in range(start_step, config.num_train_steps + config.warmup_steps):
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
    meta_state, metrics = p_step_fn(batch, meta_state)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, "Finished training step %d!", 10, step)
    report_progress(step, time.time())

    # Log the loss for this batch
    running_metrics.append(metrics)

    # Periodically sync the model state
    if step % config.sync_steps == 0 and step != 0:
      meta_state = sync_model_state(meta_state)

      # Submit the metrics to tensorboard
      if jax.host_id() == 0:
        eval_to_tensorboard(summary_writer, running_metrics, step)
        running_metrics.clear()

    # Sync the model state, evaluate and checkpoint the model
    if step % config.checkpoint_period == 0 and step != 0:
      # Checkpoint the model
      meta_state = sync_model_state(meta_state)
      checkpoint_state(meta_state)

      # Run evaluation on the model
      coco_evaluator.clear_annotations()  # Clear former annotations
      val_iter = iter(val_data)  # Refresh the eval iterator
      for _ in range(250):
        batch = jax.tree_map(lambda x: x._numpy(), next(val_iter))  # pylint: disable=protected-access
        scores, regressions, bboxes = p_infer_fn(batch, meta_state)
        coco_eval_step(bboxes, scores, batch["id"], batch["scale"],
                       coco_evaluator)

      # Compute the COCO metrics
      eval_results = coco_evaluator.compute_coco_metrics()

      # Log the reports via standard logging
      checkpoint = step // config.checkpoint_period
      logging.info("(Checkpoint #%d) COCO Metrics: %s.", checkpoint,
                   eval_results)

      # Write the evaluation results to tensorboard
      eval_to_tensorboard(
          summary_writer,
          eval_results,
          checkpoint,
          train=False,
          aggregate=False)

  return meta_state
