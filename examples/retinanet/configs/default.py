from jax.numpy import float32


class ConfigDict(dict):
  """Dictionary with dot access to values."""

  def __getattr__(self, attr):
    return self[attr]

  def __setattr__(self, key, value):
    self[key] = value

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super().__delitem__(key)
    del self.__dict__[key]


def get_config():
  """Get the default hyperparameter configuration."""
  config = ConfigDict()

  config.learning_rate = 0.01
  config.per_device_batch_size = 2
  config.num_train_steps = 90_000
  config.warmup_steps = 30_000
  config.half_precision = False
  config.try_restore = False
  config.distributed_training = True
  config.dtype = float32

  # The number of layers in the RetinaNet backbone.
  config.depth = 50

  config.sync_steps = 100
  config.checkpoint_period = 1_500
  config.seed = 42

  # Evaluation parameters
  config.eval_annotations_path = "/home/dgraur/data/files/coco_annotations/instances_val2014.json"
  config.eval_remove_background = True
  config.eval_threshold = 0.05

  return config
