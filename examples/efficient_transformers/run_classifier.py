# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Run sequence-level classification (and regression) fine-tuning."""

import datetime
import functools
import os
import typing

from absl import app
from absl import flags
from absl import logging
import dataclasses
from flax import nn
from flax import optim
import data
import import_weights
import modeling
import training
# from flax.metrics import tensorboard
# from flax.training import checkpoints
import gin
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile
import tensorflow_datasets as tfds


from sentencepiece import SentencePieceProcessor


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to the config files.')

flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


@dataclasses.dataclass
class RunClassifierConfig:
  """Configuration for fine-tuning on GLUE."""
  # The name of the task to train.
  dataset_name: str
  # The output directory where the model checkpoints will be written.
  # Whether to run training.
  do_train: str = True
  # Whether to run eval on the dev set.
  do_eval: str = True
  # Whether to run the model in inference mode on the test set.
  do_predict: str = True
  # Total batch size for training.
  train_batch_size: int = 16
  # Total batch size for eval.
  eval_batch_size: int = 8
  # Total batch size for predict.
  predict_batch_size: int = 8
  # The base learning rate for Adam.
  learning_rate: float = 1e-5
  # Total number of training epochs to perform.
  num_train_epochs: float = 3.0
  # Proportion of training to perform linear learning rate warmup for.
  # E.g., 0.1 = 10% of training.
  warmup_proportion: float = 0.1
  # Configuration for the pre-trained checkpoint.
  bert_config: typing.Any = None
  # The maximum total input sequence length after WordPiece tokenization.
  # Sequences longer than this will be truncated, and sequences shorter
  # than this will be padded.
  max_seq_length: int = 128


RunClassifierConfig.configurable = gin.external_configurable(
    RunClassifierConfig, name='RunClassifier')
gin.external_configurable(modeling.bert_base_uncased, name='bert_base_uncased')


def get_config():
  if not gin.config_is_locked():
    gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  return RunClassifierConfig.configurable()


def get_output_dir(config):
  """Get output directory location."""
  output_dir = FLAGS.output_dir
  if output_dir is None:
    dataset_name = config.dataset_name.replace('/', '_')
    output_name = '{dataset_name}_{timestamp}'.format(
        dataset_name=dataset_name,
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
    )
    output_dir = os.path.join('~', 'efficient_transformers', output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    print('No --output_dir specified')
    print('Using default output_dir:', output_dir, flush=True)
  return output_dir


def create_model(config, num_classes=2):
  """Create a model, starting with a pre-trained checkpoint."""
  model_kwargs = dict(
      config=config.bert_config,
      n_classes=num_classes,
  )
  model_def = modeling.BertForSequenceClassification.partial(**model_kwargs)
  if config.bert_config.init_checkpoint:
    initial_params = import_weights.load_params_from_tf(
        init_checkpoint=config.bert_config.init_checkpoint,
        d_model=config.bert_config.d_model,
        num_heads=config.bert_config.num_heads,
        num_classes=num_classes)
  else:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(
          jax.random.PRNGKey(0),
          [((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, 1), jnp.int32)],
          deterministic=True)
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(config, model):
  optimizer_def = optim.Adam(
      learning_rate=config.learning_rate,
      beta1=0.9,
      beta2=0.999,
      eps=1e-6,
      weight_decay=0.0)
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['type_ids'],
        batch['label'])
  return metrics['loss'], metrics


def compute_classification_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['type_ids'],
        deterministic=True)
  return {
      'idx': batch['idx'],
      'label': batch['label'],
      'prediction': y.argmax(-1)
  }


def compute_regression_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['type_ids'],
        deterministic=True)
  return {
      'idx': batch['idx'],
      'label': batch['label'],
      'prediction': y[..., 0],
  }


def create_eval_metrics_fn(dataset_name, is_regression_task):
  """Create a function that computes task-relevant metrics."""
  def get_accuracy(guess, gold):
    return (guess == gold).mean()

  def get_mcc(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    tn = ((guess == 0) & (gold == 0)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (mcc_denom + 1e-6)
    return mcc

  def get_f1(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    return f1

  def get_f1_accuracy_mean(guess, gold):
    return (get_f1(guess, gold) + get_accuracy(guess, gold)) / 2.0

  def get_pearsonr(x, y):
    return np.corrcoef(x, y)[0, 1]

  eval_metrics = {}
  if is_regression_task:
    eval_metrics['pearsonr'] = get_pearsonr
  else:
    eval_metrics['accuracy'] = get_accuracy

  if dataset_name == 'glue/cola':
    eval_metrics['mcc'] = get_mcc
  elif dataset_name in ('glue/mrpc', 'glue/qqp'):
    eval_metrics['f1_accuracy_mean'] = get_f1_accuracy_mean

  def metrics_fn(stats):
    res = {}
    for name, fn in eval_metrics.items():
      res[name] = fn(stats['label'], stats['prediction'])
    return res

  return metrics_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = get_config()

  ds_info = tfds.builder(config.dataset_name).info
  num_train_examples = ds_info.splits[tfds.Split.TRAIN].num_examples
  num_train_steps = int(
      num_train_examples * config.num_train_epochs // config.train_batch_size)
  warmup_steps = int(config.warmup_proportion * num_train_steps)
  cooldown_steps = num_train_steps - warmup_steps

  is_regression_task = (ds_info.features.dtype['label'] == np.float32)
  if is_regression_task:
    num_classes = 1
    compute_stats = compute_regression_stats
  else:
    num_classes = ds_info.features['label'].num_classes
    compute_stats = compute_classification_stats

  model = create_model(config, num_classes=num_classes)
  optimizer = create_optimizer(config, model)
  optimizer = optimizer.replicate()
  del model  # don't keep a copy of the initial model

  tokenizer = SentencePieceProcessor()
  tokenizer.Load(config.bert_config.vocab_file)
  tokenizer.SetEncodeExtraOptions('bos:eos')  # Auto-add [CLS] and [SEP] tokens
  glue_inputs = functools.partial(
      data.glue_inputs, dataset_name=config.dataset_name,
      max_len=config.max_seq_length, tokenizer=tokenizer)

  learning_rate_fn = training.create_learning_rate_scheduler(
      factors='constant * linear_warmup * cosine_decay',
      base_learning_rate=config.learning_rate,
      warmup_steps=warmup_steps,
      steps_per_cycle=cooldown_steps,
  )

  output_dir = get_output_dir(config)
  gfile.makedirs(output_dir)

  train_history = training.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_step_fn = training.create_train_step(compute_loss_and_metrics)
    train_iter = glue_inputs(
        split=tfds.Split.TRAIN, batch_size=config.train_batch_size,
        training=True)

    for step, batch in zip(range(0, num_train_steps), train_iter):
      optimizer, train_state = train_step_fn(optimizer, batch, train_state)

  if config.do_eval:
    eval_step = training.create_eval_fn(compute_stats)
    eval_metrics_fn = create_eval_metrics_fn(
        config.dataset_name, is_regression_task)
    eval_results = []

    if config.dataset_name == 'glue/mnli':
      validation_splits = ['validation_matched', 'validation_mismatched']
    else:
      validation_splits = [tfds.Split.VALIDATION]

    for split in validation_splits:
      eval_iter = glue_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      eval_stats = eval_step(optimizer, eval_iter)
      eval_metrics = eval_metrics_fn(eval_stats)
      prefix = 'eval_mismatched' if split == 'validation_mismatched' else 'eval'
      for name, val in sorted(eval_metrics.items()):
        line = f'{prefix}_{name} = {val:.06f}'
        print(line, flush=True)
        logging.info(line)
        eval_results.append(line)

    eval_results_path = os.path.join(output_dir, 'eval_results.txt')
    with gfile.GFile(eval_results_path, 'w') as f:
      for line in eval_results:
        f.write(line + '\n')

  if config.do_predict:
    predict_step = training.create_eval_fn(compute_stats)
    predict_results = []

    path_map = {
        ('glue/cola', tfds.Split.TEST): 'CoLA.tsv',
        ('glue/mrpc', tfds.Split.TEST): 'MRPC.tsv',
        ('glue/qqp', tfds.Split.TEST): 'QQP.tsv',
        ('glue/sst2', tfds.Split.TEST): 'SST-2.tsv',
        ('glue/stsb', tfds.Split.TEST): 'STS-B.tsv',
        ('glue/mnli', 'test_matched'): 'MNLI-m.tsv',
        ('glue/mnli', 'test_mismatched'): 'MNLI-mm.tsv',
        ('glue/qnli', tfds.Split.TEST): 'QNLI.tsv',
        ('glue/rte', tfds.Split.TEST): 'RTE.tsv',
        # No eval on WNLI for now. BERT accuracy on WNLI is below baseline,
        # unless a special training recipe is used.
        # ('glue/wnli', tfds.Split.TEST): 'WNLI.tsv',
    }
    label_sets = {
        'glue/cola': ['0', '1'],
        'glue/mrpc': ['0', '1'],
        'glue/qqp': ['0', '1'],
        'glue/sst2': ['0', '1'],
        'glue/mnli': ['entailment', 'neutral', 'contradiction'],
        'glue/qnli': ['entailment', 'not_entailment'],
        'glue/rte': ['entailment', 'not_entailment'],
    }

    for path_map_key in path_map:
      candidate_dataset_name, split = path_map_key
      if candidate_dataset_name != config.dataset_name:
        continue

      predict_iter = glue_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      predict_stats = predict_step(optimizer, predict_iter)
      idxs = predict_stats['idx']
      predictions = predict_stats['prediction']

      tsv_path = os.path.join(output_dir, path_map[config.dataset_name, split])
      with gfile.GFile(tsv_path, 'w') as f:
        f.write('index\tprediction\n')
        if config.dataset_name == 'glue/stsb':
          for idx, val in zip(idxs, predictions):
            f.write(f'{idx}\t{val:.06f}\n')
        else:
          label_set = label_sets[config.dataset_name]
          for idx, val in zip(idxs, predictions):
            f.write(f'{idx}\t{label_set[val]}\n')
      logging.info('Wrote %s', tsv_path)


if __name__ == '__main__':
  app.run(main)
