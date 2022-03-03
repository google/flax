# Copyright 2022 The Flax Authors.
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

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Path to load or store sentencepiece vocab file.
  config.vocab_path = None

  # Vocabulary size if `vocab_path` is not given.
  config.vocab_size = 30_000

  config.max_corpus_chars = 10**7

  # Name of TFDS translation dataset to use.
  config.dataset_name = "lm1b"

  # Optional name of TFDS translation dataset to use for evaluation.
  config.eval_dataset_name = "lm1b"
  config.eval_split = "test"

  # Per device batch size for training.
  config.per_device_batch_size = 32

  # Per device batch size for training.
  config.eval_per_device_batch_size = 32

  # Sampling temperature for language model inference.
  config.sampling_temperature = 0.6

  # Top k cutoff for logit sampling. If 0 then no top-k cutoff is used.
  config.sampling_top_k = 20

  config.num_train_steps = 500_000

  # Number of steps to take during evaluation. Large enough to evaluate all.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  config.num_eval_steps = 2_000
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  config.num_predict_steps = -1

  # Base learning rate.
  config.learning_rate = 0.0016

  # Linear learning rate warmup.
  config.warmup_steps = 1000

  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.0

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.1

  # Maximum length cutoff for training examples.
  config.max_target_length = 128
  # Maximum length cutoff for eval examples.
  config.max_eval_target_length = 512
  # Maximum length cutoff for predicted tokens.
  config.max_predict_length = 50

  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = False

  # Number of transformer layers.
  config.num_layers = 6

  # Size of query/key/value for attention.
  config.qkv_dim = 512
  # Size of embeddings.
  config.emb_dim = 512
  # Size of the MLP.
  config.mlp_dim = 2048

  # Number of attention heads.
  config.num_heads = 8

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  # Save a checkpoint every these number of steps.
  config.checkpoint_every_steps = 10_000
  # Frequency of eval during training, e.g. every 1_000 steps.
  config.eval_every_steps = 1_000

  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = True

  # Integer for PRNG random seed.
  config.seed = 0

  # Prompt for language model sampling.
  config.prompts = "I love to "

  return config
