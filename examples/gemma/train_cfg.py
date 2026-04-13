# Copyright 2026 The Flax Authors.
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

import dataclasses
import math
from typing import Any


@dataclasses.dataclass(slots=True)
class TrainConfig:
  """Configuration for training a gemma model."""

  # Path to load or store sentencepiece vocab file.
  vocab_path: str | None
  # Vocabulary size if `vocab_path` is not given.
  vocab_size: int
  # Maximum number of characters to use for training.
  max_corpus_chars: int
  # Name of TFDS translation dataset to use.
  dataset_name: str
  # Optional name of TFDS translation dataset to use for evaluation.
  eval_dataset_name: str
  # Optional name of TFDS split to use for evaluation.
  eval_split: str
  # Per device batch size for training.
  per_device_batch_size: int
  # Per device batch size for training.
  eval_per_device_batch_size: int
  # Grain prefetch number of workers.
  prefetch_num_workers: int | None

  # Prompt for language model sampling
  prompts: tuple[str, ...]
  # Temperature for top_p sampling.
  sampling_temperature: float
  # Top-p sampling threshold.
  sampling_top_p: float

  # Number of steps to take during training.
  num_train_steps: int
  # Number of steps to take during evaluation.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  num_eval_steps: int
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int
  # Base learning rate.
  learning_rate: float
  # Linear learning rate warmup.
  warmup_steps: int
  # Cross entropy loss label smoothing.
  label_smoothing: float
  # Decay factor for AdamW style weight decay.
  weight_decay: float
  # Maximum length cutoff for training examples.
  max_target_length: int
  # Maximum length cutoff for eval examples.
  max_eval_target_length: int

  # Gemma transformer name.
  # Possible values defined in transformer.TransformerConfig:
  # (gemma_2b, gemma_7b, gemma2_2b, gemma2_9b, gemma2_27b, gemma3_1b, gemma3_4b,
  # ...)
  transformer_name: str | None
  # or alternatively define the model using the dict of parameters
  transformer_params: dict[Any, Any] | None

  # Whether to save model checkpoints.
  save_checkpoints: bool
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool
  # Integer for PRNG random seed.
  seed: int

  # Parallelism
  mesh_axes: tuple[str, ...]
  data_sharding: tuple[str | tuple[str], ...]

  fsdp_parallelism: int = -1
  tensor_parallelism: int = 1

  # Profiling
  with_profiler_step_trace: bool = False

  # Dataflow choice: grain or TF tensor
  input_pipeline_type: str = "grain"  # ["grain", "tf"]
  # Flax specif configs, mostly for benchmarks
  use_nnx_tree_mode: bool = False
  use_nnx_transforms: str = "no"  # ["all", "no", "grad-only", "jit-only"]

  sow_config: tuple[str, ...] | None = None

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def __post_init__(self):
    axis_shapes = [self.fsdp_parallelism, self.tensor_parallelism]
    assert axis_shapes.count(-1) in (0, 1), (
        f"Found unspecified values (-1) for more than one parallelism axis. "
        f"At most one axis can be unspecified."
    )

  def get_mesh_shape(self, num_devices: int) -> tuple[int, int]:
    axis_shapes = [self.fsdp_parallelism, self.tensor_parallelism]
    count = math.prod(axis_shapes)
    if count < 0:
      axis_shapes[axis_shapes.index(-1)] = int(num_devices / (-count))
    else:
      assert count == num_devices
    return tuple(axis_shapes)
