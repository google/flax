# Copyright 2024 The Flax Authors.
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

"""Input pipeline for a LM1B dataset."""

from train_cfg import TrainConfig
import jax


def get_datasets(
    config: TrainConfig,
    *,
    vocab_path: str | None = None,
    data_sharding: jax.sharding.Sharding | None = None,
):
  if config.input_pipeline_type == "grain":
    from input_pipeline_grain import get_datasets

    return get_datasets(
        config, vocab_path=vocab_path, data_sharding=data_sharding
    )
  elif config.input_pipeline_type == "tf":
    from input_pipeline_tf import get_datasets

    return get_datasets(
        config, vocab_path=vocab_path, data_sharding=data_sharding
    )
  else:
    raise ValueError(f"Unknown {config.input_pipeline_type=}")
