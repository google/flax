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

"""Main file for running the VAE example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import argparse
import logging
import jax
import tensorflow as tf
import time
import train
from config import TrainingConfig, get_default_config
import os

def setup_training_args():
    """Setup training arguments with defaults from config."""
    parser = argparse.ArgumentParser(description='VAE Training Script')
    config = get_default_config()

    # Add all config parameters as arguments
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate,
                       help='Learning rate for training')
    parser.add_argument('--latents', type=int, default=config.latents,
                       help='Number of latent dimensions')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs,
                       help='Number of training epochs')
    parser.add_argument('--workdir', type=str, default='/tmp/vae',
                       help='Working directory for checkpoints and logs')

    args = parser.parse_args()

    # Convert args to TrainingConfig
    return TrainingConfig(
        learning_rate=args.learning_rate,
        latents=args.latents,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    ), args.workdir

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Parse arguments and get config
    config, workdir = setup_training_args()

    # Create workdir if it doesn't exist
    os.makedirs(workdir, exist_ok=True)

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Simple process logging
    logging.info('Starting training process %d/%d',
                jax.process_index(), jax.process_count())

    start = time.perf_counter()
    train.train_and_evaluate(config)
    logging.info('Total training time: %.2f seconds', time.perf_counter() - start)

if __name__ == '__main__':
    main()
