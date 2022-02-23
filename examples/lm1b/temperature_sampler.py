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

"""Fast decoding routines for inference from a trained language model."""

from jax import lax
from jax import random
import jax.numpy as jnp


# Constants
# The default End-of-Sentence token id is 2 (SentencePiece).
EOS_ID = 2


def temperature_sample(prompt_inputs,
                       init_cache,
                       tokens_to_logits,
                       prng_key,
                       temperature=1.0,
                       topk=20,
                       eos_token=EOS_ID):
  """Temperature sampling for language model generation.

  Args:
    prompt_inputs: array: [batch_size, max_decode_len] int32 sequence of tokens.
    init_cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    prng_key: JAX PRNGKey.
    temperature: float: sampling temperature factor. As it approaches
      zero this becomes equivalent to greedy sampling.
    topk: integer: if nonzero only use the top-k logits to sample next token,
      if zero don't use any cutoff and sample from full logits over vocabulary.
    eos_token: int: end-of-sentence token for target vocabulary.

  Returns:
    Array of sampled sequences: [batch_size, max_decode_len]
  """
  batch_size = prompt_inputs.shape[0]
  max_decode_len = prompt_inputs.shape[1]
  end_marker = jnp.array(eos_token)
  temperature = jnp.array(temperature)

  # Initialize sampling loop state.
  # initial loop PRNGKey
  rng0 = prng_key
  # loop position counter.
  i0 = jnp.array(0)
  # per batch-item holding current token in loop.
  token0 = jnp.zeros((batch_size, 1), dtype=jnp.int32)
  # per batch-item state bit indicating if sentence has finished.
  ended0 = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
  # (batch, length) array containing prefix prompt tokens for sampling loop
  # as well as the generated output of newly sampled tokens.
  sequences0 = prompt_inputs
  # Sampling loop state is stored in a simple tuple.
  sampling_loop_init_state = (i0, sequences0, init_cache, token0, ended0, rng0)

  def sampling_loop_cond_fn(state):
    """Sampling loop termination condition."""
    (i, _, _, _, ended, _) = state
    # Have we reached max decoding length?
    not_at_end = (i < max_decode_len)
    # Have all sampled sequences reached an end marker?
    all_sequences_ended = jnp.all(ended)
    return not_at_end & (~all_sequences_ended)

  def sampling_loop_body_fn(state):
    """Sampling loop state update."""
    i, sequences, cache, cur_token, ended, rng = state
    # Split RNG for sampling.
    rng1, rng2 = random.split(rng)
    # Call fast-decoder model on current tokens to get next-position logits.
    logits, new_cache = tokens_to_logits(cur_token, cache)
    # Sample next token from logits.
    # TODO(levskaya): add top-p "nucleus" sampling option.
    if topk:
      # Get top-k logits and their indices, sample within these top-k tokens.
      topk_logits, topk_idxs = lax.top_k(logits, topk)
      topk_token = jnp.expand_dims(random.categorical(
          rng1, topk_logits / temperature).astype(jnp.int32), axis=-1)
      # Return the original indices corresponding to the sampled top-k tokens.
      next_token = jnp.squeeze(
          jnp.take_along_axis(topk_idxs, topk_token, axis=-1), axis=-1)
    else:
      next_token = random.categorical(
          rng1, logits / temperature).astype(jnp.int32)
    # Only use sampled tokens if we're past provided prefix tokens.
    out_of_prompt = (sequences[:, i+1] == 0)
    next_token = (next_token * out_of_prompt +
                  sequences[:, i+1] * ~out_of_prompt)
    # If end-marker reached for batch item, only emit padding tokens.
    next_token_or_endpad = (next_token[None] * ~ended)
    ended |= (next_token_or_endpad == end_marker)
    # Add current sampled tokens to recorded sequences.
    new_sequences = lax.dynamic_update_slice(
        sequences, next_token_or_endpad, (0, i+1))
    return (i+1, new_sequences, new_cache, next_token_or_endpad, ended, rng2)

  # Run sampling loop and collect final state.
  final_state = lax.while_loop(sampling_loop_cond_fn,
                               sampling_loop_body_fn,
                               sampling_loop_init_state)

  # Pick part of the state corresponding to the sampled sequences.
  final_sequences = final_state[1]
  return final_sequences
