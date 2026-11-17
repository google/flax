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

#!/usr/bin/env python3
"""Benchmark script for comparing original LayerNorm vs PallasLayerNorm.

Usage:
    python3 pallas_benchmark.py
"""
import time

from absl import app
import chex
from flax import linen as nn
import models
import jax
import jax.numpy as jnp
import numpy as np


# Test input shapes matching typical node/edge tensor dimensions in ogbg_molpcba
TEST_SHAPES = [
    (2048, 512),
    (4096, 512),
]

DTYPE = jnp.float32
WARMUP = 5
ITERATIONS = 100
RTOL = 1e-3   # Relative tolerance
ATOL = 1e-3   # Absolute tolerance


def benchmark(fn, *args, warmup=WARMUP, iterations=ITERATIONS, label=""):
  """Benchmark with proper warmup and synchronization."""
  del label
  for _ in range(warmup):
    result = fn(*args)
    jax.block_until_ready(result)

  times = []
  for _ in range(iterations):
    t0 = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    times.append(time.perf_counter() - t0)

  stats = {
      "mean_ms": np.mean(times) * 1000,
      "std_ms": np.std(times) * 1000,
      "median_ms": np.median(times) * 1000,
      "min_ms": np.min(times) * 1000,
  }
  return stats


def check_correctness(x, key):
  """Verify numerical equivalence between nn.LayerNorm and PallasLayerNorm."""
  layer_norm = nn.LayerNorm()
  pallas_layer_norm = models.PallasLayerNorm()

  y_orig, params_orig = layer_norm.init_with_output(key, x)
  y_opt, params_opt = pallas_layer_norm.init_with_output(key, x)

  max_diff = float(jnp.max(jnp.abs(y_orig - y_opt)))
  mean_diff = float(jnp.mean(jnp.abs(y_orig - y_opt)))
  matches = bool(jnp.allclose(y_orig, y_opt, rtol=RTOL, atol=ATOL))

  try:
    chex.assert_trees_all_close(y_orig, y_opt, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    print(f"Chex forward assertion failed: {e}")
    matches = False

  # Verify VJP backward pass correctness
  try:
    fn_orig = lambda x_val: layer_norm.apply(params_orig, x_val)
    fn_opt = lambda x_val: pallas_layer_norm.apply(params_opt, x_val)
    _, vjp_orig = jax.vjp(fn_orig, x)
    _, vjp_opt = jax.vjp(fn_opt, x)
    g = jnp.ones_like(x)
    dx_orig = vjp_orig(g)[0]
    dx_opt = vjp_opt(g)[0]
    chex.assert_trees_all_close(dx_orig, dx_opt, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    print(f"Chex backward VJP assertion failed: {e}")
    matches = False

  return matches, max_diff, mean_diff, params_orig, params_opt


def main(argv):
  if len(argv) > 1:
    pass
  print("=" * 70)
  print("PALLAS LAYERNORM BENCHMARK")
  print(f"Backend: {jax.default_backend()}")
  print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
  print(f"Dtype: {DTYPE}")
  print("=" * 70)

  key = jax.random.PRNGKey(0)

  for shape in TEST_SHAPES:
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape, dtype=DTYPE)
    print(f"\n--- Shape: {shape} ---")

    # Correctness check
    matches, max_diff, mean_diff, params_orig, params_opt = check_correctness(
        x, key
    )
    status = "✅ PASS" if matches else "❌ FAIL"
    print(
        f"Correctness: {status} "
        f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
    )

    if not matches:
      print("  ⚠️  Skipping benchmark due to correctness failure")
      continue

    # Performance benchmark
    layer_norm = nn.LayerNorm()
    pallas_layer_norm = models.PallasLayerNorm()

    orig_fn = jax.jit(layer_norm.apply)
    opt_fn = jax.jit(pallas_layer_norm.apply)

    orig_stats = benchmark(orig_fn, params_orig, x, label="Original")
    opt_stats = benchmark(opt_fn, params_opt, x, label="Optimized")

    speedup = orig_stats["mean_ms"] / opt_stats["mean_ms"]

    orig_mean = orig_stats["mean_ms"]
    orig_std = orig_stats["std_ms"]
    opt_mean = opt_stats["mean_ms"]
    opt_std = opt_stats["std_ms"]

    print(f"Original:  {orig_mean:8.3f} ms ± {orig_std:.3f}")
    print(f"Optimized: {opt_mean:8.3f} ms ± {opt_std:.3f}")
    print(f"Speedup:   {speedup:.2f}x", end="")
    if speedup > 1.05:
      pct = (speedup - 1) * 100
      print(f" ({pct:.1f}% faster)")
    elif speedup < 0.95:
      pct = (1 - speedup) * 100
      print(f" ({pct:.1f}% SLOWER ⚠️)")
    else:
      print(" (no significant change)")

  print("\n" + "=" * 70)
  print("BENCHMARK COMPLETE")
  print("=" * 70)


if __name__ == "__main__":
  app.run(main)
