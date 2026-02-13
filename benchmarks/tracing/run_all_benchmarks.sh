#!/bin/bash
set -e

export XLA_FLAGS=--xla_force_host_platform_device_count=8

TARGETS=(
  mnist
  vae
  sst2
  gemma
  imagenet
  seq2seq
  lm1b
  nlp_seq
  ogbg_molpcba
  wmt
  ppo
)

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "Running benchmark: ${target}"
  echo "============================================"
  benchy "third_party/py/flax/benchmarks/tracing:${target}"
  echo ""
done
