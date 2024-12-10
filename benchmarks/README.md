# Benchmarks

These are mini benchmarks to measure the performance of NNX operations.

Sample profile command:

```shell
python -m cProfile -o ~/tmp/overhead.prof benchmarks/nnx_graph_overhead.py --mode=nnx --depth=100 --total_steps=1000
```

Sample profile inspection:

```shell
snakeviz ~/tmp/overhead.prof
```