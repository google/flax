## Predicting the Biological Activity of Molecules with Graph Neural Networks

This example trains a Graph Neural Network to classify molecules on the
basis of their biological activities.

We train and evaluate on the
[ogbg-molpcba](https://ogb.stanford.edu/docs/graphprop/)
dataset, which is part of the [Open Graph Benchmark](https://ogb.stanford.edu/),
and define our Graph Neural Network
using [Jraph](https://github.com/deepmind/jraph/).

### Requirements

We depend on
[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/ogbg_molpcba)
for ogbg-molpcba.

### How to Run

To run with the default configuration:

```shell
python main.py --workdir=./ogbg_molpcba --config=configs/default.py
```

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of epochs
and the batch size:

```shell
python main.py --workdir=./ogbg_molpcba --config=configs/default.py \
--config.num_training_epochs=10 --config.batch_size=50
```

For more extensive changes, you can add your own configuration file.

### References

- Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren,
  Bowen Liu, Michele Catasta and Jure Leskovec (2020).
  *Open Graph Benchmark: Datasets for Machine Learning on Graphs*.
  In Advances in Neural Information Processing Systems 33: Annual Conference
  on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12,
  2020, virtual. URL: https://arxiv.org/abs/2005.00687

