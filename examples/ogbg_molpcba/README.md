## Predicting Biological Activities of Molecules with Graph Neural Networks

This example trains a Graph Neural Network to classify molecules on the
basis of their biological activities.

![Prediction on a caramboxin molecule](www.gstatic.com/flax_examples/ogbg_molpcba.svg "Prediction on a caramboxin molecule")

We use [Jraph](https://github.com/deepmind/jraph/),
a JAX library for Graph Neural Networks, to
train and evaluate on the
[ogbg-molpcba](https://ogb.stanford.edu/docs/graphprop/)
dataset, part of the [Open Graph Benchmark](https://ogb.stanford.edu/).

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

For more extensive changes, you can directly edit the default
configuration file or even add your own.

### Supported Setups

This example supports only single device training.
The model should run with other configurations and hardware, but was explicitly
tested on the following.

Hardware | Batch size | Training time | Test mean AP  | Validation mean AP | Metrics
-------- | ---------- | ------------- | ------- | ------- | ---------------
1x V100  | 256        |   3h20m       | 0.244   | 0.252   |[2021-08-03](https://tensorboard.dev/experiment//)

These metrics reported above are obtained at the end of training.
We observed that slightly higher metrics can be obtained with
early-stopping based on the validation mean AP:

Hardware | Batch size | Training time | Test mean AP  | Validation mean AP | Metrics
-------- | ---------- | ------------- | ------- | ------- | ---------------
1x V100  | 256        |   2h55m       | 0.249   | 0.257   |[2021-08-03](https://tensorboard.dev/experiment//)


### Model Description

The default configuration corresponds to a
[Graph Convolutional Network](https://arxiv.org/abs/1609.02907)
model with 695,936 parameters.

We noticed diminishing gains when training for longer.
Further, the addition of self-loops and undirected edges significantly
helped performance.
Minor improvements were seen with skip-connections across message-passing
steps, together with [LayerNorm](https://arxiv.org/abs/1607.06450).
On the contrary, we found that the addition of
[virtual nodes](https://arxiv.org/abs/1709.03741),
which are connected to all nodes in each graph,
did not improve performance.

### References

- Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren,
  Bowen Liu, Michele Catasta and Jure Leskovec (2020).
  *Open Graph Benchmark: Datasets for Machine Learning on Graphs.*
  In Advances in Neural Information Processing Systems 33: Annual
  Conference on Neural Information Processing Systems 2020,
  NeurIPS 2020, December 6-12,
  2020, virtual.

- Thomas N. Kipf and Max Welling (2016). *Semi-supervised classification
  with graph convolutional networks.* arXiv preprint arXiv:1609.02907.

- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton (2016). *Layer
  normalization.* arXiv preprint arXiv:1607.06450.

- Junying Li, Deng Cai and Xiaofei He (2017). *Learning graph-level
  representation for drug discovery.* arXiv preprint arXiv:1709.03741.

The caramboxin molecule diagram depicted above was obtained and modified from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Caramboxin.svg),
available in the public domain.
