## SST-2 classification

Trains a simple text classifier on the SST-2 sentiment classification dataset.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/google/flax/blob/main/examples/sst2/sst2.ipynb

### Requirements
* TensorFlow dataset `glue/sst2` will be downloaded and prepared automatically, if necessary.

### Example output

| Name    | Platform  |  Epochs | Walltime   | Accuracy   | Metrics                                                                                                               | Workdir                                                                                                                        |
|:--------|:--------|--------:|:-----------|:-----------------|:----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| default | TPU     |     10 | 4.3m    | 85.21%           | [tensorboard.dev](https://tensorboard.dev/experiment/yTQjjRY9RlGRrZzg8h9PJw/) | |

```
INFO:absl:train epoch 010 loss 0.1918 accuracy 92.41
INFO:absl:eval  epoch 010 loss 0.4144 accuracy 85.21
```

### How to run

```bash
python main.py --workdir=/tmp/sst2 --config=configs/default.py`
```

#### Overriding Hyperparameter configurations

The SST2 example allows specifying a hyperparameter configuration by means of
setting the `--config` flag. The configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
    --workdir=/tmp/sst2 --config=configs/default.py \
    --config.learning_rate=0.05 --config.num_epochs=5
```
