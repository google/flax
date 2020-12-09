## ImageNet classification
Trains a ResNet50 model (He *et al.*, 2015) for the ImageNet classification task (Russakovsky *et al.*, 2015).

This example uses linear learning rate warmup and cosine learning rate schedule.

### Requirements
* TensorFlow dataset `imagenet2012:5.*.*`
* `≈180GB` of RAM if you want to cache the dataset in memory for faster IO

### Supported setups
The model should run with other configurations and hardware, but explicitely tested on the following.

| Name                    |   Steps | Walltime   | Top-1 accuracy   | Metrics                                                                                                                               | Workdir                                                                                                                                                              |
|:------------------------|--------:|:-----------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v100_x8                 |  250199 | 13.7h      | 76.72%           | [tfhub.dev](https://tensorboard.dev/experiment/iJzNKovmS0q6k5t6k5wvOw/#scalars&_smoothingWeight=0&regexInput=v100_x8$)                 | [gs://flax_public/examples/imagenet/v100_x8](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet/v100_x8)                                 |
| v100_x8_mixed_precision |   62499 | 5.0h       | 76.47%           | [tfhub.dev](https://tensorboard.dev/experiment/iJzNKovmS0q6k5t6k5wvOw/#scalars&_smoothingWeight=0&regexInput=v100_x8_mixed_precision) | [gs://flax_public/examples/imagenet/v100_x8_mixed_precision](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet/v100_x8_mixed_precision) |

### How to run

```shell
python main.py --workdir=./imagenet
```

#### Overriding Hyperparameter configurations

Specify a hyperparameter configuration by the means of setting `--config` flag.
Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py --workdir=./imagenet_default --config.num_epochs=100
```

#### 8 x Nvidia V100 (16GB)

```shell
python main.py \
--workdir=./imagenet_v100_x8 --config=configs/v100_x8.py
```

#### 8 x Nvidia V100 (16GB), mixed precision

```shell
python main.py \
--workdir=./imagenet_v100_x8_mixed_precision \
--config=configs/v100_x8_mixed_precision.py
```

### Reproducibility

Making the ImageNet classification example reproducible is WIP. 
See: [#291](https://github.com/google/flax/issues/291).