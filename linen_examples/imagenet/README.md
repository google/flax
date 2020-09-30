## ImageNet classification
Trains a ResNet50 model (He *et al.*, 2015) for the ImageNet classification task (Russakovsky *et al.*, 2015).

This example uses linear learning rate warmup and cosine learning rate schedule.

### Requirements
* TensorFlow dataset `imagenet2012:5.*.*`
* `≈180GB` of RAM if you want to cache the dataset in memory for faster IO

### Supported setups
The model should run with other configurations and hardware, but explicitely tested on the following.

| Hardware | Batch size | Training time | Top-1 accuracy  | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| 8 x Nvidia V100 (16GB)  | 512  |  13h 25m  | 76.63% | [2020-03-12](https://tensorboard.dev/experiment/jrvtbnlETgai0joLBXhASw/) |
| 8 x Nvidia V100 (16GB), mixed precision  | 2048  | 6h 4m | 76.39% | [2020-03-11](https://tensorboard.dev/experiment/F5rM1GGQRpKNX207i30qGQ/) |

### How to run

```shell
python imagenet_main.py --model_dir=./imagenet
```

#### Overriding Hyperparameter configurations

Specify a hyperparameter configuration by the means of setting `--config` flag.
Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python imagenet_main.py --model_dir=./imagenet_default --config.num_epochs=100
```

#### 8 x Nvidia V100 (16GB)
```shell
python imagenet_main.py \
--model_dir=./imagenet_v100_x8 --config=configs/v100_x8.py
```

#### 8 x Nvidia V100 (16GB), mixed precision
```shell
python imagenet_main.py \
--model_dir=./imagenet_v100_x8_mixed_precision \
--config=configs/v100_x8_mixed_precision.py
```

### Reproducibility
Making the ImageNet classification example reproducible is WIP. 
See: [#291](https://github.com/google/flax/issues/291).