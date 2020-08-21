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

#### 8 x Nvidia V100 (16GB)
`python imagenet_main.py --batch_size=512 --cache=True --model_dir=./imagenet_fp32_bs512`

#### 8 x Nvidia V100 (16GB), mixed precision
`python imagenet_main.py --batch_size=2048 --cache=True --model_dir=./imagenet_fp16_bs2048 --half_precision=True --loss_scaling=256.`

### Reproducibility
Making the ImageNet classification example reproducible is WIP. For more details, follow [#291](https://github.com/google/flax/issues/291).
