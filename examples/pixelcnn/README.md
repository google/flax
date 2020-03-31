## PixelCNN++ image modelling
Trains a PixelCNN++ model [(Salimans et al.,
2017)](https://arxiv.org/abs/1701.05517) for image generation on the CIFAR-10 dataset.
Only unconditional image generation is implemented, trained using ADAM
on the negative log-likelihood. As in the original [OpenAI implementation](https://github.com/openai/pixel-cnn)
we use weightnorm parameterization with data-dependent initialization.

### Requirements
* [TF datasets](https://www.tensorflow.org/datasets), which will download and cache the CIFAR-10 dataset the first time you
  run `train.py`.

### Supported setups
The model should run with other configurations and hardware, but was tested on the following.

| Hardware | Batch size | Training time | Log-likelihood (bits/dimension) | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| 8 x Nvidia V100 (16GB)  | 320  |  1d 14h | 2.92 | [2020-04-23](https://tensorboard.dev/experiment/t8fM3u2zSJG7tAx6YbXHkQ/) |

### How to run
#### 8 x Nvidia V100 (16GB)
```
python train.py --batch_size=320
```
