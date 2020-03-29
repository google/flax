## CIFAR10 classification
Trains a **family** ResNet-style models (He *et al.*, 2015; Zagoruyko and Komodakis, 2017; Han *et al.*, 2017) for the CIFAR10 classification task (Krizhevsky, 2009).

This example implements several architectures, regularization methods (Gastaldi, 2017; Yamada *et al.*, 2018) and learning rate schedules that can be used in various combinations.

### Requirements
* TensorFlow dataset `cifar10`

### Supported setups
The model should run with other configurations and hardware, but explicitly tested on the following.

#### Wide ResNet: 26 layers, 10x width (Zagoruyko and Komodakis, 2017)
| Hardware | Epochs | Learning rate | Training time | Error rate | TensorBoard.dev |
| --- | --- | --- | --- | --- | --- |
| 1 x Nvidia V100 (16GB) | 200 | Piece-wise constant | 4h 36m | 4.45% | [2020-03-22](https://tensorboard.dev/experiment/1kE2bq9RR7WrT7Zw14fNpg/) |
| 8 x Nvidia V100 (16GB) | 200 | Piece-wise constant | 57m | 3.93% | [2020-03-22](https://tensorboard.dev/experiment/IpzU0txnR7uZnExGmk7IEA/) |

#### Wide ResNet: 26 layers, 6x width, Shake-Shake regularization (Gastaldi, 2017)
| Hardware | Epochs | Learning rate | Training time | Error rate | TensorBoard.dev |
| --- | --- | --- | --- | --- | --- |
| 1 x Nvidia V100 (16GB) | 200 | Piece-wise constant | 3h 38m | 3.43% | [2020-03-22](https://tensorboard.dev/experiment/dJc9e4k1R5mC0DPVZyYrpg/) |
| 8 x Nvidia V100 (16GB) | 200 | Piece-wise constant | 54m | 3.39% | [2020-03-26](https://tensorboard.dev/experiment/l2CPAqpnTlCjjZKbnuOomg/) |
| 1 x Nvidia V100 (16GB) | 1800 | Cosine | 1d 9h 25m | 2.97% | [2020-03-22](https://tensorboard.dev/experiment/aXyhYX2oSxKH5lIlj4Ao5g/) |
| 8 x Nvidia V100 (16GB) | 1800 | Cosine | 8h 5m | 2.82% | [2020-03-26](https://tensorboard.dev/experiment/kq303ri1RHygWxb5YrLp1Q/) |

#### PyramidNet, Shake-drop regularization (Han *et al.*, 2017; Yamada *et al.*, 2018)
| Hardware | Epochs | Learning rate | Training time | Error rate | TensorBoard.dev |
| --- | --- | --- | --- | --- | --- |
| 8 x Nvidia V100 (16GB) | 300 | Piece-wise constant | 6h 41m | 3.25% | [2020-03-24](https://tensorboard.dev/experiment/OpZLDnVjRhmJKXq62RoaUQ/) |
| 8 x Nvidia V100 (16GB) | 1800 | Cosine | 1d 16h 27m | 2.75% | [2020-03-24](https://tensorboard.dev/experiment/MNyJ2ixAROmFlVnbUxp29w/) |

### How to run
All models were trained with a global batch size of `256`.

#### Wide ResNet: 26 layers, 10x width
`python train.py --arch=wrn26_10 --model_dir=./cifar10_wrn26_10_bs=256_lr=0.1`

#### Wide ResNet: 26 layers, 6x width, Shake-shake regularization
`python train.py --arch=wrn26_6_ss --model_dir=./cifar10_wrn26_6_ss_bs=256_lr=0.1`
or
`python train.py --arch=wrn26_6_ss --lr_schedule=cosine --num_epochs=1800 --model_dir=./cifar10_wrn26_6_ss_bs=256_lr=cosine_epochs=1800`

#### PyramidNet, Shake-drop regularization
`python train.py --arch=pyramid --lr_sched_steps="[[150,0.1],[225,0.01]]" --num_epochs=300 --l2_reg=0.0001 --model_dir=./cifar10_pyramid_bs=256_lr=0.1_l2=0.0001_epoch=300`
or
`python train.py --arch=pyramid --lr_sched_steps=cosine --num_epochs=1800 --l2_reg=0.0001 --model_dir=./cifar10_pyramid_bs=256_lr=cosine_l2=0.0001_epochs=1800`

## Known issues
L2 regularization is applied to model kernels *and* biases, instead of only being applied to kernels.

### References
This example consulted the following open-source repositories for implementation details and hyper-parameters:
* [Shake-shake author's implementation](https://github.com/xgastaldi/shake-shake)
* [PyTorch implementation of shake-shake](https://github.com/owruby/shake-shake_pytorch)
* [Fast AutoAugment that uses a number of different models](https://github.com/kakaobrain/fast-autoaugment)