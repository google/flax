## MNIST classification

Trains a simple convolutional network on the MNIST dataset.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb

### Requirements
* TensorFlow dataset `mnist` will be downloaded and prepared automatically, if necessary

### Example output

|  Name   | Epochs | Walltime | Top-1 accuracy |   Metrics   |                  Workdir                  |
| :------ | -----: | :------- | :------------- | :---------- | :---------------------------------------- |
| default |     10 | 7.7m     | 99.17%         | [tfhub.dev] | [gs://flax_public/examples/mnist/default] |

[tfhub.dev]: https://tensorboard.dev/experiment/1G9SvrW5RQyojRtMKNmMuQ/#scalars&_smoothingWeight=0&regexInput=default
[gs://flax_public/examples/mnist/default]: https://console.cloud.google.com/storage/browser/flax_public/examples/mnist/default

```
I0828 08:51:41.821526 139971964110656 train.py:130] train epoch: 10, loss: 0.0097, accuracy: 99.69
I0828 08:51:42.248714 139971964110656 train.py:180] eval epoch: 10, loss: 0.0299, accuracy: 99.14
```

### How to run

`python main.py --workdir=/tmp/mnist --config=configs/default.py`

#### Overriding Hyperparameter configurations

MNIST example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
--workdir=/tmp/mnist --config=configs/default.py \
--config.learning_rate=0.05 --config.num_epochs=5
```
