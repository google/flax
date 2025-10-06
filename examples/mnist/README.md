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
I1009 17:56:42.674334 3280981 train.py:175] epoch: 10, train_loss: 0.0073, train_accuracy: 99.75, test_loss: 0.0294, test_accuracy: 99.25
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
