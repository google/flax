## MNIST classification

Trains a simple convolutional network on the MNIST dataset.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/google/flax/blob/master/linen_examples/mnist/mnist.ipynb

### Requirements
* TensorFlow dataset `mnist` will be downloaded and prepared automatically, if necessary

### Example output

```
I0828 08:51:41.821526 139971964110656 mnist_lib.py:130] train epoch: 10, loss: 0.0097, accuracy: 99.69
I0828 08:51:42.248714 139971964110656 mnist_lib.py:180] eval epoch: 10, loss: 0.0299, accuracy: 99.14
```

### How to run

`python mnist_main.py --model_dir=/tmp/mnist`

#### Overriding Hyperparameter configurations

MNIST example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python mnist_main.py \
--model_dir=/tmp/mnist --config=configs/default.py \
--config.learning_rate=0.05 --config.num_epochs=5
```
