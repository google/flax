## MNIST classification

Trains a simple convolutional network on the MNIST dataset
with differential privacy.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/google/flax/blob/main/examples/mnist_dp/mnist_dp.ipynb

### Requirements
* TensorFlow dataset `mnist` will be downloaded and prepared automatically,
  if necessary.

### Example Output

|  Name   | Epochs | Walltime | Top-1 accuracy |   Metrics   |                  Workdir                  |
| :------ | -----: | :------- | :------------- | :---------- | :---------------------------------------- |
| default |     |  |         | |  |


### How to Run

`python main.py --workdir=/tmp/mnist_dp --config=configs/default.py`

#### Overriding Hyperparameter configurations

MNIST example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
--workdir=/tmp/mnist_dp --config=configs/default.py \
--config.learning_rate=0.05 --config.num_epochs=5
```
