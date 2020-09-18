## MNIST classification

Trains a simple convolutional network on the MNIST dataset.

### Requirements
* TensorFlow dataset `mnist` will be downloaded and prepared automatically, if necessary

### Example output

```
I0527 09:46:48.913452 139989059204928 mnist_lib.py:128] train epoch: 10, loss: 0.0064, accuracy: 99.82
I0527 09:46:50.714237 139989059204928 mnist_lib.py:178] eval epoch: 10, loss: 0.0327, accuracy: 99.07
```

### How to run

`python mnist_main.py --model_dir=/tmp/mnist`

Optionally set `--config` flag value to a python file path which returns
a hyperparameter configuration. This usually is an instance of [ConfigDict](https://github.com/google/ml_collections/tree/master#configdict).
By default, this is set to `configs/default.py`.