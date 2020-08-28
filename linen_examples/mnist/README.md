## MNIST classification

Trains a simple convolutional network on the MNIST dataset.

### Requirements
* TensorFlow dataset `mnist` will be downloaded and prepared automatically, if necessary

### Example output

```
I0828 08:51:41.821526 139971964110656 mnist_lib.py:128] train epoch: 10, loss: 0.0097, accuracy: 99.69
I0828 08:51:42.248714 139971964110656 mnist_lib.py:178] eval epoch: 10, loss: 0.0299, accuracy: 99.14
```

### How to run

`python mnist_main.py --model_dir=/tmp/mnist`
