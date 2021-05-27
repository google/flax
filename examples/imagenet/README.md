## ImageNet classification

Trains a ResNet50 model ([He *et al.*, 2015]) for the ImageNet classification task
([Russakovsky *et al.*, 2015]).

This example uses linear learning rate warmup and cosine learning rate schedule.

[He *et al.*, 2015]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[Russakovsky *et al.*, 2015]: https://arxiv.org/pdf/1409.0575.pdf

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/google/flax/blob/master/examples/imagenet/imagenet.ipynb

The Colab also demonstrates how to load pretrained checkpoints from Cloud
storage at
[gs://flax_public/examples/imagenet/](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet)

### Requirements

* TensorFlow dataset `imagenet2012:5.*.*`
* `≈180GB` of RAM if you want to cache the dataset in memory for faster IO

### Supported setups

While the example should run on a variety of hardware, 
we have tested the different configurations on TPU v3-8 and 8x V100 (16GB)
and got the following results:

| Name                    |   Steps | Walltime   | Top-1 accuracy   | Metrics                                                                                                                               | Workdir                                                                                                                                                              |
|:------------------------|--------:|:-----------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TPU v3-8                 |  125100 | 4.4h      | 76.37%           | [tfhub.dev](https://tensorboard.dev/experiment/JwxRMYrsR4O6V6fnkn3dmg/)                 | [gs://flax_public/examples/imagenet/tpu](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet/tpu)                                 |
| v100_x8                 |  250199 | 13.7h      | 76.72%           | [tfhub.dev](https://tensorboard.dev/experiment/iJzNKovmS0q6k5t6k5wvOw/#scalars&_smoothingWeight=0&regexInput=v100_x8$)                 | [gs://flax_public/examples/imagenet/v100_x8](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet/v100_x8)                                 |
| v100_x8_mixed_precision |   62499 | 5.0h       | 76.47%           | [tfhub.dev](https://tensorboard.dev/experiment/iJzNKovmS0q6k5t6k5wvOw/#scalars&_smoothingWeight=0&regexInput=v100_x8_mixed_precision) | [gs://flax_public/examples/imagenet/v100_x8_mixed_precision](https://console.cloud.google.com/storage/browser/flax_public/examples/imagenet/v100_x8_mixed_precision) |

### Preparing the dataset on Cloud

For running the ResNet50 model on imagenet dataset, 
you first need to prepare the `imagenet2012` dataset. 
Download the data from http://image-net.org/ as described in the
[tensorflow_datasets catalog](https://www.tensorflow.org/datasets/catalog/imagenet2012). 
Then point the environment variable `$IMAGENET_DOWNLOAD_PATH` 
to the directory where the downloads are stored and prepare the dataset 
by running

```shell
python -c "
import tensorflow_datasets as tfds
tfds.builder('imagenet2012').download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        manual_dir='$IMAGENET_DOWNLOAD_PATH'))
"
```

The contents of the directory `~/tensorflow_datasets` should be copied to your gcs bucket. Point the environment variable `GCS_TFDS_BUCKET` to your bucket and run the following command:

```shell
gsutil cp -r ~/tensorflow_datasets gs://$GCS_TFDS_BUCKET/datasets
```

### How to run on Cloud TPU



Setup the TPU VM and install the Flax dependencies on it as described 
[here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

Set the environment variable of `GCS_TFDS_BUCKET` to your bucket on the TPU VM 
and run the model after setting `TFDS_DATA_DIR` parameter:

```shell
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets
python3 main.py --workdir=./imagenet_tpu --config=configs/tpu.py
```



### How to run

```shell
python main.py --workdir=./imagenet --config=configs/default.py
```

#### Overriding Hyperparameter configurations

Specify a hyperparameter configuration by the means of setting `--config` flag.
Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py --workdir=./imagenet_default --config=configs/default.py --config.num_epochs=100
```

#### 8 x Nvidia V100 (16GB)

```shell
python main.py \
--workdir=./imagenet_v100_x8 --config=configs/v100_x8.py
```

#### 8 x Nvidia V100 (16GB), mixed precision

```shell
python main.py \
--workdir=./imagenet_v100_x8_mixed_precision \
--config=configs/v100_x8_mixed_precision.py
```
#### How to run on Cloud GPU

See commands in [../cloud/README.md](../cloud/README.md)

### Reproducibility

Making the ImageNet classification example reproducible is WIP. 
See: [#291](https://github.com/google/flax/issues/291).
