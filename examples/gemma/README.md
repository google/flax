
## Language modeling
Trains Gemma model on the One Billion Word Benchmark (lm1b; Chelba *et al.*, 2013).

This example is based on lm1b and similarly uses linear learning rate warmup and inverse square root learning rate schedule.


### Requirements

*   TensorFlow datasets `lm1b` need to be downloaded and prepared (see below).
    A sentencepiece tokenizer vocabulary will be automatically generated
    and saved on each training run.
*   This example additionally depends on the `sentencepiece` and `tensorflow-text` packages.

### Downloading the LM1B Datasets

We recommend downloading and preparing the TFDS datasets beforehand. You can download and prepare LM1B datasets using TFDS directly: `python -m tensorflow_datasets.scripts.download_and_prepare --datasets=lm1b`.

#### Using Cloud Storage FUSE for TPUs

For Cloud TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be mounted to the TPU VM using [Cloud Storage FUSE](https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket).

##### Copy the preprocessed dataset to the Cloud Storage

We assume that the dataset was downloaded and prepared. We also assume we have configured `gcloud` CLI. The following commands helps to setup the storage and copy the dataset:

```bash
# Install gcsfuse CLI
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
# For example, GCSFUSE_REPO=gcsfuse-noble for Ubuntu 24.04

echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y fuse gcsfuse --no-install-recommends

gcsfuse -v
# gcsfuse version 2.12.2 (Go version go1.24.0)
```

Let's get where LM1B dataset was locally stored:
```bash
python -c "import tensorflow_datasets as tfds; b=tfds.builder('lm1b'); print(b.info.data_dir)"
# For example: /home/user/tensorflow_datasets/lm1b/1.1.0
```

Let's create a GCS bucket for the dataset and link the bucket to a local folder. We choose the bucket name "flax-lm1b-tfdataset" but this can be changed.
```bash
gcloud storage buckets create gs://flax-lm1b-tfdataset

mkdir -p $HOME/data
gcsfuse flax-lm1b-tfdataset $HOME/data
```

Now let's copy the data to the bucket:
```bash
# Let's assume that prepared dataset is at $HOME/tensorflow_datasets/lm1b/
cp -R $HOME/tensorflow_datasets/lm1b $HOME/data
```

##### Setup the dataset on TPU VM

We previously have choosen the bucket name "flax-lm1b-tfdataset" where stored the dataset, adapt this name to your situation.

```bash
# On the TPU VM
gcsfuse flax-lm1b-tfdataset $HOME/tensorflow_datasets

ls $HOME/tensorflow_datasets/lm1b/1.1.0/
```

### How to run on GPU(s)

Install Jax with CUDA support, Flax and the example dependencies with the following command:
```bash
pip install jax[cuda12]
# Check whether GPUs are available:
# python3 -c "import jax; print(jax.devices())"

git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
pip install -e .
cd examples/gemma
pip install -r requirements.txt
```

Start the training:

- train a small transformer model:
```bash
python3 main.py --workdir=$HOME/logs/small_gemma_lm1b --config=configs/small.py
```

- train Gemma3-4B model:
```bash
python3 main.py --workdir=$HOME/logs/gemma3-4b_lm1b --config=configs/gemma3_4b.py
```

To monitor the trainings with the TensorBoard:
```bash
tensorboard --logdir=$HOME/logs
```


### How to run on Cloud TPUs

Setup the TPU VM and install the Flax dependencies on it as described
[here](https://cloud.google.com/tpu/docs/jax-pods) for creating pod slices, or
[here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) for a single
v4-8 TPU.


First create a single TPUv4-8 VM and connect to it (you can find more detailed
instructions [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)):

```bash
ZONE=us-central1-a
TPU_TYPE=v4-8
TPU_NAME=$USER-flax-gemma-lm1b
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version tpu-ubuntu2204-base

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE -- \
    -L 6006:localhost:6006
```

When connected install JAX:

```bash
pip install "jax[tpu]>=0.2.16" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install Flax + the example dependencies:

```bash
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
pip install -e .
cd examples/gemma
pip install -r requirements.txt
```

In case of errors when installing example dependencies, try to upgrade existing `pip` package and downgrade `setuptools` and repeat the installation command
```bash
# Optionally
# pip install -U pip
# pip install -U "setuptools<70"
# pip install -r requirements.txt
```

And finally start the training:

```bash
python3 main.py --workdir=$HOME/logs/gemma_lm1b_256 --config.per_device_batch_size=32
```

Note that you might want to set `TFDS_DATA_DIR` as explained below. You probably
also want to start the long-running command above in a `tmux` session and start
some monitoring in a separate pane (note that we forwarded port 6006 locally
above):

```bash
tensorboard --logdir=$HOME/logs
```
