
## Language modeling
Trains a Transformer-based model (Vaswani *et al.*, 2017) on the One Billion
Word Benchmark (lm1b; Chelba *et al.*, 2013).

This example uses linear learning rate warmup and inverse square root learning
rate schedule. Based off of Machine Translation `wmt` example.


### Requirements

*   TensorFlow datasets `lm1b` need to be downloaded and prepared.
    A sentencepiece tokenizer vocabulary will be automatically generated
    and saved on each training run.
*   This example additionally depends on the `sentencepiece` and
    `tensorflow-text` packages.

### How to run

`python main.py --workdir=./lm1b_64 --config=configs/default.py`

### How to run on Cloud TPUs

Creating a [Cloud TPU](https://cloud.google.com/tpu/docs/quickstart) involves
creating the user GCE VM and the TPU node.

To create a user GCE VM, run the following command from your GCP console or your
computer terminal where you have
[gcloud installed](https://cloud.google.com/sdk/install).

```
export ZONE=europe-west4-a
gcloud compute instances create $USER-user-vm-0001 \
   --machine-type=n1-standard-16 \
   --image-project=ml-images \
   --image-family=tf-2-2 \
   --boot-disk-size=200GB \
   --scopes=cloud-platform \
   --zone=$ZONE
```

To create a larger GCE VM, choose a different
[machine type](https://cloud.google.com/compute/docs/machine-types).

Next, create the TPU node, following these
[guidelines](https://cloud.google.com/tpu/docs/internal-ip-blocks) to choose a
<TPU_IP_ADDRESS>. e.g.:

```
export TPU_IP_ADDRESS=192.168.0.2
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v3-8 \
      --range=$TPU_IP_ADDRESS \
      --version=tpu_driver_nightly
```

Now that you have created both the user GCE VM and the TPU node, ssh to the GCE
VM by executing the following command, including a local port forwarding rule
for viewing tensorboard:

```
gcloud compute ssh $USER-user-vm-0001 -- -L 2222:localhost:8888
```

Be sure to install the latest `jax` and `jaxlib` packages alongside the other
requirements above. e.g. as of April 2020 the following package versions were
used successfully: `pip install -U pip pip install -U setuptools wheel pip
install -U pip jax jaxlib sentencepiece \ tensorflow==2.2.0rc3
tensorflow-datasets==3.0.0 \ tensorflow-text==2.2.0rc2 tensorboard git clone
https://github.com/google/flax pip install -e flax`

Then, if your TPU is at IP `192.168.0.2`: `python main.py --workdir=./lm1b_64
--config.per_device_batch_size=64 --jax_backend_target="grpc://192.168.0.2:8470"`

A tensorboard instance can then be launched and viewed on your local 2222 port
via the tunnel: `tensorboard --logdir lm1b_256 --port 8888`

### Downloading the LM1B Datasets

We recommend downloading and preparing the TFDS datasets beforehand. For Cloud
TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be loaded directly. Set the
`TFDS_DATA_DIR` to your storage bucket path (`gs://<bucket name>`).

You can download and prepare LM1B datasets using TFDS directly:
`python -m tensorflow_datasets.scripts.download_and_prepare
--datasets=lm1b`

