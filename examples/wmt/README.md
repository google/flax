## Machine Translation

Trains a Transformer-based model (Vaswani *et al.*, 2017) on the WMT Machine
Translation en-de dataset.

This example uses linear learning rate warmup and inverse square root learning
rate schedule.

### Requirements

*   TensorFlow datasets `wmt17_translate/de-en` and `wmt14_translate/de-en` need
    to be downloaded and prepared. A sentencepiece tokenizer vocabulary will be
    automatically generated and saved on each training run.
*   This example additionally depends on the `sentencepiece` and
    `tensorflow-text` packages.

### Supported setups

The model should run with other configurations and hardware, but was explicitly
tested on the following.

Hardware | Batch size | Training time | BLEU  | TensorBoard.dev
-------- | ---------- | ------------- | ----- | ---------------
TPU v3-8 | 256        | 1h 35m        | 25.13 | [2020-04-21](https://tensorboard.dev/experiment/9lsbEw7DQzKdv881v4nIQA/)

### How to run

`python main.py --workdir=./wmt_256 --config=configs/default.py --config.reverse_translation=True`

### How to run on Cloud TPUs

Creating a [Cloud TPU](https://cloud.google.com/tpu/docs/quickstart) involves
creating the user GCE VM and the TPU node.

To create a user GCE VM, run the following command from your GCP console or your
computer terminal where you have
[gcloud installed](https://cloud.google.com/sdk/install).

Depending on current availability, you might need to choose a different
[zone with TPUs](https://cloud.google.com/tpu/docs/types-zones).

```
export ZONE=europe-west4-a
gcloud compute instances create $USER-user-vm-0001 \
   --machine-type=n1-standard-16 \
   --image-project=ml-images \
   --image-family=tf-2-4-2 \
   --boot-disk-size=200GB \
   --scopes=cloud-platform \
   --zone=$ZONE
```

To create a larger GCE VM, choose a different
[machine type](https://cloud.google.com/compute/docs/machine-types).

```
export TPU_IP_RANGE=192.168.0.0/29
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v3-8 \
      --range=$TPU_IP_RANGE \
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
used successfully:

```
pip3 install -U pip &&
pip3 install -U setuptools wheel &&
pip3 install -U pip jax jaxlib sentencepiece &&
pip3 install -U tensorflow==2.4.1 tensorflow-datasets tensorflow-text>=2.4.0 &&
git clone https://github.com/google/flax &&
pip3 install -e flax
```

Assuming the TPU node is at IP `192.168.0.2` (default with above arguments; you
can see address via `gcloud compute tpus list --zone=$ZONE`), start the
training (see note below about setting `TFDS_DATA_DIR`):

```
cd flax/examples/wmt

python3 main.py --workdir=./wmt_256 \
    --config.per_device_batch_size=32 \
    --jax_backend_target="grpc://192.168.0.2:8470"`
```

A tensorboard instance can then be launched and viewed on your local 2222 port
via the tunnel: `tensorboard --logdir wmt_256 --port 8888`

### Downloading the WMT Datasets

We recommend downloading and preparing the TFDS datasets beforehand. For Cloud
TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be loaded directly. Set the
`TFDS_DATA_DIR` to your storage bucket path (`gs://<bucket name>`).

You can download and prepare any of the WMT datasets using TFDS directly:
`python -m tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt17_translate/de-en`

The typical academic BLEU evaluation also uses the WMT 2014 Test set: `python -m
tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt14_translate/de-en`
