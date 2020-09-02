# Object Detection

Trains a RetinaNet model ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) on the COCO/2014 `trainval35k` split.

Also features logic for anchor manipulation, label and regression targets inference, and post-inference processing (Thresholding, Top-K selection, Non-Maximum Suppression). 

This example uses a division based fixed learning rate schedule. 

## Requirements

* TensorFlow datasets `coco/2014` need to be downloaded. The location of the dataset needs to be specified in the `COCO_ANNOTATIONS_PATH` environment variable.
* The example also depends on the `pycocotools` and `mlconfig` packages. 

## Supported setups
The model should run with other configurations and hardware, but was explicitly tested on the following.

| Hardware | Batch size | Training time | mAP | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| TPU v3-8  | 16=2x8  |  -  | - | - |

## How to run

`python main.py --config=<config_path> --workdir=<workdir_path> --jax_backend_target=<backend_target>`

Below is an explanation of the CL parameters:

* **config**: a path to the `mlconfig` config source. For more details on the available parameters, plese see the `configs/default.py` source.
* **workdir**: the workdir for the model. This is generally useful for TensorFlow elements of the application. 
* **jax_backend_target**: refers to the computational resource used to execute `jax` operations.

## Running on Cloud TPUs

Creating a [Cloud TPU](https://cloud.google.com/tpu/docs/quickstart) involves creating the user Google Compute Engine (GCE) VM and the TPU node.

To create a user GCE VM, run the following command from your Google Cloud Platform (GCP) console or your computer terminal where you have [gCloud SDK installed](https://cloud.google.com/sdk/install).

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

To create a larger GCE VM, choose a different [`machine type`](https://cloud.google.com/compute/docs/machine-types).

Next, create the TPU node, following these [guidelines](https://cloud.google.com/tpu/docs/internal-ip-blocks) to choose a <TPU_IP_ADDRESS>. e.g.:

```
export TPU_IP_ADDRESS=192.168.0.2
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v3-8 \
      --range=$TPU_IP_ADDRESS \
      --version=tpu_driver_nightly
```

Now that you have created both the user GCE VM and the TPU node, ssh to the GCE VM by executing the following command, including a local port forwarding rule for viewing tensorboard:

```
gcloud compute ssh $USER-user-vm-0001 -- -L 2222:localhost:8888
```

Be sure to install the example's requirements by running the commands below. For a detailed list of the required libraries, see `requirements.txt`.

```
pip install -U pip
pip install -U setuptools wheel
pip install -U -r requirements.txt 
git clone https://github.com/google/flax
pip install -e flax
```

Then, if your TPU is at IP `192.168.0.2`:

`python main.py --config=<config_path> --workdir=<workdir_path> --jax_backend_target="grpc://192.168.0.2:8470"`

A tensorboard instance can then be launched and viewed on your local 2222 port via the tunnel:

`tensorboard --logdir wmt_256 --port 8888`

## Downloading TFDS Datasets

We recommend downloading the TFDS datasets beforehand. For Cloud TPUs, we recommend using a cheap standard instance and saving the prepared TFDS data on a storage bucket from where it can be loaded directly during training using the `export COCO_ANNOTATIONS_PATH=gs://...` option.
