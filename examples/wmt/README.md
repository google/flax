## Machine Translation

Trains a Transformer-based model (Vaswani *et al.*, 2017) on the WMT Machine
Translation en-de dataset.

This example uses linear learning rate warmup and inverse square root learning
rate schedule.

Table of contents:

- [Requirements](#requirements)
- [Example runs](#example-runs)
- [Running on Cloud](#running-on-cloud)
  - [Preparing the dataset](#preparing-the-dataset)
  - [Google Cloud TPU](#google-cloud-tpu)

### Requirements

*   TensorFlow datasets `wmt17_translate/de-en` and `wmt14_translate/de-en` need
    to be downloaded and prepared. A sentencepiece tokenizer vocabulary will be
    automatically generated and saved on each training run.
*   This example additionally depends on the `sentencepiece` and
    `tensorflow-text` packages.

### Example runs

You should expect to get numbers similar to these:


Hardware | config  | Training time |      BLEU      |                             TensorBoard.dev                              |                                                          Workdir
-------- | ------- | ------------- | -------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------
TPU v3-8 | default | 24m<br>13h18m | 25.55<br>32.87 | [2021-08-04](https://tensorboard.dev/experiment/nnH7JNCxTgC1ROakWePTlg/) | [gs://flax_public/examples/wmt/default](https://console.cloud.google.com/storage/browser/flax_public/examples/wmt/default)
TPU v3-32 | default | 3h1m | 32.45 | [2021-11-05](https://tensorboard.dev/experiment/7IKeXjoeRKiMtqysQlbqzw/) | [gs://flax_public/examples/wmt/default_v3-32](https://console.cloud.google.com/storage/browser/flax_public/examples/wmt/default_v3-32)
GPU V100 x8 (Mixed Precision) | gpu_mixed_precision        | 1h 58m       | 25.69 | [2021-07-07](https://tensorboard.dev/experiment/9S2WuqNWRDemmBuQE8K6Ew/) | -


### Running on Cloud

#### Preparing the WMT Datasets

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

#### Google Cloud TPU

See below for commands to set up a single VM with 8 TPUs attached
(`--accelerator-type v3-8`), or for an entire TPU slice spanning multiple
VMs (e.g. `--accelerator-type v3-32`). For more details about how to set up and
use TPUs, refer to Cloud docs for
[single VM setup](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
and [pod slice setup](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

First create a single TPUv3-8 VM and connect to it:

```
ZONE=us-central1-a
TPU_TYPE=v3-8
VM_NAME=wmt

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version v2-alpha

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
    -L 6006:localhost:6006
```

When connected install JAX:

```
pip install "jax[tpu]>=0.2.21" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install Flax + the example dependencies:

```
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
pip install -e .
cd examples/wmt
pip install -r requirements.txt
```

And finally start the training:

```
python3 main.py --workdir=$HOME/logs/wmt_256 \
    --config.per_device_batch_size=32 \
    --jax_backend_target="grpc://192.168.0.2:8470"
```

Note that you might want to set `TFDS_DATA_DIR` as explained above. You probably
also want to start the long-running command above in a `tmux` session and start
some monitoring in a separate pane (note that we forwarded port 6006 locally
above):

```
tensorboard --logdir=$HOME/logs
```

When running on pod slices, after creating the TPU VM, there are different ways
of running the training in SPMD fashion on the hosts connected to the TPUs that
make up the slice. We simply send the same installation/execution shell commands
to all hosts in parallel with the command below. If anything fails it's
usually a good idea to connect to a single host and execute the commands
interactively.

For convenience, the TPU creation commands are inlined below. Please note that
we define `GCS_TFDS_BUCKET` to where your data stands in your cloud bucket.
Also `YOUR_BUCKET` is the work directory you are experimenting in. You should
choose ZONE based on where your TPU and work directory is. [Here](https://cloud.google.com/tpu/docs/types-zones)
has some useful information on which zones you can have different types of TPUs.

```shell
VM_NAME=wmt
REPO=https://github.com/google/flax
BRANCH=main
WORKDIR=gs://$YOUR_BUCKET/flax/examples/wmt/$(date +%Y%m%d_%H%M)

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --version v2-alpha --accelerator-type v3-32
FLAGS="--config.num_train_steps=$(( 100 * 1000 * 8/32 ))
--config.warmup_steps=$(( 1000 * 8/32 ))
--config.checkpoint_every_steps=$(( 10 * 1000 * 8/32 ))"

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
set -x
pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
pip install --user git+$REPO.git &&
(test -d flax || git clone --depth=1 -b $BRANCH $REPO) &&
cd flax &&
pip install -e . &&
cd examples/wmt &&
pip install -r requirements.txt &&
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets &&
python3 main.py --workdir=$WORKDIR --config=configs/default.py $FLAGS
"
```
Please don't forget to disconnect and delete your vm after you are done:

```
gcloud alpha compute tpus tpu-vm delete $VM_NAME \
  --zone $ZONE
```

#### Training with FP8 on NVIDIA Hopper GPUs

NVIDIA H100 GPU introduced support for a new datatype, FP8(8-bit floating point), 
enabling higher throughput of matrix multiplies and convolutions. To start 
training with FP8, append command line arguments as
```
python3 main.py --workdir=$HOME/logs/wmt_256 \
    --config.per_device_batch_size=32 \
    --jax_backend_target="grpc://192.168.0.2:8470" \
    --config.use_fp8=True
```

##### Layer configuration
Users can utilize the FP8 feature of nn.Dense or nn.DenseGeneral by passing in 
our custom quantization op nn.Fp8DenseGeneralOp to dot_general_cls.[#3322](https://github.com/google/flax/pull/3322).
More generally, the dot_general_cls from any GEMM-based FLAX layers can be 
injected with the FP8 quantization op if FP8 GEMM is desired. For the 
transformer-based models, we recommend users to configure the QKV projection, 
attention output projection, and linear layers in MLP to utilize the FP8 custom 
op, as demonstrated in this example.

##### TrainState configuration
FP8 usage introduces extra model variables to scale the to-be-quantized 
inputs/outputs of the GEMM. These variables are categorized under the parameter 
collection fp8_params which is in the same level of params. Therefore, the 
pytree structure will be like:

```
{'params': {'kernel':..., ...},
 'fp8_params': {'input_scale':..., ...}}.
```

##### Precision issue
To ensure the numerical stability, the usage of FP8 often requires the wider 
type being fp32 or BFloat16.

To start fp 