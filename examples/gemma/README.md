
## Language modeling

Trains Gemma model on the One Billion Word Benchmark (lm1b; Chelba *et al.*, 2013).

This example is based on `lm1b_nnx` example script.


### Requirements

*   TensorFlow datasets `lm1b` needs to be downloaded and prepared (see below).
    A sentencepiece tokenizer vocabulary will be automatically generated
    and saved on each training run.
*   This example additionally depends on the `sentencepiece` and [`grain`](https://google-grain.readthedocs.io/en/latest/) packages.

### Downloading the LM1B Datasets

We recommend downloading and preparing the TFDS datasets beforehand. You can download and prepare LM1B datasets using TFDS directly:
```bash
tfds build lm1b --file_format=array_record
# To specify the location of downloaded dataset:
# tfds build lm1b --file_format=array_record --data_dir=~/tensorflow_datasets/
# export TFDS_DATA_DIR=~/tensorflow_datasets/
```

#### Using Cloud Storage FUSE for TPUs

For Cloud TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be mounted to the TPU VM using [Cloud Storage FUSE](https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket).

##### Copy the preprocessed dataset to the Cloud Storage

We assume that the dataset was downloaded and prepared. We also assume we have installed and configured `gcloud` CLI (otherwise, please check [this guide](https://cloud.google.com/sdk/docs/install)). The following commands helps to setup the storage and copy the dataset:

```bash
# Install gcsfuse CLI: https://cloud.google.com/storage/docs/cloud-storage-fuse/install
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
# For example, GCSFUSE_REPO=gcsfuse-noble for Ubuntu 24.04
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install -y fuse gcsfuse --no-install-recommends

gcsfuse -v
# gcsfuse version 3.4.0 (Go version go1.24.5)
```

Let's get where LM1B dataset was locally stored:
```bash
python -c "import tensorflow_datasets as tfds; b=tfds.builder('lm1b'); print(b.info.data_dir)"
# For example: /home/user/tensorflow_datasets/lm1b/1.1.0
```

Let's create a GCS bucket for the dataset and link the bucket to a local folder. We choose the bucket name "flax-lm1b-arrayrecords" but this can be changed.
```bash
gcloud storage buckets create gs://flax-lm1b-arrayrecords

mkdir -p $HOME/data
gcsfuse flax-lm1b-arrayrecords $HOME/data
```

Now let's copy the data to the bucket:
```bash
# Let's assume that prepared dataset is at $HOME/tensorflow_datasets/lm1b/
cp -R $HOME/tensorflow_datasets/lm1b $HOME/data
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

- train Gemma3-1B model:
```bash
# Make sure to have set: export TFDS_DATA_DIR=/path/to/tensorflow_datasets/
python3 main.py --workdir=$HOME/logs/gemma3-1b_lm1b --config=configs/default.py
```

- train Gemma3-4B model:
```bash
# Make sure to have set: export TFDS_DATA_DIR=/path/to/tensorflow_datasets/
python3 main.py --workdir=$HOME/logs/gemma3-4b_lm1b --config=configs/gemma3_4b.py
```

To monitor the trainings with the TensorBoard:
```bash
# Open in another terminal:
tensorboard --logdir=$HOME/logs
```


### How to run on Cloud TPUs

#### Single TPU

Setup the TPU VM and install the Flax dependencies on it as described
[here](https://cloud.google.com/tpu/docs/jax-pods) for creating pod slices, or
[here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) for a single
v4-8 TPU.

First, let's create a single v4-8 TPU VM and connect to it (you can find more detailed
instructions [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)):

```bash
ZONE=us-central2-b
TPU_TYPE=v4-8
TPU_NAME=$USER-flax-gemma-lm1b-$TPU_TYPE
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version tpu-ubuntu2204-base

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE -- -L 6006:localhost:6006
```

##### Software setup

When connected to the TPU VM we can install JAX:

```bash
# Setup Python 3.12 env with UV
python -m pip install uv
uv venv --python 3.12 /tmp/venv
source /tmp/venv/bin/activate
uv pip install pip
# which python && python -VV && pip --version

pip install "jax[tpu]"

# Check whether TPUs are available:
# python3 -c "import jax; print(jax.devices())"
```

Then install Flax and the example dependencies:

```bash
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
pip install -e .
cd examples/gemma
pip install -r requirements.txt
```

##### Data setup

Now, let's set up the data access. We previously have choosen the bucket name "flax-lm1b-arrayrecords" where stored the dataset, adapt this name to your situation.

We may need to install gcsfuse on the TPU VM:
```bash
# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-\`lsb_release -c -s\`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc
apt-get update
apt-get install -y fuse gcsfuse --no-install-recommends
```

```bash
mkdir -p $HOME/tensorflow_datasets

gcsfuse -o ro \
  --type-cache-max-size-mb=-1 \
  --stat-cache-max-size-mb=-1 \
  --kernel-list-cache-ttl-secs=-1 \
  --metadata-cache-ttl-secs=-1 \
  flax-lm1b-arrayrecords $HOME/tensorflow_datasets

ls $HOME/tensorflow_datasets/lm1b/1.1.0/
export TFDS_DATA_DIR=$HOME/tensorflow_datasets
```

##### Training

Finally we can start the training:

```bash
python3 main.py --workdir=$HOME/logs/gemma_lm1b_256 --config.per_device_batch_size=32
```

Note that we store the checkpoints and the logs on the TPU VM, we can also mount another cloud bucket for that. You also probably want to start the long-running command above in a `tmux` session and start some monitoring in a separate pane (note that we forwarded port 6006 locally above):

```bash
tensorboard --logdir=$HOME/logs
```

##### Clean-up

Finally, once we are done and TPU VM is unused, let's delete it:
```bash
yes | gcloud compute tpus tpu-vm delete $TPU_NAME --async
```

#### Multi-host TPUs

It is preferable to train large models on multiple TPU VMs to speed-up the training.
Below, we will be using v4-32 TPU containing 32 devices on 4 VMs.


##### TPU setup

As v4-32 TPU has multiple TPU VMs we will create a startup bash script to run on start-up on each VM.

```bash
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=tpu-ubuntu2204-base
export TPU_NAME=flax-gemma-lm1b-${ACCELERATOR_TYPE}

cat << EOF > /tmp/example_startup.sh
#!/bin/bash

set -xeu

python -m pip install uv
uv venv --python 3.12 /tmp/venv
source /tmp/venv/bin/activate
uv pip install pip

echo "source /tmp/venv/bin/activate" > /root/.bashrc

# Install JAX, FLAX and other deps
python -m pip install jax[tpu]

cd /root
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
python -m pip install -e .

python -m pip install \
  "absl-py~=2.2" \
  "clu==0.0.12" \
  "mlcroissant~=1.0" \
  "numpy~=2.2" \
  "optax~=0.2" \
  "sentencepiece~=0.2" \
  "jaxtyping~=0.3" \
  "tensorflow-cpu~=2.19" \
  "tensorflow-datasets~=4.9" \
  "grain~=0.2"

# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-\`lsb_release -c -s\`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt \$GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc
apt-get update
apt-get install -y fuse gcsfuse --no-install-recommends

mkdir -p /root/arrayrecord_datasets
gcsfuse -o ro \
  --type-cache-max-size-mb=-1 \
  --stat-cache-max-size-mb=-1 \
  --kernel-list-cache-ttl-secs=-1 \
  --metadata-cache-ttl-secs=-1 \
  flax-lm1b-arrayrecords /root/arrayrecord_datasets

mkdir -p /root/logs
gcsfuse flax-gemma-example-training /root/logs

EOF

gcloud compute tpus tpu-vm create $TPU_NAME --spot \
    --zone $ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --metadata-from-file=startup-script=/tmp/example_startup.sh
```

We can check the setup progress on all VMs:
```bash
gcloud compute tpus tpu-vm ssh --worker=all --command="journalctl -u google-startup-scripts.service | tail -n 5" $TPU_NAME

# ...
# Oct 15 15:11:29 t1v-n-b688059a-w-0 systemd[1]: Finished Google Compute Engine Startup Scripts.
# Oct 15 15:11:29 t1v-n-b688059a-w-0 systemd[1]: google-startup-scripts.service: Consumed 1min 38.976s CPU time.
# Oct 15 15:10:56 t1v-n-b688059a-w-2 systemd[1]: Finished Google Compute Engine Startup Scripts.
# Oct 15 15:10:56 t1v-n-b688059a-w-2 systemd[1]: google-startup-scripts.service: Consumed 1min 39.757s CPU time.
# Oct 15 15:10:42 t1v-n-b688059a-w-1 systemd[1]: Finished Google Compute Engine Startup Scripts.
# Oct 15 15:10:42 t1v-n-b688059a-w-1 systemd[1]: google-startup-scripts.service: Consumed 1min 40.667s CPU time.
# Oct 15 15:10:27 t1v-n-b688059a-w-3 systemd[1]: Finished Google Compute Engine Startup Scripts.
# Oct 15 15:10:27 t1v-n-b688059a-w-3 systemd[1]: google-startup-scripts.service: Consumed 1min 25.137s CPU time.
```
Make sure above logs does not show any errors like `startup-script exit status 100`.


Once all done, we should see `flax`, `logs` and `arrayrecord_datasets` folders:
```bash
gcloud compute tpus tpu-vm ssh --worker=all --command="ls /root/" $TPU_NAME

# arrayrecord_datasets
# flax
# logs
# snap
# ... 4 times ...
```

Check python version and available TPUs:
```bash
gcloud compute tpus tpu-vm ssh --worker=all --command="python -VV" $TPU_NAME

# Python 3.12.12 (main, Oct 14 2025, 21:25:31) [Clang 20.1.4 ]
# Python 3.12.12 (main, Oct 14 2025, 21:25:31) [Clang 20.1.4 ]
# Python 3.12.12 (main, Oct 14 2025, 21:25:31) [Clang 20.1.4 ]
# Python 3.12.12 (main, Oct 14 2025, 21:25:31) [Clang 20.1.4 ]

gcloud compute tpus tpu-vm ssh --worker=all --command="python -c 'import jax; print(f\"{jax.process_index()=}, num devices={len(jax.devices())}\")'" $TPU_NAME

# jax.process_index()=0, num devices=16
# jax.process_index()=3, num devices=16
# jax.process_index()=2, num devices=16
# jax.process_index()=1, num devices=16
```

##### Training

Let's assume that we have locally the training code. We can copy the code from the current folder to TPU VMs:
```bash
gcloud compute tpus tpu-vm scp --recurse . $TPU_NAME:/root/gemma-example --worker=all
```

Let's create the output folder using worker 0 only:
```bash
export out_dir=/root/logs/gemma3-1b_lm1b_run-$ACCELERATOR_TYPE

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="export out_dir=$out_dir && mkdir -p \$out_dir"
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="ls /root/logs"
```

Let's set up the training command:
```bash
# env vars:
setup_command="export TFDS_DATA_DIR=/root/arrayrecord_datasets && export out_dir=$out_dir"

# get current host id for logs files:
get_proc_id_command="export proc_id=\`python -c \"import jax; print(jax.process_index())\"\` && echo \"proc_id=\$proc_id\""

# python command to run:
command="cd /root/gemma-example && python -u main.py --workdir=\$out_dir --config=configs/default.py &> \$out_dir/output.w\$proc_id.log"

# full command with tmux:
full_command="tmux new -d -s gemma '$setup_command && $get_proc_id_command && $command'"

gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --worker=all \
  --command="$full_command"
```

We can check whether python processes are running:
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="ps -aux | grep -E 'python -u main.py'"
```

We can also check the logs files:
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="cat $out_dir/output.*.log"
```

If we need to stop the python processes:
```bash
# gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="pkill python"
```


##### Clean-up

Finally, once we are done and TPU VMs are unused, let's delete them:
```bash
yes | gcloud compute tpus tpu-vm delete $TPU_NAME --async
```
