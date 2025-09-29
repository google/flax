
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

cd /root
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
python -m pip install -e .

# Install gcsfuse
# Avoid errors like: Could not get lock /var/lib/dpkg/lock-frontend. It is held by process 8196 (unattended-upgr)
pkill -9 -f unattended-upgrade

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
Make sure above logs do not show any errors like `startup-script exit status 100`.


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
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="cat $out_dir/output.w0.log"
```
<details>
<summary>
Example output
</summary>

```
I1110 16:26:42.416053 139892847893632 main.py:47] JAX version: 0.8.0
I1110 16:26:42.436207 139892847893632 main.py:48] Flax version: 0.12.0
I1110 16:26:46.382005 139892847893632 main.py:49] JAX process: 0 / 4
I1110 16:26:46.382269 139892847893632 main.py:50] JAX local devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0)]
I1110 16:26:46.382400 139892847893632 local.py:45] Setting task status: process_index: 0, process_count: 4
I1110 16:26:46.382778 139892847893632 local.py:50] Created artifact workdir of type ArtifactType.DIRECTORY and value /root/logs/gemma3-1b_lm1b_run-v4-32-8.
I1110 16:26:46.404435 139892847893632 utils.py:55] Devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,0,1), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,0,1), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,1,1), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,1,1), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,2), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,2), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,2), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,2), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,0,3), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,0,3), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,1,3), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,1,3), core_on_chip=0)]
I1110 16:26:46.404541 139892847893632 utils.py:56] Number of devices: 16
I1110 16:26:46.405172 139892847893632 utils.py:86] Decided on mesh: [[[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
   TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0)
   TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0)
   TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0)]
  [TpuDevice(id=4, process_index=1, coords=(0,0,1), core_on_chip=0)
   TpuDevice(id=6, process_index=1, coords=(0,1,1), core_on_chip=0)
   TpuDevice(id=5, process_index=1, coords=(1,0,1), core_on_chip=0)
   TpuDevice(id=7, process_index=1, coords=(1,1,1), core_on_chip=0)]
  [TpuDevice(id=8, process_index=2, coords=(0,0,2), core_on_chip=0)
   TpuDevice(id=10, process_index=2, coords=(0,1,2), core_on_chip=0)
   TpuDevice(id=9, process_index=2, coords=(1,0,2), core_on_chip=0)
   TpuDevice(id=11, process_index=2, coords=(1,1,2), core_on_chip=0)]
  [TpuDevice(id=12, process_index=3, coords=(0,0,3), core_on_chip=0)
   TpuDevice(id=14, process_index=3, coords=(0,1,3), core_on_chip=0)
   TpuDevice(id=13, process_index=3, coords=(1,0,3), core_on_chip=0)
   TpuDevice(id=15, process_index=3, coords=(1,1,3), core_on_chip=0)]]]
I1110 16:26:46.405273 139892847893632 utils.py:87] Mesh shape: (1, 4, 4)
I1110 16:26:46.405409 139892847893632 train.py:410] Initializing dataset.
I1110 16:26:46.532675 139892847893632 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I1110 16:26:46.535846 139892847893632 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I1110 16:26:46.543797 139892847893632 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I1110 16:26:46.546356 139892847893632 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I1110 16:26:49.007439 139892847893632 train.py:420] Initializing model, optimizer, and step functions.
I1110 16:26:57.990381 139808585451072 logging_writer.py:80] [Hyperparameters] {'vocab_path': '/root/logs/gemma3-1b_lm1b_run-v4-32-8/sentencepiece_model', 'vocab_size': 35008, 'max_corpus_chars': 10000000, 'dataset_name': 'lm1b', 'eval_dataset_name': 'lm1b', 'eval_split': 'test', 'per_device_batch_size': 32, 'eval_per_device_batch_size': 32, 'prefetch_num_workers': None, 'prompts': ('Paris is a the capital', 'Flax is a', 'The shutdown was aimed at creating efficiencies as', 'A big theme of this hire is that there are parts of', 'Because of Bear Stearns , many analysts are', 'Next month , the Brazilian bourse'), 'sampling_temperature': 0.0, 'sampling_top_p': 0.95, 'num_train_steps': 500000, 'num_eval_steps': 100, 'num_predict_steps': 50, 'learning_rate': 0.0016, 'warmup_steps': 1000, 'label_smoothing': 0.0, 'weight_decay': 0.1, 'max_target_length': 128, 'max_eval_target_length': 512, 'transformer_name': 'gemma3_1b', 'transformer_params': None, 'save_checkpoints': True, 'restore_checkpoints': True, 'checkpoint_every_steps': 100, 'eval_every_steps': 150, 'use_bfloat16': True, 'seed': 0, 'mesh_axes': ('data', 'fsdp', 'tensor'), 'data_sharding': ('data', 'fsdp'), 'dcn_data_parallelism': -1, 'dcn_fsdp_parallelism': 1, 'dcn_tensor_parallelism': 1, 'ici_data_parallelism': 1, 'ici_fsdp_parallelism': -1, 'ici_tensor_parallelism': 4}
I1110 16:26:58.028725 139892847893632 train.py:510] Starting training loop.
I1110 16:26:58.130373 139807526266432 grain_pool.py:367] Grain pool will use 120 processes.
I1110 16:26:58.309086 139807526266432 grain_pool.py:440] Grain pool will start child processes.
I1110 16:26:58.626513 139807526266432 grain_pool.py:448] Grain pool started all child processes.
I1110 16:29:22.349542 139892847893632 train.py:545] Finished training step 0. Batch size: 512, Loss: 10.51445, LR: 0.00000
I1110 16:31:15.423126 139892847893632 train.py:545] Finished training step 1. Batch size: 512, Loss: 10.51927, LR: 0.00000
I1110 16:31:20.588910 139892847893632 local.py:41] Setting work unit notes: 0.0 steps/s, 0.0% (1/500000), ETA: 660d5h42m (4m : 5.4% data, 94.5% train_step)
I1110 16:31:20.589719 139808585451072 logging_writer.py:48] [1] steps_per_sec=0.00876507
I1110 16:31:20.590493 139808585451072 logging_writer.py:48] [1] uptime=262.56
I1110 16:31:20.633740 139892847893632 train.py:545] Finished training step 2. Batch size: 512, Loss: 10.43962, LR: 0.00000
I1110 16:31:22.343976 139892847893632 train.py:545] Finished training step 3. Batch size: 512, Loss: 10.30984, LR: 0.00001
I1110 16:31:24.015815 139892847893632 train.py:545] Finished training step 4. Batch size: 512, Loss: 10.17181, LR: 0.00001
I1110 16:31:25.698209 139892847893632 train.py:545] Finished training step 5. Batch size: 512, Loss: 10.03922, LR: 0.00001
I1110 16:31:27.375142 139892847893632 train.py:545] Finished training step 6. Batch size: 512, Loss: 9.89501, LR: 0.00001
I1110 16:31:29.051768 139892847893632 train.py:545] Finished training step 7. Batch size: 512, Loss: 9.75061, LR: 0.00001
I1110 16:31:30.730251 139892847893632 train.py:545] Finished training step 8. Batch size: 512, Loss: 9.62513, LR: 0.00001
I1110 16:31:32.766706 139892847893632 train.py:545] Finished training step 9. Batch size: 512, Loss: 9.52952, LR: 0.00002
I1110 16:31:34.443128 139892847893632 train.py:545] Finished training step 10. Batch size: 512, Loss: 9.44326, LR: 0.00002
I1110 16:31:36.234637 139892847893632 train.py:545] Finished training step 11. Batch size: 512, Loss: 9.38389, LR: 0.00002
I1110 16:31:37.922191 139892847893632 train.py:545] Finished training step 12. Batch size: 512, Loss: 9.31117, LR: 0.00002
I1110 16:31:39.610730 139892847893632 train.py:545] Finished training step 13. Batch size: 512, Loss: 9.22744, LR: 0.00002
I1110 16:31:41.298381 139892847893632 train.py:545] Finished training step 14. Batch size: 512, Loss: 9.15212, LR: 0.00002
I1110 16:31:42.992159 139892847893632 train.py:545] Finished training step 15. Batch size: 512, Loss: 9.05383, LR: 0.00003
I1110 16:32:11.217436 139892847893632 local.py:50] Created artifact [10] Profile of type ArtifactType.URL and value None.
I1110 16:32:11.258966 139892847893632 train.py:545] Finished training step 16. Batch size: 512, Loss: 8.98678, LR: 0.00003
I1110 16:32:12.933948 139892847893632 train.py:545] Finished training step 17. Batch size: 512, Loss: 8.90182, LR: 0.00003
I1110 16:32:14.613593 139892847893632 train.py:545] Finished training step 18. Batch size: 512, Loss: 8.85543, LR: 0.00003
I1110 16:32:16.291847 139892847893632 train.py:545] Finished training step 19. Batch size: 512, Loss: 8.78520, LR: 0.00003
I1110 16:32:21.278177 139892847893632 local.py:41] Setting work unit notes: 0.4 steps/s, 0.0% (24/500000), ETA: 15d6h27m (5m : 4.4% data, 87.0% train_step)
I1110 16:32:21.279012 139808585451072 logging_writer.py:48] [24] steps_per_sec=0.37898
I1110 16:32:21.280410 139808585451072 logging_writer.py:48] [24] uptime=323.25
I1110 16:33:22.810923 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.0% (61/500000), ETA: 9d14h57m (6m : 3.7% data, 89.1% train_step)
I1110 16:33:22.811652 139808585451072 logging_writer.py:48] [61] steps_per_sec=0.601305
I1110 16:33:22.812520 139808585451072 logging_writer.py:48] [61] uptime=384.782
I1110 16:34:24.343969 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.0% (98/500000), ETA: 9d14h56m (7m : 3.2% data, 90.6% train_step)
I1110 16:34:24.344751 139808585451072 logging_writer.py:48] [98] steps_per_sec=0.601303
I1110 16:34:24.345644 139808585451072 logging_writer.py:48] [98] uptime=446.315
I1110 16:35:25.877253 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.0% (135/500000), ETA: 9d14h55m (8m : 2.8% data, 91.7% train_step)
I1110 16:35:25.877956 139808585451072 logging_writer.py:48] [135] steps_per_sec=0.601301
I1110 16:35:25.878781 139808585451072 logging_writer.py:48] [135] uptime=507.849
I1110 16:35:50.823009 139892847893632 train.py:560] Gathering training metrics.
I1110 16:35:55.822068 139808585451072 logging_writer.py:48] [150] train_accuracy=0.17894981801509857, train_loss=6.644538879394531, train_perplexity=768.5744018554688
I1110 16:36:24.860724 139808585451072 logging_writer.py:64] [150] Got texts: {'samples': ['Paris is a the capital of the first time of these people have been held in the first time .', 'Flax is a very good from the first time of these people , and these are not expected to be a very good .', 'The shutdown was aimed at creating efficiencies as a $ 1 billion in the first time .', 'A big theme of this hire is that there are parts of the first time .', 'Because of Bear Stearns , many analysts are also expected to be a very good , and these are also expected to be a very good .', 'Next month , the Brazilian bourse of the $ 1 billion in the first time , while the $ 1 billion of the $ 1 billion in the first time .']}.
I1110 16:36:25.014281 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 16:36:25.055885 139803636819520 grain_pool.py:367] Grain pool will use 120 processes.
I1110 16:36:25.224769 139803636819520 grain_pool.py:440] Grain pool will start child processes.
I1110 16:36:25.883547 139803636819520 grain_pool.py:448] Grain pool started all child processes.
I1110 16:39:55.238663 139803636819520 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:39:59.840258 139803636819520 grain_pool.py:542] Grain pool is exiting.
I1110 16:39:59.904247 139803636819520 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:39:59.905262 139803636819520 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:40:24.727105 139892847893632 local.py:41] Setting work unit notes: 0.1 steps/s, 0.0% (151/500000), ETA: 108d1h23m (13m : 1.8% data, 3.6% generate_text, 61.5% train_step, 0.0% training_metrics)
I1110 16:41:11.543605 139808585451072 logging_writer.py:48] [150] eval_accuracy=0.26453155279159546, eval_loss=4.96261739730835, eval_perplexity=142.96762084960938
I1110 16:41:11.544275 139808585451072 logging_writer.py:48] [151] steps_per_sec=0.0535386
I1110 16:41:11.544603 139808585451072 logging_writer.py:48] [151] uptime=806.699
I1110 16:41:24.846365 139892847893632 local.py:41] Setting work unit notes: 0.2 steps/s, 0.0% (161/500000), ETA: 34d18h43m (14m : 1.7% data, 33.1% eval, 3.4% generate_text, 58.7% train_step, 0.0% training_metrics)
I1110 16:41:24.847136 139808585451072 logging_writer.py:48] [161] steps_per_sec=0.166336
I1110 16:41:24.847942 139808585451072 logging_writer.py:48] [161] uptime=866.818
I1110 16:42:26.379191 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.0% (198/500000), ETA: 9d14h53m (15m : 1.5% data, 30.9% eval, 3.1% generate_text, 61.5% train_step, 0.0% training_metrics)
I1110 16:42:26.379944 139808585451072 logging_writer.py:48] [198] steps_per_sec=0.601305
I1110 16:42:26.380872 139808585451072 logging_writer.py:48] [198] uptime=928.351
I1110 16:43:27.911852 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.0% (235/500000), ETA: 9d14h52m (16m : 1.4% data, 28.9% eval, 2.9% generate_text, 63.9% train_step, 0.0% training_metrics)
I1110 16:43:27.912540 139808585451072 logging_writer.py:48] [235] steps_per_sec=0.601307
I1110 16:43:27.913388 139808585451072 logging_writer.py:48] [235] uptime=989.883
I1110 16:44:29.444911 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (272/500000), ETA: 9d14h51m (17m : 1.4% data, 27.3% eval, 2.8% generate_text, 66.0% train_step, 0.0% training_metrics)
I1110 16:44:29.448316 139808585451072 logging_writer.py:48] [272] steps_per_sec=0.601303
I1110 16:44:29.449286 139808585451072 logging_writer.py:48] [272] uptime=1051.42
I1110 16:45:16.011366 139892847893632 train.py:560] Gathering training metrics.
I1110 16:45:21.003551 139808585451072 logging_writer.py:48] [300] train_accuracy=0.32114270329475403, train_loss=4.42142391204834, train_perplexity=83.21463775634766
I1110 16:45:22.012165 139808585451072 logging_writer.py:64] [300] Got texts: {'samples': ["Paris is a the capital , and the world 's largest oil company has been held by the U.S. government .", "Flax is a very good player , but it 's a very good thing .", 'The shutdown was aimed at creating efficiencies as well as a $ 1 billion loan .', 'A big theme of this hire is that there are parts of the world , and these are the most important things that are the most important things that are going to be .', 'Because of Bear Stearns , many analysts are still trying to make these new rules .', 'Next month , the Brazilian bourse of the game was held by the U.S. Open , which was held by the U.S. Open .']}.
I1110 16:45:22.024208 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 16:45:22.172550 139803736536640 grain_pool.py:367] Grain pool will use 120 processes.
I1110 16:45:22.298648 139803736536640 grain_pool.py:440] Grain pool will start child processes.
I1110 16:45:22.966312 139803736536640 grain_pool.py:448] Grain pool started all child processes.
I1110 16:48:28.193296 139803736536640 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:48:32.688385 139803736536640 grain_pool.py:542] Grain pool is exiting.
I1110 16:48:32.750683 139803736536640 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:48:32.751191 139803736536640 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:48:57.685840 139892847893632 local.py:41] Setting work unit notes: 0.1 steps/s, 0.1% (301/500000), ETA: 53d11h54m (21m : 1.1% data, 21.7% eval, 2.3% generate_text, 56.5% train_step, 0.0% training_metrics)
I1110 16:49:44.502426 139808585451072 logging_writer.py:48] [300] eval_accuracy=0.3245704174041748, eval_loss=4.122683525085449, eval_perplexity=61.724571228027344
I1110 16:49:44.503173 139808585451072 logging_writer.py:48] [301] steps_per_sec=0.108112
I1110 16:49:44.503610 139808585451072 logging_writer.py:48] [301] uptime=1319.66
I1110 16:49:57.805528 139892847893632 local.py:41] Setting work unit notes: 0.2 steps/s, 0.1% (311/500000), ETA: 34d18h28m (22m : 1.0% data, 39.8% eval, 2.2% generate_text, 55.0% train_step, 0.0% training_metrics)
I1110 16:49:57.806386 139808585451072 logging_writer.py:48] [311] steps_per_sec=0.166335
I1110 16:49:57.807326 139808585451072 logging_writer.py:48] [311] uptime=1379.78
I1110 16:50:59.338149 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (348/500000), ETA: 9d14h49m (24m : 1.0% data, 38.1% eval, 2.1% generate_text, 56.9% train_step, 0.0% training_metrics)
I1110 16:50:59.338990 139808585451072 logging_writer.py:48] [348] steps_per_sec=0.601307
I1110 16:50:59.339883 139808585451072 logging_writer.py:48] [348] uptime=1441.31
I1110 16:52:00.871247 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (385/500000), ETA: 9d14h48m (25m : 1.0% data, 36.5% eval, 2.0% generate_text, 58.6% train_step, 0.0% training_metrics)
I1110 16:52:00.872014 139808585451072 logging_writer.py:48] [385] steps_per_sec=0.601302
I1110 16:52:00.872804 139808585451072 logging_writer.py:48] [385] uptime=1502.84
I1110 16:53:02.403934 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (422/500000), ETA: 9d14h47m (26m : 0.9% data, 35.1% eval, 1.9% generate_text, 60.3% train_step, 0.0% training_metrics)
I1110 16:53:02.405987 139808585451072 logging_writer.py:48] [422] steps_per_sec=0.601307
I1110 16:53:02.406986 139808585451072 logging_writer.py:48] [422] uptime=1564.38
I1110 16:53:48.969695 139892847893632 train.py:560] Gathering training metrics.
I1110 16:53:53.963266 139808585451072 logging_writer.py:48] [450] train_accuracy=0.3538951575756073, train_loss=3.9279839992523193, train_perplexity=50.80431365966797
I1110 16:53:55.494154 139808585451072 logging_writer.py:64] [450] Got texts: {'samples': ['Paris is a the capital of the country , but it is not the case .', 'Flax is a leading provider of communications and services , and is a leading provider of communications and services .', "The shutdown was aimed at creating efficiencies as the government announced it would take a $ 1 billion stimulus package to help the government to raise the cost of the government 's $ 700 billion budget deficit .", 'A big theme of this hire is that there are parts of the world that are not in the world .', 'Because of Bear Stearns , many analysts are still looking to buy the banks , which are not the only way to get the banks to buy the banks .', 'Next month , the Brazilian bourse said it would cut its share of the shares by 20 percent to $ 1 billion .']}.
I1110 16:53:55.506319 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 16:53:55.689530 139803753322048 grain_pool.py:367] Grain pool will use 120 processes.
I1110 16:53:55.812076 139803753322048 grain_pool.py:440] Grain pool will start child processes.
I1110 16:53:56.542339 139803753322048 grain_pool.py:448] Grain pool started all child processes.
I1110 16:56:56.015909 139803753322048 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:57:00.150132 139803753322048 grain_pool.py:542] Grain pool is exiting.
I1110 16:57:00.208169 139803753322048 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:57:00.208819 139803753322048 grain_pool.py:547] Shutting down multiprocessing system.
I1110 16:57:25.508115 139892847893632 local.py:41] Setting work unit notes: 0.1 steps/s, 0.1% (451/500000), ETA: 52d10h56m (30m : 0.8% data, 30.0% eval, 1.7% generate_text, 54.4% train_step, 0.0% training_metrics)
I1110 16:58:12.324599 139808585451072 logging_writer.py:48] [450] eval_accuracy=0.34616002440452576, eval_loss=3.822641134262085, eval_perplexity=45.72483444213867
I1110 16:58:12.325167 139808585451072 logging_writer.py:48] [451] steps_per_sec=0.110222
I1110 16:58:12.325560 139808585451072 logging_writer.py:48] [451] uptime=1827.48
I1110 16:58:25.627465 139892847893632 local.py:41] Setting work unit notes: 0.2 steps/s, 0.1% (461/500000), ETA: 34d18h13m (31m : 0.8% data, 42.7% eval, 1.7% generate_text, 53.4% train_step, 0.0% training_metrics)
I1110 16:58:25.628161 139808585451072 logging_writer.py:48] [461] steps_per_sec=0.166336
I1110 16:58:25.629033 139808585451072 logging_writer.py:48] [461] uptime=1887.6
I1110 16:59:27.160371 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (498/500000), ETA: 9d14h44m (32m : 0.7% data, 41.3% eval, 1.6% generate_text, 54.9% train_step, 0.0% training_metrics)
I1110 16:59:27.161170 139808585451072 logging_writer.py:48] [498] steps_per_sec=0.601304
I1110 16:59:27.162021 139808585451072 logging_writer.py:48] [498] uptime=1949.13
I1110 17:00:28.693432 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (535/500000), ETA: 9d14h43m (33m : 0.7% data, 40.1% eval, 1.6% generate_text, 56.2% train_step, 0.0% training_metrics)
I1110 17:00:28.694186 139808585451072 logging_writer.py:48] [535] steps_per_sec=0.601303
I1110 17:00:28.695176 139808585451072 logging_writer.py:48] [535] uptime=2010.66
I1110 17:01:30.226571 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (572/500000), ETA: 9d14h42m (34m : 0.7% data, 38.9% eval, 1.5% generate_text, 57.5% train_step, 0.0% training_metrics)
I1110 17:01:30.227326 139808585451072 logging_writer.py:48] [572] steps_per_sec=0.601302
I1110 17:01:30.228344 139808585451072 logging_writer.py:48] [572] uptime=2072.2
I1110 17:02:16.792247 139892847893632 train.py:560] Gathering training metrics.
I1110 17:02:21.785352 139808585451072 logging_writer.py:48] [600] train_accuracy=0.3674505650997162, train_loss=3.7325258255004883, train_perplexity=41.784427642822266
I1110 17:02:22.978659 139808585451072 logging_writer.py:64] [600] Got texts: {'samples': ["Paris is a the capital of the capital , with the country 's largest economy expected to grow by a third of the economy .", 'Flax is a former New York City police officer whose job is to be held in the city .', 'The shutdown was aimed at creating efficiencies as the economy improved .', "A big theme of this hire is that there are parts of the public , such as the ones in which the company 's stocks are trading .", 'Because of Bear Stearns , many analysts are optimistic that the economy will be in a recession .', "Next month , the Brazilian bourse the world 's leading players , with the United States , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany , Germany ,"]}.
I1110 17:02:22.991259 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 17:02:23.264263 139803778500160 grain_pool.py:367] Grain pool will use 120 processes.
I1110 17:02:23.394217 139803778500160 grain_pool.py:440] Grain pool will start child processes.
I1110 17:02:24.073271 139803778500160 grain_pool.py:448] Grain pool started all child processes.
I1110 17:05:26.524962 139803778500160 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:05:31.244337 139803778500160 grain_pool.py:542] Grain pool is exiting.
I1110 17:05:31.312958 139803778500160 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:05:31.313490 139803778500160 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:05:56.018352 139892847893632 local.py:41] Setting work unit notes: 0.1 steps/s, 0.1% (601/500000), ETA: 52d23h25m (38m : 0.6% data, 34.5% eval, 1.4% generate_text, 53.2% train_step, 0.0% training_metrics)
I1110 17:06:42.835155 139808585451072 logging_writer.py:48] [600] eval_accuracy=0.3553972542285919, eval_loss=3.697735548019409, eval_perplexity=40.35570526123047
I1110 17:06:42.835707 139808585451072 logging_writer.py:48] [601] steps_per_sec=0.109108
I1110 17:06:42.835844 139808585451072 logging_writer.py:48] [601] uptime=2337.99
I1110 17:06:56.137494 139892847893632 local.py:41] Setting work unit notes: 0.2 steps/s, 0.1% (611/500000), ETA: 34d17h58m (39m : 0.6% data, 44.4% eval, 1.4% generate_text, 52.4% train_step, 0.0% training_metrics)
I1110 17:06:56.138241 139808585451072 logging_writer.py:48] [611] steps_per_sec=0.166336
I1110 17:06:56.139184 139808585451072 logging_writer.py:48] [611] uptime=2398.11
I1110 17:07:57.670742 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (648/500000), ETA: 9d14h40m (40m : 0.6% data, 43.3% eval, 1.3% generate_text, 53.6% train_step, 0.0% training_metrics)
I1110 17:07:57.671507 139808585451072 logging_writer.py:48] [648] steps_per_sec=0.601301
I1110 17:07:57.672353 139808585451072 logging_writer.py:48] [648] uptime=2459.64
I1110 17:08:59.204406 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (685/500000), ETA: 9d14h39m (42m : 0.6% data, 42.3% eval, 1.3% generate_text, 54.7% train_step, 0.0% training_metrics)
I1110 17:08:59.205625 139808585451072 logging_writer.py:48] [685] steps_per_sec=0.601297
I1110 17:08:59.206557 139808585451072 logging_writer.py:48] [685] uptime=2521.18
I1110 17:10:00.736587 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.1% (722/500000), ETA: 9d14h38m (43m : 0.6% data, 41.3% eval, 1.3% generate_text, 55.8% train_step, 0.0% training_metrics)
I1110 17:10:00.737588 139808585451072 logging_writer.py:48] [722] steps_per_sec=0.601311
I1110 17:10:00.739156 139808585451072 logging_writer.py:48] [722] uptime=2582.71
I1110 17:10:47.301956 139892847893632 train.py:560] Gathering training metrics.
I1110 17:10:52.295369 139808585451072 logging_writer.py:48] [750] train_accuracy=0.37607187032699585, train_loss=3.6280879974365234, train_perplexity=37.640750885009766
I1110 17:10:53.479609 139808585451072 logging_writer.py:64] [750] Got texts: {'samples': ["Paris is a the capital of the United States , and the world 's largest and most expensive city .", 'Flax is a leading online retailer with a number of brands including Apple , Apple , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft , Microsoft ,', 'The shutdown was aimed at creating efficiencies as the government is expected to announce a $ 100 billion stimulus package .', 'A big theme of this hire is that there are parts of the world , including the world , and the world .', 'Because of Bear Stearns , many analysts are optimistic that the economy will be better off with a better-than-expected recovery .', 'Next month , the Brazilian bourse will be able to secure a $ 100 million contract for the club , which is expected to be announced later this year .']}.
I1110 17:10:53.491786 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 17:10:53.679328 139803719751232 grain_pool.py:367] Grain pool will use 120 processes.
I1110 17:10:53.809175 139803719751232 grain_pool.py:440] Grain pool will start child processes.
I1110 17:10:54.447105 139803719751232 grain_pool.py:448] Grain pool started all child processes.
I1110 17:13:53.780019 139803719751232 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:13:58.256169 139803719751232 grain_pool.py:542] Grain pool is exiting.
I1110 17:13:58.320006 139803719751232 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:13:58.320730 139803719751232 grain_pool.py:547] Shutting down multiprocessing system.
I1110 17:14:23.273147 139892847893632 local.py:41] Setting work unit notes: 0.1 steps/s, 0.2% (751/500000), ETA: 52d7h28m (47m : 0.5% data, 37.5% eval, 1.2% generate_text, 52.5% train_step, 0.0% training_metrics)
I1110 17:15:10.089522 139808585451072 logging_writer.py:48] [750] eval_accuracy=0.3620672821998596, eval_loss=3.6101763248443604, eval_perplexity=36.97251892089844
I1110 17:15:10.090173 139808585451072 logging_writer.py:48] [751] steps_per_sec=0.110461
I1110 17:15:10.090519 139808585451072 logging_writer.py:48] [751] uptime=2845.24
I1110 17:15:23.392529 139892847893632 local.py:41] Setting work unit notes: 0.2 steps/s, 0.2% (761/500000), ETA: 34d17h43m (48m : 0.5% data, 45.5% eval, 1.2% generate_text, 51.8% train_step, 0.0% training_metrics)
I1110 17:15:23.393419 139808585451072 logging_writer.py:48] [761] steps_per_sec=0.166336
I1110 17:15:23.394225 139808585451072 logging_writer.py:48] [761] uptime=2905.36
I1110 17:16:24.925530 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.2% (798/500000), ETA: 9d14h36m (49m : 0.5% data, 44.6% eval, 1.1% generate_text, 52.8% train_step, 0.0% training_metrics)
I1110 17:16:24.926407 139808585451072 logging_writer.py:48] [798] steps_per_sec=0.601303
I1110 17:16:24.927347 139808585451072 logging_writer.py:48] [798] uptime=2966.9
I1110 17:17:26.458402 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.2% (835/500000), ETA: 9d14h35m (50m : 0.5% data, 43.7% eval, 1.1% generate_text, 53.8% train_step, 0.0% training_metrics)
I1110 17:17:26.459366 139808585451072 logging_writer.py:48] [835] steps_per_sec=0.601305
I1110 17:17:26.460280 139808585451072 logging_writer.py:48] [835] uptime=3028.43
I1110 17:18:27.991161 139892847893632 local.py:41] Setting work unit notes: 0.6 steps/s, 0.2% (872/500000), ETA: 9d14h34m (51m : 0.5% data, 42.8% eval, 1.1% generate_text, 54.7% train_step, 0.0% training_metrics)
I1110 17:18:27.991982 139808585451072 logging_writer.py:48] [872] steps_per_sec=0.601306
I1110 17:18:27.992802 139808585451072 logging_writer.py:48] [872] uptime=3089.96
I1110 17:19:14.556560 139892847893632 train.py:560] Gathering training metrics.
I1110 17:19:19.550159 139808585451072 logging_writer.py:48] [900] train_accuracy=0.3829336166381836, train_loss=3.5548300743103027, train_perplexity=34.98186492919922
I1110 17:19:21.242841 139808585451072 logging_writer.py:64] [900] Got texts: {'samples': ['Paris is a the capital of the world .', "Flax is a very good player , and it 's a great team .", 'The shutdown was aimed at creating efficiencies as a result of the economic downturn .', 'A big theme of this hire is that there are parts of the world that are not always always always the best .', 'Because of Bear Stearns , many analysts are worried about the future of the company .', "Next month , the Brazilian bourse of the world 's largest mobile phone maker , the Chinese electronics company , announced a $ 1 billion ( $ 2 billion ) rescue of the world 's largest mobile phone maker , the world 's largest mobile phone maker ."]}.
I1110 17:19:21.255329 139892847893632 train.py:370] Gathering evaluation metrics.
I1110 17:19:21.454909 139803769058880 grain_pool.py:367] Grain pool will use 120 processes.
I1110 17:19:21.583253 139803769058880 grain_pool.py:440] Grain pool will start child processes.
I1110 17:19:22.221782 139803769058880 grain_pool.py:448] Grain pool started all child processes.
I1110 17:22:28.556169 139803769058880 grain_pool.py:547] Shutting down multiprocessing system.

```

</details>

To see TPUs usage, check the [Metrics Explorer](https://console.cloud.google.com/monitoring/metrics-explorer), select `tpu.googleapis.com/accelerator/memory_used` and `tpu.googleapis.com/accelerator/duty_cycle`.


If we need to stop the python processes:
```bash
# gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="pkill -9 -f python && tmux kill-session -t gemma"
```


##### Clean-up

Finally, once we are done and TPU VMs are unused, let's delete them:
```bash
yes | gcloud compute tpus tpu-vm delete $TPU_NAME --async
```
