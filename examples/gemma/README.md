
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
export GCS_OUTPUT_BUCKET=flax-gemma-example-training
export GCS_DATA_BUCKET=flax-lm1b-arrayrecords

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
  $GCS_DATA_BUCKET /root/arrayrecord_datasets

mkdir -p /root/logs
gcsfuse $GCS_OUTPUT_BUCKET /root/logs

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
export out_name=gemma3-1b_lm1b_run-$ACCELERATOR_TYPE
export out_dir=/root/logs/$out_name
export chpt_bucket=gs://$GCS_OUTPUT_BUCKET/$out_name/checkpoint

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
command="cd /root/gemma-example && python -u main.py --workdir=\$out_dir --chpt_bucket=$chpt_bucket --config=configs/default.py &> \$out_dir/output.w\$proc_id.log"

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
I1127 10:16:57.510740 140023579745408 main.py:51] JAX version: 0.8.1
I1127 10:16:57.532955 140023579745408 main.py:52] Flax version: 0.12.1
INFO:2025-11-27 10:16:57,547:jax._src.distributed:157: Connecting to JAX distributed service on 10.130.0.11:8482
I1127 10:16:57.547805 140023579745408 distributed.py:157] Connecting to JAX distributed service on 10.130.0.11:8482
I1127 10:17:02.688621 140023579745408 main.py:61] JAX process: 0 / 4
I1127 10:17:02.747381 140023579745408 main.py:62] JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,0,1), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,0,1), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,1,1), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,1,1), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,2), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,2), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,2), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,2), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,0,3), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,0,3), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,1,3), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,1,3), core_on_chip=0)]
I1127 10:17:02.747522 140023579745408 main.py:63] FLAGS:
I1127 10:17:02.747666 140023579745408 main.py:64] - FLAGS.config=TrainConfig(vocab_path=None, vocab_size=35008, max_corpus_chars=10000000, dataset_name='lm1b', eval_dataset_name='lm1b', eval_split='test', per_device_batch_size=32, eval_per_device_batch_size=32, prefetch_num_workers=None, prompts=('Paris is a the capital', 'Flax is a', 'The shutdown was aimed at creating efficiencies as', 'A big theme of this hire is that there are parts of', 'Because of Bear Stearns , many analysts are', 'Next month , the Brazilian bourse'), sampling_temperature=0.0, sampling_top_p=0.95, num_train_steps=500000, num_eval_steps=2000, num_predict_steps=50, learning_rate=0.0016, warmup_steps=1000, label_smoothing=0.0, weight_decay=0.1, max_target_length=128, max_eval_target_length=512, transformer_name='gemma3_1b', transformer_params=None, save_checkpoints=True, restore_checkpoints=True, checkpoint_every_steps=10000, eval_every_steps=2000, use_bfloat16=True, seed=0, mesh_axes=('data', 'fsdp', 'tensor'), data_sharding=('data', 'fsdp'), dcn_data_parallelism=-1, dcn_fsdp_parallelism=1, dcn_tensor_parallelism=1, ici_data_parallelism=1, ici_fsdp_parallelism=-1, ici_tensor_parallelism=4)
I1127 10:17:02.747861 140023579745408 main.py:65] - FLAGS.workdir='/root/logs/gemma3-1b_lm1b_run-v4-32-10'
I1127 10:17:02.748032 140023579745408 main.py:66] - FLAGS.chpt_bucket='gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v4-32-10/checkpoint'
I1127 10:17:02.748138 140023579745408 local.py:45] Setting task status: process_index: 0, process_count: 4
I1127 10:17:02.748230 140023579745408 local.py:50] Created artifact workdir of type ArtifactType.DIRECTORY and value /root/logs/gemma3-1b_lm1b_run-v4-32-10.
I1127 10:17:02.774250 140023579745408 utils.py:55] Devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,0,1), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,0,1), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,1,1), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,1,1), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,2), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,2), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,2), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,2), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,0,3), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,0,3), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,1,3), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,1,3), core_on_chip=0)]
I1127 10:17:02.774368 140023579745408 utils.py:56] Number of devices: 16
I1127 10:17:02.774989 140023579745408 utils.py:86] Decided on mesh: [[[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)
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
I1127 10:17:02.775086 140023579745408 utils.py:87] Mesh shape: (1, 4, 4)
I1127 10:17:02.775220 140023579745408 train.py:409] Initializing dataset.
I1127 10:17:02.901412 140023579745408 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I1127 10:17:02.904461 140023579745408 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I1127 10:17:02.912177 140023579745408 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I1127 10:17:02.914429 140023579745408 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I1127 10:17:04.754870 140023579745408 train.py:419] Initializing model, optimizer, and step functions.
I1127 10:18:32.063298 140023579745408 checkpoint_manager.py:702] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=None, item_handlers=None, handler_registry=None
I1127 10:18:32.189736 140023579745408 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f46fc217620>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f46fc217620>}).
I1127 10:18:32.190546 140023579745408 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.28
I1127 10:18:32.190716 140023579745408 async_checkpointer.py:177] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f46fc1ef420> timeout: 600 secs and primary_host=0 for async checkpoint writes
I1127 10:18:32.514777 140023579745408 checkpoint_manager.py:558] Created directory=gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v4-32-10/checkpoint
I1127 10:18:33.655884 140023579745408 checkpoint_manager.py:1788] Found 0 checkpoint steps in gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v4-32-10/checkpoint
I1127 10:18:33.661173 140023579745408 checkpoint_manager.py:921] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_hns=False, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=<class 'orbax.checkpoint._src.path.atomicity.CommitFileTemporaryPath'>, save_decision_policy=None, preservation_policy=LatestN(n=1), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False), root_directory=gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v4-32-10/checkpoint: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f46fc1f8a70>
I1127 10:18:33.688808 140023579745408 train.py:493]
Model Number of Parameters:
- Total (B): 0.73822528
- Embedding (B): 0.040329216
- Attentions (B): 0.076690432
- MLPs (B): 0.621084672

I1127 10:18:33.970078 139938806449728 logging_writer.py:80] [Hyperparameters] {'vocab_path': '/root/logs/gemma3-1b_lm1b_run-v4-32-10/sentencepiece_model', 'vocab_size': 35008, 'max_corpus_chars': 10000000, 'dataset_name': 'lm1b', 'eval_dataset_name': 'lm1b', 'eval_split': 'test', 'per_device_batch_size': 32, 'eval_per_device_batch_size': 32, 'prefetch_num_workers': None, 'prompts': ('Paris is a the capital', 'Flax is a', 'The shutdown was aimed at creating efficiencies as', 'A big theme of this hire is that there are parts of', 'Because of Bear Stearns , many analysts are', 'Next month , the Brazilian bourse'), 'sampling_temperature': 0.0, 'sampling_top_p': 0.95, 'num_train_steps': 500000, 'num_eval_steps': 2000, 'num_predict_steps': 50, 'learning_rate': 0.0016, 'warmup_steps': 1000, 'label_smoothing': 0.0, 'weight_decay': 0.1, 'max_target_length': 128, 'max_eval_target_length': 512, 'transformer_name': 'gemma3_1b', 'transformer_params': None, 'save_checkpoints': True, 'restore_checkpoints': True, 'checkpoint_every_steps': 10000, 'eval_every_steps': 2000, 'use_bfloat16': True, 'seed': 0, 'mesh_axes': ('data', 'fsdp', 'tensor'), 'data_sharding': ('data', 'fsdp'), 'dcn_data_parallelism': -1, 'dcn_fsdp_parallelism': 1, 'dcn_tensor_parallelism': 1, 'ici_data_parallelism': 1, 'ici_fsdp_parallelism': -1, 'ici_tensor_parallelism': 4}
I1127 10:18:34.037433 140023579745408 train.py:537] Starting training loop.
I1127 10:18:34.152173 139937774147136 grain_pool.py:367] Grain pool will use 120 processes.
I1127 10:18:34.344028 139937774147136 grain_pool.py:440] Grain pool will start child processes.
I1127 10:18:34.642666 139937774147136 grain_pool.py:448] Grain pool started all child processes.
I1127 10:20:52.517470 140023579745408 train.py:570] Finished training step 0. Batch size: 512, Loss: 10.51423, LR: 0.00000
I1127 10:22:20.758177 140023579745408 train.py:570] Finished training step 1. Batch size: 512, Loss: 10.51876, LR: 0.00000
I1127 10:22:26.139804 140023579745408 local.py:41] Setting work unit notes: 0.0 steps/s, 0.0% (1/500000), ETA: 541d10h55m (3m : 8.4% data, 91.4% train_step)
I1127 10:22:26.140852 139938806449728 logging_writer.py:48] [1] steps_per_sec=0.0106879
I1127 10:22:26.141596 139938806449728 logging_writer.py:48] [1] uptime=232.101
I1127 10:22:26.688264 140023579745408 train.py:570] Finished training step 2. Batch size: 512, Loss: 10.43878, LR: 0.00000
I1127 10:22:27.041123 140023579745408 train.py:570] Finished training step 3. Batch size: 512, Loss: 10.30890, LR: 0.00001
I1127 10:22:27.421134 140023579745408 train.py:570] Finished training step 4. Batch size: 512, Loss: 10.17096, LR: 0.00001
I1127 10:22:27.758622 140023579745408 train.py:570] Finished training step 5. Batch size: 512, Loss: 10.04002, LR: 0.00001
I1127 10:22:28.075268 140023579745408 train.py:570] Finished training step 6. Batch size: 512, Loss: 9.89495, LR: 0.00001
I1127 10:22:28.401953 140023579745408 train.py:570] Finished training step 7. Batch size: 512, Loss: 9.75182, LR: 0.00001
I1127 10:22:29.101258 140023579745408 train.py:570] Finished training step 8. Batch size: 512, Loss: 9.62800, LR: 0.00001
I1127 10:22:29.421448 140023579745408 train.py:570] Finished training step 9. Batch size: 512, Loss: 9.53127, LR: 0.00002
I1127 10:22:29.746428 140023579745408 train.py:570] Finished training step 10. Batch size: 512, Loss: 9.44367, LR: 0.00002
I1127 10:22:30.167810 140023579745408 train.py:570] Finished training step 11. Batch size: 512, Loss: 9.38389, LR: 0.00002
I1127 10:22:30.498484 140023579745408 train.py:570] Finished training step 12. Batch size: 512, Loss: 9.31090, LR: 0.00002
I1127 10:22:30.830883 140023579745408 train.py:570] Finished training step 13. Batch size: 512, Loss: 9.22675, LR: 0.00002
I1127 10:22:31.161866 140023579745408 train.py:570] Finished training step 14. Batch size: 512, Loss: 9.15227, LR: 0.00002
I1127 10:22:31.494368 140023579745408 train.py:570] Finished training step 15. Batch size: 512, Loss: 9.05391, LR: 0.00003
I1127 10:22:31.826439 140023579745408 train.py:570] Finished training step 16. Batch size: 512, Loss: 8.98621, LR: 0.00003
I1127 10:22:32.157356 140023579745408 train.py:570] Finished training step 17. Batch size: 512, Loss: 8.90162, LR: 0.00003
I1127 10:22:32.498005 140023579745408 train.py:570] Finished training step 18. Batch size: 512, Loss: 8.85539, LR: 0.00003
I1127 10:22:32.826037 140023579745408 train.py:570] Finished training step 19. Batch size: 512, Loss: 8.78531, LR: 0.00003
I1127 10:23:04.798539 140023579745408 local.py:50] Created artifact [10] Profile of type ArtifactType.URL and value None.
I1127 10:23:26.310736 140023579745408 local.py:41] Setting work unit notes: 1.5 steps/s, 0.0% (91/500000), ETA: 3d20h50m (4m : 6.8% data, 92.5% train_step)
I1127 10:23:26.311457 139938806449728 logging_writer.py:48] [91] steps_per_sec=1.49573
I1127 10:23:26.312351 139938806449728 logging_writer.py:48] [91] uptime=292.272
I1127 10:24:26.534986 140023579745408 local.py:41] Setting work unit notes: 3.3 steps/s, 0.1% (288/500000), ETA: 1d18h26m (5m : 6.0% data, 93.2% train_step)
I1127 10:24:26.535624 139938806449728 logging_writer.py:48] [288] steps_per_sec=3.27111
I1127 10:24:26.536640 139938806449728 logging_writer.py:48] [288] uptime=352.497
I1127 10:25:26.676166 140023579745408 local.py:41] Setting work unit notes: 3.2 steps/s, 0.1% (481/500000), ETA: 1d19h14m (6m : 5.6% data, 93.4% train_step)
I1127 10:25:26.731281 139938806449728 logging_writer.py:48] [481] steps_per_sec=3.20912
I1127 10:25:26.732170 139938806449728 logging_writer.py:48] [481] uptime=412.692
I1127 10:26:26.948779 140023579745408 local.py:41] Setting work unit notes: 3.2 steps/s, 0.1% (676/500000), ETA: 1d18h52m (7m : 5.2% data, 93.7% train_step)
I1127 10:26:27.029459 139938806449728 logging_writer.py:48] [676] steps_per_sec=3.2353
I1127 10:26:27.030550 139938806449728 logging_writer.py:48] [676] uptime=472.99
I1127 10:27:27.063030 140023579745408 local.py:41] Setting work unit notes: 3.2 steps/s, 0.2% (871/500000), ETA: 1d18h44m (8m : 5.0% data, 93.9% train_step)
I1127 10:27:27.063734 139938806449728 logging_writer.py:48] [871] steps_per_sec=3.24382
I1127 10:27:27.064775 139938806449728 logging_writer.py:48] [871] uptime=533.025
I1127 10:28:27.160631 140023579745408 local.py:41] Setting work unit notes: 3.2 steps/s, 0.2% (1065/500000), ETA: 1d18h56m (9m : 4.6% data, 94.2% train_step)
I1127 10:28:27.161365 139938806449728 logging_writer.py:48] [1065] steps_per_sec=3.22808
I1127 10:28:27.162531 139938806449728 logging_writer.py:48] [1065] uptime=593.122
I1127 10:29:27.389725 140023579745408 local.py:41] Setting work unit notes: 3.3 steps/s, 0.3% (1262/500000), ETA: 1d18h21m (10m : 4.4% data, 94.5% train_step)
I1127 10:29:27.390433 139938806449728 logging_writer.py:48] [1262] steps_per_sec=3.27084
I1127 10:29:27.391242 139938806449728 logging_writer.py:48] [1262] uptime=653.351
I1127 10:30:27.586359 140023579745408 local.py:41] Setting work unit notes: 3.3 steps/s, 0.3% (1459/500000), ETA: 1d18h18m (11m : 4.2% data, 94.6% train_step)
I1127 10:30:27.586997 139938806449728 logging_writer.py:48] [1459] steps_per_sec=3.27261
I1127 10:30:27.587883 139938806449728 logging_writer.py:48] [1459] uptime=713.548
I1127 10:31:27.769018 140023579745408 local.py:41] Setting work unit notes: 3.2 steps/s, 0.3% (1650/500000), ETA: 1d19h37m (12m : 4.0% data, 94.6% train_step)
I1127 10:31:27.769735 139938806449728 logging_writer.py:48] [1650] steps_per_sec=3.17367
I1127 10:31:27.770569 139938806449728 logging_writer.py:48] [1650] uptime=773.731
I1127 10:32:27.824230 140023579745408 local.py:41] Setting work unit notes: 3.3 steps/s, 0.4% (1846/500000), ETA: 1d18h23m (13m : 3.9% data, 94.6% train_step)
I1127 10:32:27.824931 139938806449728 logging_writer.py:48] [1846] steps_per_sec=3.26366
I1127 10:32:27.825839 139938806449728 logging_writer.py:48] [1846] uptime=833.786
I1127 10:33:15.066679 140023579745408 train.py:585] Gathering training metrics.
I1127 10:33:54.584734 139938806449728 logging_writer.py:48] [2000] train_accuracy=0.37296992540359497, train_loss=3.7653286457061768, train_perplexity=43.1778678894043
I1127 10:34:20.541454 139938806449728 logging_writer.py:64] [2000] Got texts: {'samples': ["Paris is a the capital of the world , but it is also a major hub for the world 's biggest airline .", 'Flax is a very good thing .', 'The shutdown was aimed at creating efficiencies as well as reducing the number of jobs .', "A big theme of this hire is that there are parts of the city that are not being used to the city 's cultural and cultural heritage .", 'Because of Bear Stearns , many analysts are reluctant to take the risk of a downgrade .', 'Next month , the Brazilian bourse will open at 9 : 00 pm ( 1230 GMT ) , with the benchmark Nikkei 225 stock average rising to a new all-time high of 9,096 .']}.
I1127 10:34:20.610977 140023579745408 train.py:368] Gathering evaluation metrics.
I1127 10:34:20.699224 139933874234944 grain_pool.py:367] Grain pool will use 120 processes.
I1127 10:34:20.860212 139933874234944 grain_pool.py:440] Grain pool will start child processes.
I1127 10:34:21.501270 139933874234944 grain_pool.py:448] Grain pool started all child processes.
I1127 10:37:47.312749 140023579745408 local.py:41] Setting work unit notes: 0.5 steps/s, 0.4% (2001/500000), ETA: 11d21h8m (19m : 2.9% data, 2.3% generate_text, 72.4% train_step, 3.4% training_metrics)
I1127 10:37:51.637447 139938806449728 logging_writer.py:48] [2000] eval_accuracy=0.40467867255210876, eval_loss=3.1962645053863525, eval_perplexity=24.441028594970703
I1127 10:37:51.638035 139938806449728 logging_writer.py:48] [2001] steps_per_sec=0.48515
I1127 10:37:51.638352 139938806449728 logging_writer.py:48] [2001] uptime=1153.27
I1127 10:38:47.558625 140023579745408 local.py:41] Setting work unit notes: 2.9 steps/s, 0.4% (2176/500000), ETA: 1d23h36m (20m : 3.0% data, 17.4% eval, 2.1% generate_text, 73.1% train_step, 3.2% training_metrics)
I1127 10:38:47.559422 139938806449728 logging_writer.py:48] [2176] steps_per_sec=2.90476
I1127 10:38:47.560148 139938806449728 logging_writer.py:48] [2176] uptime=1213.52
I1127 10:39:47.629116 140023579745408 local.py:41] Setting work unit notes: 3.1 steps/s, 0.5% (2362/500000), ETA: 1d20h38m (21m : 2.9% data, 16.6% eval, 2.0% generate_text, 74.2% train_step, 3.0% training_metrics)
I1127 10:39:47.629872 139938806449728 logging_writer.py:48] [2362] steps_per_sec=3.09636
I1127 10:39:47.630731 139938806449728 logging_writer.py:48] [2362] uptime=1273.59
I1127 10:40:48.133224 140023579745408 local.py:41] Setting work unit notes: 3.1 steps/s, 0.5% (2547/500000), ETA: 1d21h11m (22m : 2.8% data, 15.8% eval, 1.9% generate_text, 75.1% train_step, 2.9% training_metrics)
I1127 10:40:48.133990 139938806449728 logging_writer.py:48] [2547] steps_per_sec=3.05764
I1127 10:40:48.134911 139938806449728 logging_writer.py:48] [2547] uptime=1334.09
I1127 10:41:48.265512 140023579745408 local.py:41] Setting work unit notes: 3.1 steps/s, 0.5% (2734/500000), ETA: 1d20h25m (23m : 2.7% data, 15.1% eval, 1.9% generate_text, 76.1% train_step, 2.8% training_metrics)
I1127 10:41:48.266175 139938806449728 logging_writer.py:48] [2734] steps_per_sec=3.10981
I1127 10:41:48.267104 139938806449728 logging_writer.py:48] [2734] uptime=1394.23
I1127 10:42:48.429848 140023579745408 local.py:41] Setting work unit notes: 3.1 steps/s, 0.6% (2922/500000), ETA: 1d20h11m (24m : 2.7% data, 14.5% eval, 1.8% generate_text, 76.9% train_step, 2.7% training_metrics)
I1127 10:42:48.430514 139938806449728 logging_writer.py:48] [2922] steps_per_sec=3.12478
I1127 10:42:48.431390 139938806449728 logging_writer.py:48] [2922] uptime=1454.39
```

</details>


We can also start TensorBoard on one worker to inspect the training:
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="pip install xprof"
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="tensorboard --logdir=$out_dir --port 7007" -- -L 7007:localhost:7007
```


To see TPUs usage on GCP website, we can check the [Metrics Explorer](https://console.cloud.google.com/monitoring/metrics-explorer), select `tpu.googleapis.com/accelerator/memory_used` and `tpu.googleapis.com/accelerator/duty_cycle`.


If we need to stop the python processes:
```bash
# gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="pkill -9 -f python"
# gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="tmux kill-session -t gemma"
```


##### Clean-up

Finally, once we are done and TPU VMs are unused, let's delete them:
```bash
yes | gcloud compute tpus tpu-vm delete $TPU_NAME --async
```
