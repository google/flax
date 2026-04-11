
## Language modeling

Trains Gemma model on the One Billion Word Benchmark (lm1b; Chelba *et al.*, 2013).

This example is using Flax NNX API.


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
pip install jax[cuda13]
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
Below, we will be using TPU containing 16 devices on 4 VMs.


##### TPU setup

The TPU has multiple TPU VMs we will create a startup bash script to run on start-up on each VM.

```bash
# v4 TPUs
# export ZONE=us-central2-b
# export ACCELERATOR_TYPE=v4-32
# export RUNTIME_VERSION=tpu-ubuntu2204-base

# v6e TPUs
# Available zones: https://docs.cloud.google.com/tpu/docs/regions-zones
# Runtimes: https://docs.cloud.google.com/tpu/docs/runtimes
# export ZONE=us-east1-d
# export ZONpE=us-central1-b
# export ZONE=us-east5-a
# export ZONE=us-east5-b
# export ACCELERATOR_TYPE=v6e-16
# export RUNTIME_VERSION=v2-alpha-tpuv6e
# export TPU_NAME=flax-gemma-lm1b-${ACCELERATOR_TYPE}

# v5p TPUs
# Available zones: https://docs.cloud.google.com/tpu/docs/regions-zones
# Runtimes: https://docs.cloud.google.com/tpu/docs/runtimes
# export ZONE=us-central1-a
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v5p-32
export RUNTIME_VERSION=v2-alpha-tpuv5
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
  "numpy~=2.3" \
  "optax~=0.2" \
  "sentencepiece~=0.2" \
  "jaxtyping~=0.3" \
  "tensorflow-cpu~=2.20" \
  "tensorboard~=2.20" \
  "tensorflow-datasets~=4.9" \
  "grain~=0.2" \
  "orbax-checkpoint[gcp]~=0.11" \
  "google-cloud-storage"

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
gcloud compute tpus tpu-vm ssh --zone $ZONE --worker=all --command="journalctl -u google-startup-scripts.service | tail -n 5" $TPU_NAME

# ...
# Mar 25 13:03:58 t1v-n-51d8f787-w-3 systemd[1]: Finished Google Compute Engine Startup Scripts.
# Mar 25 13:03:58 t1v-n-51d8f787-w-3 systemd[1]: google-startup-scripts.service: Consumed 1min 6.152s CPU time.
```
Make sure above logs do not show any errors like `startup-script exit status 100`.


Once all done, we should see `flax`, `logs` and `arrayrecord_datasets` folders:
```bash
gcloud compute tpus tpu-vm ssh --zone $ZONE --worker=all --command="ls /root/" $TPU_NAME
# flax
# logs
# arrayrecord_datasets
# ... multiple times ...

gcloud compute tpus tpu-vm ssh --zone $ZONE --worker=all --command="ls /root/arrayrecord_datasets" $TPU_NAME

# lm1b
# ... multiple times ...
```

Check python version and available TPUs:
```bash
gcloud compute tpus tpu-vm ssh --zone $ZONE --worker=all --command="python -VV" $TPU_NAME

# Python 3.12.13 (main, Mar 20 2026, 00:33:26) [Clang 22.1.1 ]
# ... multiple times ...

gcloud compute tpus tpu-vm ssh --zone $ZONE --worker=all --command="python -c 'import jax; print(f\"{jax.process_index()=}, num devices={len(jax.devices())}\")'" $TPU_NAME

# jax.process_index()=0, num devices=16
# jax.process_index()=3, num devices=16
# jax.process_index()=2, num devices=16
# jax.process_index()=1, num devices=16
```

##### Training

Let's assume that we have locally the training code. We can copy the code from the current folder to TPU VMs:
```bash
gcloud compute tpus tpu-vm scp --zone $ZONE --recurse *.py configs $TPU_NAME:/root/gemma-example --worker=all
```

Let's create the output folder using worker 0 only:
```bash
export out_name=gemma3-1b_lm1b_run-$ACCELERATOR_TYPE-$(date -u +%Y%m%d-%H%M)
export out_dir=/root/logs/$out_name
export chpt_bucket=gs://$GCS_OUTPUT_BUCKET/$out_name/checkpoint

gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=0 --command="export out_dir=$out_dir && mkdir -p \$out_dir"
gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=0 --command="ls /root/logs"
```

Let's set up the training command:
```bash
# env vars:
setup_command="export TF_ENABLE_ONEDNN_OPTS=0 && export TFDS_DATA_DIR=/root/arrayrecord_datasets && export out_dir=$out_dir"

# get current host id for logs files:
get_proc_id_command="export proc_id=\`python -c \"import jax; print(jax.process_index())\"\` && echo \"proc_id=\$proc_id\""

# python command to run:
command="cd /root/gemma-example && python -u main.py --workdir=\$out_dir --chpt_bucket=$chpt_bucket --config=configs/default.py --config.prefetch_num_workers=8 &> \$out_dir/output.w\$proc_id.log"

# full command with tmux:
full_command="tmux new -d -s gemma '$setup_command && $get_proc_id_command && $command'"

gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME \
  --worker=all \
  --command="$full_command"
```

We can check whether python processes are running:
```bash
gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=all --command="ps -aux | grep -E 'python -u main.py'"
```

We can also check the logs files:
```bash
gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=0 --command="cat $out_dir/output.w0.log"
```
<details>
<summary>
Example output
</summary>

```
I0325 13:58:52.664953 130295542103168 main.py:56] JAX version: 0.9.2
I0325 13:58:52.705875 130295542103168 main.py:57] Flax version: 0.12.6
I0325 13:58:52.714860 130295542103168 distributed.py:149] Starting JAX distributed service on [::]:8482
I0325 13:58:52.716724 130295542103168 distributed.py:172] Connecting to JAX distributed service on 10.202.0.27:8482
I0325 13:58:57.936235 130295542103168 main.py:66] JAX process: 0 / 4
I0325 13:58:57.936445 130295542103168 main.py:67] JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=2, process_index=1, coords=(2,0,0), core_on_chip=0), TpuDevice(id=3, process_index=1, coords=(3,0,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(2,1,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(3,1,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,2,0), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,2,0), core_on_chip=0), TpuDevice(id=12, process_index=2, coords=(0,3,0), core_on_chip=0), TpuDevice(id=13, process_index=2, coords=(1,3,0), core_on_chip=0), TpuDevice(id=10, process_index=3, coords=(2,2,0), core_on_chip=0), TpuDevice(id=11, process_index=3, coords=(3,2,0), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(2,3,0), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(3,3,0), core_on_chip=0)]
I0325 13:58:57.936518 130295542103168 main.py:68] FLAGS:
I0325 13:58:57.936607 130295542103168 main.py:69] - FLAGS.config=TrainConfig(vocab_path=None, vocab_size=35008, max_corpus_chars=10000000, dataset_name='lm1b', eval_dataset_name='lm1b', eval_split='test', per_device_batch_size=32, eval_per_device_batch_size=32, prefetch_num_workers=8, prompts=('Paris is a the capital', 'Flax is a', 'The shutdown was aimed at creating efficiencies as', 'A big theme of this hire is that there are parts of', 'Because of Bear Stearns , many analysts are', 'Next month , the Brazilian bourse'), sampling_temperature=0.0, sampling_top_p=0.95, num_train_steps=500000, num_eval_steps=2000, num_predict_steps=50, learning_rate=0.0016, warmup_steps=1000, label_smoothing=0.0, weight_decay=0.1, max_target_length=128, max_eval_target_length=512, transformer_name='gemma3_1b', transformer_params=None, save_checkpoints=True, restore_checkpoints=True, checkpoint_every_steps=10000, eval_every_steps=2000, use_bfloat16=True, seed=0, mesh_axes=('fsdp', 'tensor'), data_sharding=('fsdp',), fsdp_parallelism=-1, tensor_parallelism=1, with_profiler_step_trace=False, input_pipeline_type='grain', use_nnx_tree_mode=False, use_nnx_transforms='no')
I0325 13:58:57.936660 130295542103168 main.py:70] - FLAGS.workdir='/root/logs/gemma3-1b_lm1b_run-v6e-16-20260325'
I0325 13:58:57.936708 130295542103168 main.py:72] - FLAGS.chpt_bucket='gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v6e-16-20260325/checkpoint'
I0325 13:58:57.936762 130295542103168 local.py:45] Setting task status: process_index: 0, process_count: 4
I0325 13:58:57.936818 130295542103168 local.py:50] Created artifact workdir of type ArtifactType.DIRECTORY and value /root/logs/gemma3-1b_lm1b_run-v6e-16-20260325.
I0325 13:58:57.984114 130295542103168 train.py:468] Using mesh: Mesh('fsdp': 16, 'tensor': 1, axis_types=(Explicit, Explicit))
I0325 13:58:57.984221 130295542103168 train.py:472] Initializing dataset.
I0325 13:58:58.267392 130295542103168 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I0325 13:58:58.379809 130295542103168 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I0325 13:58:58.385863 130295542103168 dataset_info.py:707] Load dataset info from /root/arrayrecord_datasets/lm1b/1.1.0
I0325 13:58:58.387347 130295542103168 dataset_builder.py:892] Found random access formats: . Chose to use FileFormat.ARRAY_RECORD. Overriding file format in the dataset info.
I0325 13:58:58.463381 130295542103168 train.py:482] Initializing model, optimizer, and step functions.
I0325 13:59:06.872116 130295542103168 checkpoint_manager.py:709] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=None, item_handlers=None, handler_registry=None
I0325 13:59:06.960237 130295542103168 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x767f90bf21b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x767f90bf21b0>}).
I0325 13:59:06.960630 130295542103168 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.33
I0325 13:59:06.960731 130295542103168 async_checkpointer.py:177] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x767f987c4f40> timeout: 600 secs and primary_host=0 for async checkpoint writes
I0325 13:59:07.537147 130295542103168 checkpoint_manager.py:1818] Found 0 checkpoint steps in gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v6e-16-20260325/checkpoint
I0325 13:59:07.539128 130295542103168 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=<class 'orbax.checkpoint._src.path.atomicity.CommitFileTemporaryPath'>, save_decision_policy=None, preservation_policy=LatestN(n=1), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://flax-gemma-example-training/gemma3-1b_lm1b_run-v6e-16-20260325/checkpoint: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x767f983e0740>
I0325 13:59:07.554096 130295542103168 train.py:559]
Model Number of Parameters:
- Total (B): 0.73822528
- Embedding (B): 0.040329216
- Attentions (B): 0.076690432
- MLPs (B): 0.621084672

I0325 13:59:07.753760 130245504009792 logging_writer.py:80] [Hyperparameters] {'vocab_path': '/root/logs/gemma3-1b_lm1b_run-v6e-16-20260325/sentencepiece_model', 'vocab_size': 35008, 'max_corpus_chars': 10000000, 'dataset_name': 'lm1b', 'eval_dataset_name': 'lm1b', 'eval_split': 'test', 'per_device_batch_size': 32, 'eval_per_device_batch_size': 32, 'prefetch_num_workers': 8, 'prompts': ('Paris is a the capital', 'Flax is a', 'The shutdown was aimed at creating efficiencies as', 'A big theme of this hire is that there are parts of', 'Because of Bear Stearns , many analysts are', 'Next month , the Brazilian bourse'), 'sampling_temperature': 0.0, 'sampling_top_p': 0.95, 'num_train_steps': 500000, 'num_eval_steps': 2000, 'num_predict_steps': 50, 'learning_rate': 0.0016, 'warmup_steps': 1000, 'label_smoothing': 0.0, 'weight_decay': 0.1, 'max_target_length': 128, 'max_eval_target_length': 512, 'transformer_name': 'gemma3_1b', 'transformer_params': None, 'save_checkpoints': True, 'restore_checkpoints': True, 'checkpoint_every_steps': 10000, 'eval_every_steps': 2000, 'use_bfloat16': True, 'seed': 0, 'mesh_axes': ('fsdp', 'tensor'), 'data_sharding': ('fsdp',), 'fsdp_parallelism': -1, 'tensor_parallelism': 1, 'with_profiler_step_trace': False, 'input_pipeline_type': 'grain', 'use_nnx_tree_mode': False, 'use_nnx_transforms': 'no'}
I0325 13:59:07.812757 130295542103168 train.py:612] Starting training loop.
I0325 14:00:22.976243 130295542103168 train.py:652] Finished training step 0. Batch size: 512, Loss: 10.96665, LR: 0.00000
I0325 14:01:31.933391 130295542103168 train.py:652] Finished training step 1. Batch size: 512, Loss: 10.96693, LR: 0.00000
I0325 14:01:32.334728 130295542103168 local.py:41] Setting work unit notes: 0.0 steps/s, 0.0% (1/500000), ETA: 400d22h17m (2m : 4.3% data, 95.2% train_step)
I0325 14:01:32.335362 130245504009792 logging_writer.py:48] [1] steps_per_sec=0.014434
I0325 14:01:32.335719 130245504009792 logging_writer.py:48] [1] uptime=144.522
I0325 14:01:32.381767 130295542103168 train.py:652] Finished training step 2. Batch size: 512, Loss: 10.91155, LR: 0.00000
I0325 14:01:32.476979 130295542103168 train.py:652] Finished training step 3. Batch size: 512, Loss: 10.81033, LR: 0.00001
I0325 14:01:32.548323 130295542103168 train.py:652] Finished training step 4. Batch size: 512, Loss: 10.67908, LR: 0.00001
I0325 14:01:32.624506 130295542103168 train.py:652] Finished training step 5. Batch size: 512, Loss: 10.53421, LR: 0.00001
I0325 14:01:33.070958 130295542103168 train.py:652] Finished training step 6. Batch size: 512, Loss: 10.37890, LR: 0.00001
I0325 14:01:33.124931 130295542103168 train.py:652] Finished training step 7. Batch size: 512, Loss: 10.21490, LR: 0.00001
I0325 14:01:33.201636 130295542103168 train.py:652] Finished training step 8. Batch size: 512, Loss: 10.05384, LR: 0.00001
I0325 14:01:33.595465 130295542103168 train.py:652] Finished training step 9. Batch size: 512, Loss: 9.90343, LR: 0.00002
I0325 14:01:33.673433 130295542103168 train.py:652] Finished training step 10. Batch size: 512, Loss: 9.76386, LR: 0.00002
I0325 14:01:33.887800 130295542103168 train.py:652] Finished training step 11. Batch size: 512, Loss: 9.63710, LR: 0.00002
I0325 14:01:34.272973 130295542103168 train.py:652] Finished training step 12. Batch size: 512, Loss: 9.52143, LR: 0.00002
I0325 14:01:34.367378 130295542103168 train.py:652] Finished training step 13. Batch size: 512, Loss: 9.41437, LR: 0.00002
I0325 14:01:34.461776 130295542103168 train.py:652] Finished training step 14. Batch size: 512, Loss: 9.31737, LR: 0.00002
I0325 14:01:34.959434 130295542103168 train.py:652] Finished training step 15. Batch size: 512, Loss: 9.22604, LR: 0.00003
I0325 14:01:35.052989 130295542103168 train.py:652] Finished training step 16. Batch size: 512, Loss: 9.14349, LR: 0.00003
I0325 14:01:35.146110 130295542103168 train.py:652] Finished training step 17. Batch size: 512, Loss: 9.06840, LR: 0.00003
I0325 14:01:35.239722 130295542103168 train.py:652] Finished training step 18. Batch size: 512, Loss: 8.99818, LR: 0.00003
I0325 14:01:35.339613 130295542103168 train.py:652] Finished training step 19. Batch size: 512, Loss: 8.93296, LR: 0.00003
I0325 14:02:06.263196 130295542103168 local.py:50] Created artifact [10] Profile of type ArtifactType.URL and value None.
I0325 14:02:32.371587 130295542103168 local.py:41] Setting work unit notes: 5.5 steps/s, 0.1% (332/500000), ETA: 1d1h10m (3m : 4.4% data, 80.2% train_step)
I0325 14:02:32.468958 130245504009792 logging_writer.py:48] [332] steps_per_sec=5.51328
I0325 14:02:32.469507 130245504009792 logging_writer.py:48] [332] uptime=204.656
I0325 14:03:32.393238 130295542103168 local.py:41] Setting work unit notes: 13.5 steps/s, 0.2% (1145/500000), ETA: 10h13m (4m : 3.5% data, 84.2% train_step)
I0325 14:03:32.536956 130245504009792 logging_writer.py:48] [1145] steps_per_sec=13.5451
I0325 14:03:32.537706 130245504009792 logging_writer.py:48] [1145] uptime=264.724
I0325 14:04:32.438827 130295542103168 local.py:41] Setting work unit notes: 13.1 steps/s, 0.4% (1934/500000), ETA: 10h31m (5m : 3.0% data, 86.8% train_step)
I0325 14:04:32.528346 130245504009792 logging_writer.py:48] [1934] steps_per_sec=13.14
I0325 14:04:32.529090 130245504009792 logging_writer.py:48] [1934] uptime=324.715
I0325 14:04:37.407858 130295542103168 train.py:665] Gathering training metrics.
I0325 14:04:37.610165 130245504009792 logging_writer.py:48] [2000] train_accuracy=0.37408044934272766, train_loss=3.729034185409546
I0325 14:04:50.093986 130245504009792 logging_writer.py:64] [2000] Got texts: {'samples': ['Paris is a the capital of the world , and the city is always a place for the people to live .', 'Flax is a very good player , but he is a very good player .', 'The shutdown was aimed at creating efficiencies as well as improving the quality of the service .', 'A big theme of this hire is that there are parts of the industry that are not only in the business of making the most of these things , but also in the process of making them work .', 'Because of Bear Stearns , many analysts are still unconvinced that the Fed will be able to keep its key interest rate at a record low of zero to 0.25 percent .', 'Next month , the Brazilian bourse will open at the end of the year .']}.
I0325 14:04:50.096882 130295542103168 train.py:420] Gathering evaluation metrics.
I0325 14:08:19.201115 130295542103168 local.py:41] Setting work unit notes: 0.3 steps/s, 0.4% (2001/500000), ETA: 19d12h11m (9m : 1.8% data, 2.3% generate_text, 52.0% train_step, 0.0% training_metrics)
I0325 14:08:24.184369 130245504009792 logging_writer.py:48] [2000] eval_accuracy=0.4045374393463135, eval_loss=3.198998212814331, eval_perplexity=24.50796127319336
I0325 14:08:24.272225 130245504009792 logging_writer.py:48] [2001] steps_per_sec=0.295464
I0325 14:08:24.272729 130245504009792 logging_writer.py:48] [2001] uptime=551.51
I0325 14:09:19.241172 130295542103168 local.py:41] Setting work unit notes: 10.6 steps/s, 0.5% (2639/500000), ETA: 13h0m (10m : 1.7% data, 35.0% eval, 2.0% generate_text, 55.7% train_step, 0.0% training_metrics)
I0325 14:09:19.379712 130245504009792 logging_writer.py:48] [2639] steps_per_sec=10.6262
I0325 14:09:19.380425 130245504009792 logging_writer.py:48] [2639] uptime=611.566
I0325 14:10:19.286858 130295542103168 local.py:41] Setting work unit notes: 11.5 steps/s, 0.7% (3330/500000), ETA: 11h59m (11m : 1.8% data, 31.9% eval, 1.9% generate_text, 59.4% train_step, 0.0% training_metrics)
I0325 14:10:19.411864 130245504009792 logging_writer.py:48] [3330] steps_per_sec=11.5079
I0325 14:10:19.412623 130245504009792 logging_writer.py:48] [3330] uptime=671.599
I0325 14:11:17.737772 130295542103168 train.py:665] Gathering training metrics.
I0325 14:11:17.857037 130245504009792 logging_writer.py:48] [4000] train_accuracy=0.4054645895957947, train_loss=3.389246940612793
I0325 14:11:19.468421 130245504009792 logging_writer.py:64] [4000] Got texts: {'samples': ["Paris is a the capital of the world , and the world 's capital .", "Flax is a former member of the U.S. Army 's 82nd Airborne Division , which has been involved in the fighting in Iraq and Afghanistan .", "The shutdown was aimed at creating efficiencies as the company 's profits grew , and the company 's shares fell .", 'A big theme of this hire is that there are parts of the world that are not in the same position as the US .', 'Because of Bear Stearns , many analysts are now speculating that the bank might be forced to sell itself to raise capital .', 'Next month , the Brazilian bourse will be closed for the rest of the year .']}.
I0325 14:11:19.470139 130295542103168 train.py:420] Gathering evaluation metrics.
I0325 14:14:02.040402 130295542103168 local.py:41] Setting work unit notes: 3.0 steps/s, 0.8% (4001/500000), ETA: 1d21h44m (14m : 1.4% data, 23.9% eval, 1.6% generate_text, 51.0% train_step, 0.0% training_metrics)
I0325 14:14:07.022615 130245504009792 logging_writer.py:48] [4000] eval_accuracy=0.42954444885253906, eval_loss=2.988433599472046, eval_perplexity=19.85456085205078
I0325 14:14:07.470407 130245504009792 logging_writer.py:48] [4001] steps_per_sec=3.0123
I0325 14:14:07.474070 130245504009792 logging_writer.py:48] [4001] uptime=894.341
```

</details>


We can also start TensorBoard on one worker to inspect the training:
```bash
gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=0 --command="pip install xprof"
gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=0 --command="tensorboard --logdir=$out_dir --port 7007" -- -L 7007:localhost:7007
```


To see TPUs usage on GCP website, we can check the [Metrics Explorer](https://console.cloud.google.com/monitoring/metrics-explorer), select `tpu.googleapis.com/accelerator/memory_used` and `tpu.googleapis.com/accelerator/duty_cycle`.


If we need to stop the python processes:
```bash
# gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=all --command="pkill -9 -f python"
# gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --worker=all --command="tmux kill-session -t gemma"
```


##### Clean-up

Finally, once we are done and TPU VMs are unused, let's delete them:
```bash
yes | gcloud compute tpus tpu-vm delete --zone $ZONE $TPU_NAME --async
```

### Benchmarks

We run benchmarks using Gemma3 1B model for 500 train steps and 100 eval steps on GPUs and TPUs.

Dataflow | NNX tree-mode | Transforms | Hardware | Last train steps/second | Total time | CMD | Notes | Log
---|---|---|---|---|---|---|---|---
Grain | False | Jax | 2xGPU (RTX 8000) | 0.577668 | 26.48 mins  | `export TFDS_DATA_DIR=/space/arrayrecord_datasets && export out_dir=/output/gemma-1b-grain-graphmode-jax-2gpus && rm -rf $out_dir && mkdir -p $out_dir && python -u main.py --config=configs/benchmarks/gemma3_1b_grain.py --workdir=$out_dir &> $out_dir/run.log` | FSDP only, Mixed Precision: bfloat16, Time profiling: 0.9% checkpoint, 2.7% data, 22.5% eval, 0.9% generate_text, 72.4% train_step, 0.0% training_metrics | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-gemma-1b-grain-graphmode-jax-2gpus-log)
TF | False | Jax | 2xGPU (RTX 8000) | 0.581955 | 24.28 mins | `export TFDS_DATA_DIR=/space/tensorflow_datasets/ && export out_dir=/output/gemma-1b-tf-graphmode-jax-2gpus && rm -rf $out_dir && mkdir -p $out_dir && python -u main.py --config=configs/benchmarks/gemma3_1b_tf.py --workdir=$out_dir &> $out_dir/run.log` | FSDP only, Mixed Precision: bfloat16, Time profiling: 0.8% checkpoint, 0.0% data, 21.2% eval, 1.0% generate_text, 76.3% train_step, 0.0% training_metrics | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-gemma-1b-tf-graphmode-jax-2gpus-log)
Grain | True | NNX | 2xGPU (RTX 8000) | 0.581487 | 26.01 mins | `export TFDS_DATA_DIR=/space/arrayrecord_datasets && export out_dir=/output/gemma-1b-grain-treemode-nnx-2gpus && rm -rf $out_dir && mkdir -p $out_dir && python -u main.py --config=configs/benchmarks/gemma3_1b_grain.py --workdir=$out_dir --config.use_nnx_tree_mode=True --config.use_nnx_transforms=all &> $out_dir/run.log` | FSDP only, Mixed Precision: bfloat16, Time profiling: 0.8% checkpoint, 0.8% data, 22.9% eval, 0.9% generate_text, 74.0% train_step, 0.0% training_metrics | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-gemma-1b-grain-treemode-nnx-2gpus-log)
TF | True | NNX | 2xGPU (RTX 8000) | 0.572604 | 24.56 mins | `export TFDS_DATA_DIR=/space/tensorflow_datasets/ && export out_dir=/output/gemma-1b-tf-treemode-nnx-2gpus && rm -rf $out_dir && mkdir -p $out_dir && python -u main.py --config=configs/benchmarks/gemma3_1b_tf.py --workdir=$out_dir --config.use_nnx_tree_mode=True --config.use_nnx_transforms=all &> $out_dir/run.log` | FSDP only, Mixed Precision: bfloat16, Time profiling: 0.8% checkpoint, 0.0% data, 21.0% eval, 1.1% generate_text, 76.5% train_step, 0.0% training_metrics | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-gemma-1b-tf-treemode-nnx-2gpus-log)
Grain | False | Jax | TPU v5p-32 | - | 6.21 mins | `export out_name=bench-grain-jax-graphmode-gemma3-1b_lm1b_run-$ACCELERATOR_TYPE-$(date -u +%Y%m%d-%H%M)` / `command="cd /root/gemma-example && python -u main.py --workdir=\$out_dir --chpt_bucket=$chpt_bucket --config=configs/benchmarks/gemma3_1b_grain.py --config.prefetch_num_workers=8 &> \$out_dir/output.w\$proc_id.log"` | FSDP only, Mixed Precision: bfloat16, Time profiling:  | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-bench-grain-jax-graphmode-gemma3-1b_lm1b_run-v5p-32-log)
Grain | True | NNX | TPU v5p-32 | - | 6.22 mins | `export out_name=bench-grain-nnx-treemode-gemma3-1b_lm1b_run-$ACCELERATOR_TYPE-$(date -u +%Y%m%d-%H%M)` / `command="cd /root/gemma-example && python -u main.py --workdir=\$out_dir --chpt_bucket=$chpt_bucket --config=configs/benchmarks/gemma3_1b_grain.py --config.prefetch_num_workers=8 --config.use_nnx_tree_mode=True --config.use_nnx_transforms=all &> \$out_dir/output.w\$proc_id.log"` | FSDP only, Mixed Precision: bfloat16, Time profiling:  | [Log](https://gist.github.com/vfdev-5/147f9243c781b2bb699aa5898aea2baa#file-bench-grain-nnx-treemode-gemma3-1b_lm1b_run-v5p-32-log)