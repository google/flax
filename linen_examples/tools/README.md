# Launching jobs on Google Cloud

This directory provides a simple script that can be used to create a new VM
on Google Cloud, train an example on that VM and then shutting it down.

The training is implemented in a shell that is run on the VM after startup by
setting the `startup_script_file` in the metadata. The script opens a TMUX
session, installs Flax repository from Github with all dependencies, and then
runs the training in parallel with `gsutil rsync` that copies the training
artifacts in a storage bucket.

The advantage of this approach is that every training is run in a single VM
that contains all code and configuration, so it is easy to run multiple
experiments in parallel without interference. The individual trainings can also
be inspected by logging into the VM via SSH and attaching to the tmux session.

The script `launch_gce.py` launches the VM and prints out the relevant commands
to track the progress update and to log into the machine.

Note that the VM also shuts down if an error is encountered, after waiting for
five minutes.

## Preparation

Prerequisites:

1. Create a Google Cloud account.
2. Setup billing.
3. Create a storage bucket (GCS).
4. Optional : Get quota for accelerators. This is usually granted with a short
   delay.

## Training the imagenet example.

Note that you need to first prepare `imagenet2012` dataset. For this, download
the data from http://image-net.org/ as described in the
[tensorflow_datasets catalog](https://www.tensorflow.org/datasets/catalog/imagenet2012)
and then prepare the dataset by running

```shell
python -c "
import tensorflow_datasets as tfds
tfds.builder('imagenet2012').download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        manual_dir='$IMAGENET_DOWNLOAD_PATH'))
"
```

Then copy the directory `~/tensorflow_datasets` to your storage bucket.

After this preparation you can run the imagenet example with the following
command (make sure to set `$PROJECT` and `$GCS_BUCKET` accordingly):

```shell
python launch_gce.py \
  --project=$PROJECT \
  --zone=us-central1-a \
  --gcs_model_dir=gs://$GCS_BUCKET/model_dir \
  --repo=https://github.com/google/flax \
  --branch=master \
  --example=imagenet \
  --args="--data_dir=gs://$GCS_BUCKET/tensorflow_datasets --config=configs/v100_x8_mixed_precision.py" \
  --name=mixed_precision
```