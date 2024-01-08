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
2. Set up billing: https://console.cloud.google.com/billing
3. Create a storage bucket (GCS).
4. Optional: Get quota for accelerators. This is usually granted with a short
   delay: https://console.cloud.google.com/iam-admin/quotas

## Setting up your environment

The commands below use the same set of pre-defined environment variables.

Mandatory environment variables:

- `$PROJECT`: The name of your Google Cloud project.
- `$GCS_BUCKET`: The name of the Google Cloud Storage bucket where the model
  output (artifacts, final checkpoint) are to be stored.
- `$ZONE`: Compute zone (e.g. `central1-a`).

Optional environment variables:

- `$REPO`: Alternative repo to use instead of the default
  https://github.com/google/flax - this is useful for development.
- `$BRANCH`: Alternative branch to use instead of the default `main`.

## Training the MNIST example

Use the following command to launch the MNIST example on cloud (make sure to set
`$PROJECT` and `$GCS_BUCKET` accordingly):

```shell
python examples/cloud/launch_gce.py \
  --project=$PROJECT \
  --zone=us-west1-a \
  --machine_type=n2-standard-2 \
  --gcs_workdir_base=gs://$GCS_BUCKET/workdir_base \
  --repo=${REPO:-https://github.com/google/flax} \
  --branch=${BRANCH:-main} \
  --example=mnist \
  --args='--config=configs/default.py' \
  --name=default
```

## Training the imagenet example

Note that you need to first prepare the `imagenet2012` dataset. For this,
download the data from http://image-net.org/ as described in the
[tensorflow_datasets catalog](https://www.tensorflow.org/datasets/catalog/imagenet2012).
Then point the environment variable `$IMAGENET_DOWNLOAD_PATH` to the directory
where the downloads are stored and prepare the dataset by running

```shell
python -c "
import tensorflow_datasets as tfds
tfds.builder('imagenet2012').download_and_prepare(
    download_config=tfds.download.DownloadConfig(
        manual_dir='$IMAGENET_DOWNLOAD_PATH'))
"
```

Then copy the contents of the directory `~/tensorflow_datasets` into the
directory `gs://$GCS_TFDS_BUCKET/datasets` (note that `$GCS_TFDS_BUCKET` and
`$GCS_BUCKET` can be identical).

After this preparation you can run the imagenet example with the following
command (make sure to set `$PROJECT`, `$GCS_BUCKET` and `$GCS_TFDS_BUCKET`
accordingly):

```shell
python examples/cloud/launch_gce.py \
  --project=$PROJECT \
  --zone=us-west1-a \
  --machine_type=n1-standard-96 \
  --accelerator_type=nvidia-tesla-v100 --accelerator_count=8 \
  --gcs_workdir_base=gs://$GCS_BUCKET/workdir_base \
  --tfds_data_dir=gs://$GCS_TFDS_BUCKET/datasets \
  --repo=${REPO:-https://github.com/google/flax} \
  --branch=${BRANCH:-main} \
  --example=imagenet \
  --args='--config=configs/v100_x8_mixed_precision.py' \
  --name=v100_x8_mixed_precision
```

## Tips

You can add `--connect` to above commands to directly land in the training
session once the VM is ready. This is very helpful for debugging when changing
things. Note that the VM automatically shuts down after 5 minutes of inactivity,
both in case of success as in case of failure. On OS X this could be combined
with `VM_READY_CMD="osascript -e 'display notification \"VM ready\"'"` so get
undistracted when the VM is up and running.

When tweaking the startup script or individual arguments, it is often helpful to
connect to the VM, stop the scripts and end the tmux session, and then copy and
paste the contents of the generated `flax-...-startup_script.sh`, after
modifying these contents accordingly.