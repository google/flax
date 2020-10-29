#!/bin/bash

# Installs prerequisites and trains model.
# See ./README.md

set -e

VM=flax-examples-imagenet-$(date +%F-%H%M%S)
PROJECT=flax-xgcp
# https://pantheon.corp.google.com/iam-admin/quotas/details;servicem=compute.googleapis.com;metricm=compute.googleapis.com%2Fnvidia_v100_gpus;limitIdm=1%2F%7Bproject%7D
# https://cloud.google.com/compute/docs/gpus
# ZONE=us-central1-a
ZONE=us-west1-a

# --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE \
# --metadata="install-nvidia-driver=True,proxy-mode=service_account" \
# --tags=deeplearning-vm,ssh-tunnel-iap \
# --image-family=tf-2-3 --image-project=ml-images \

gcloud compute instances create $VM \
--project $PROJECT --zone=$ZONE \
--image=c1-deeplearning-common-cu100-v20201015-ubuntu-1804 --image-project=ml-images \
--machine-type=n1-standard-96 \
--maintenance-policy=TERMINATE \
--accelerator=type=nvidia-tesla-v100,count=8  \
--scopes=cloud-platform,storage-full \
--boot-disk-size=256GB --boot-disk-type=pd-ssd \
--metadata=install-nvidia-driver=True \
--metadata-from-file=startup-script=$(dirname $0)/install_train.sh

echo
echo Created VM $VM
echo
echo You can try logging in after a little while:
echo
echo gcloud compute ssh --project $PROJECT --zone $ZONE $VM
echo
echo 'Then attach to the process using `sudo su; tmux a`'
echo
