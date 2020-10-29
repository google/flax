#!/bin/bash

# output in /var/log/syslog

REPO=https://github.com/andsteing/flax
BRANCH=imagenet
CONFIG='configs/v100_x8_mixed_precision.py'
CONFIG='configs/v100_x8.py'

GCS_LOGS_TEMP='gs://flax_us/logs_temp'
GCS_LOGS='gs://flax_us/logs'
# DATA_DIR='gs://tensorflow-datasets/datasets'
DATA_DIR='gs://flax_us/datasets'
NOW=$(date +%F_%H%M%S)
NAME="imagenet_half_$NOW"
NAME="imagenet_full_$NOW"

cd
mv flax flax_$NOW
mv logs logs_$NOW

tmux new-session -s train_imagenet -d htop ENTER
tmux split-window
tmux send "
git clone -b $BRANCH $REPO &&
cd flax &&

python3 -m pip install virtualenv &&
python3 -m virtualenv env &&
. env/bin/activate &&

pip install -U pip &&
pip install --upgrade jax jaxlib==0.1.55+cuda100 -f https://storage.googleapis.com/jax-releases/jax_releases.html &&
pip install -e . &&

cd linen_examples/imagenet &&
pip install -r requirements.txt &&

python imagenet_main.py --model_dir=../../../logs/$NAME --data_dir=$DATA_DIR --config=$CONFIG &&
gsutil cp -R ../../../logs/$NAME $GCS_LOGS/

echo
echo WILL SHUT DOWN IN 5 MIN ...
sleep 300 && sudo shutdown now
"
tmux split-window -h
tmux send "while true; do gsutil rsync -r logs $GCS_LOGS_TEMP; sleep 60; done" ENTER

echo install_train.sh FINISHED
