#!/bin/bash

# Replaced by launch_gce.py
REPO='__REPO__'
BRANCH='__BRANCH__'
EXAMPLE='__EXAMPLE__'
TIMESTAMP='__TIMESTAMP__'
NAME='__EXAMPLE__/__NAME__/__TIMESTAMP__'
ARGS='__ARGS__'
GCS_WORKDIR_BASE='__GCS_WORKDIR_BASE__'
TFDS_DATA_DIR='__TFDS_DATA_DIR__'
ACCELERATOR_TYPE='__ACCELERATOR_TYPE__'

HOME=/train

echo 'tmux a' > /attach.sh
chmod a+x /attach.sh

mkdir -p $HOME
cd $HOME

tmux new-session -s flax -d htop ENTER
tmux split-window
tmux send "

(
  set -x

  [ -d flax ] || (
    git clone -b $BRANCH $REPO &&
    cd flax &&

    python3 -m pip install virtualenv &&
    python3 -m virtualenv env &&
    . env/bin/activate &&

    pip install -U pip &&
    pip install -e . &&
    if [[ $ACCELERATOR_TYPE =~ ^nvidia- ]]; then
      pip install --upgrade jax jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    fi &&

    cd linen_examples/$EXAMPLE &&
    pip install -r requirements.txt &&
    cd $HOME
  ) &&

  cd flax &&
  . env/bin/activate &&
  cd linen_examples/$EXAMPLE &&

  TFDS_DATA_DIR='$TFDS_DATA_DIR' python main.py --workdir=$HOME/workdir_base/$NAME $ARGS &&

  gsutil cp -R $HOME/workdir_base/$NAME $GCS_WORKDIR_BASE

) 2>&1 | tee -a setup_train_log_${TIMESTAMP}.txt >(logger -t flax)

echo
echo WILL SHUT DOWN IN 5 MIN ...
sleep 300 && sudo shutdown now
"
tmux split-window -h
tmux send "
while true; do
  gsutil rsync -x 'checkpoint_*' -r workdir_base $GCS_WORKDIR_BASE
  sleep 60
done 2>&1 | tee -a gcs_rsync_${TIMESTAMP}.txt >(logger -t flax)
" ENTER
