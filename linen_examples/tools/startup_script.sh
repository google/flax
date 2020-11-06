#!/bin/bash

# Replaced by launch_gce.py
REPO='__REPO__'
BRANCH='__BRANCH__'
EXAMPLE='__EXAMPLE__'
TIMESTAMP='__TIMESTAMP__'
NAME='__EXAMPLE__/__NAME__/__TIMESTAMP__'
ARGS='__ARGS__'
GCS_MODEL_DIR='__GCS_MODEL_DIR__'

HOME=/train

echo 'tmux a' > /attach.sh
chmod a+x /attach.sh

mkdir -p $HOME
cd $HOME

tmux new-session -s flax -d htop ENTER
tmux split-window
tmux send "

(

  [ -d flax ] || (
    git clone -b $BRANCH $REPO &&
    cd flax &&

    python3 -m pip install virtualenv &&
    python3 -m virtualenv env &&
    . env/bin/activate &&

    pip install -U pip &&
    pip install --upgrade jax jaxlib==0.1.55+cuda100 -f https://storage.googleapis.com/jax-releases/jax_releases.html &&
    pip install -e . &&

    cd linen_examples/$EXAMPLE &&
    pip install -r requirements.txt &&
    cd $HOME
  ) &&

  cd flax &&
  . env/bin/activate &&
  cd linen_examples/$EXAMPLE &&

  python ${EXAMPLE}_main.py --model_dir=$HOME/model_dir/$NAME $ARGS &&

  gsutil cp -R $HOME/model_dir/$NAME $GCS_MODEL_DIR


) 2>&1 | tee -a setup_train_log_${TIMESTAMP}.txt >(logger -t flax)

echo
echo WILL SHUT DOWN IN 5 MIN ...
sleep 300 && sudo shutdown now
"
tmux split-window -h
tmux send "
while true; do
  gsutil rsync -x '*/checkpoint_*' -r model_dir $GCS_MODEL_DIR
  sleep 60
done 2>&1 | tee -a gcs_rsync_${TIMESTAMP}.txt >(logger -t flax)
" ENTER
