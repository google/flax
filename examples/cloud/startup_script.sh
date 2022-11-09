#!/bin/bash

BASE=/train

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
WORKDIR="$BASE/workdir_base/$NAME"


# Login directly with:
# gcloud compute ssh $VM -- /sudo_tmux_a.sh
echo -e '#!/bin/bash\nsudo /tmux_a.sh' > /sudo_tmux_a.sh
chmod a+x /sudo_tmux_a.sh
echo -e '#!/bin/bash\ntmux a' > /tmux_a.sh
chmod a+x /tmux_a.sh

mkdir -p $BASE
cd $BASE

tmux new-session -s flax -d htop ENTER
tmux split-window
tmux send "

(
  set -x

  conda activate flax &&

  [ -d flax ] || (
    git clone --depth 1 -b $BRANCH $REPO &&
    cd flax &&

    conda create -yn flax python==3.9 &&
    conda activate flax &&

    pip install -U pip &&
    pip install -e . &&

    cd examples/$EXAMPLE &&
    pip install -r requirements.txt &&
    cd $BASE
  ) &&

  conda activate flax &&
  cd flax &&
  cd examples/$EXAMPLE &&

  TFDS_DATA_DIR='$TFDS_DATA_DIR' python main.py --workdir=$WORKDIR $ARGS

) 2>&1 | tee -a $WORKDIR/setup_train_log_${TIMESTAMP}.txt

echo
echo WILL SHUT DOWN IN 5 MIN ...
sleep 300 && sudo shutdown now
"
tmux split-window -h
tmux send "
while true; do
  gsutil rsync -r workdir_base $GCS_WORKDIR_BASE
  sleep 60
done 2>&1 | tee -a $WORKDIR/gcs_rsync_${TIMESTAMP}.txt
" ENTER
