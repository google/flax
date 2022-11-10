#!/bin/bash

# Note that all __XYZ__ strings are replaced by launch_gce.py

WORKDIR="/train/workdir_base/__EXAMPLE__/__NAME__/__TIMESTAMP__"

mkdir -p /train
cd /train

# Login directly with:
# gcloud compute ssh $VM -- /sudo_tmux_a.sh
echo -e '#!/bin/bash\nsudo /tmux_a.sh' > /sudo_tmux_a.sh
chmod a+x /sudo_tmux_a.sh
echo -e '#!/bin/bash\ntmux a' > /tmux_a.sh
chmod a+x /tmux_a.sh

# Main script running in bottom left tmux pane.
cat >/install_train_stop.sh <<EOF
set -x
(
  conda activate flax &&

  [ -d flax ] || (
    git clone --depth 1 -b __BRANCH__ __REPO__ &&
    cd flax &&

    conda create -yn flax python==3.9 &&
    conda activate flax &&

    pip install -U pip &&
    pip install -e . &&

    cd examples/__EXAMPLE__ &&
    pip install -r requirements.txt &&
    cd /train
  ) &&

  conda activate flax &&
  cd flax &&
  cd examples/__EXAMPLE__ &&

  TFDS_DATA_DIR='__TFDS_DATA_DIR__' python main.py --workdir=$WORKDIR __ARGS__

) 2>&1 | tee -a $WORKDIR/setup_train_log_${TIMESTAMP}.txt

if [ __SHUTDOWN_SECS__ -gt 0 ]; then
  echo
  echo WILL SHUT DOWN IN $((__SHUTDOWN_SECS__/60)) MIN ...
  sleep __SHUTDOWN_SECS__ && shutdown now
fi

EOF


# Set up TMUX panes:
tmux new-session -s flax -d
# - top left: htop
tmux send 'htop
'
tmux split-window
tmux selectp -U
tmux split-window -h
# - top right: htop
tmux send 'watch nvidia-smi
'
tmux selectp -D
# - bottom left: main script
tmux send '. /install_train_stop.sh
'
tmux split-window -h
# - bottom right: rsync files to GCS bucket.
tmux send "
while true; do
  gsutil rsync -r workdir_base __GCS_WORKDIR_BASE__
  sleep 60
done 2>&1 | tee -a $WORKDIR/gcs_rsync_'__TIMESTAMP__'.txt
"
