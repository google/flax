#!/bin/bash

# Manual
# mkdir train
# git clone https://github.com/raj-prince/flax.git
# cd train
# cd flax
# git checkout princer_jax

# and run this script from the /flax folder

# assuming in flax folder
conda create -yn flax_env python==3.10
conda activate flax_env

pip install -U pip
pip install -e .

cd ../examples/imagenet
mkdir ws

pip install -r requirements.txt
pip install tensorrt

ln -s /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.8 /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.7
ln -s /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.8 /opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/flax_env/lib/python3.10/site-packages/tensorrt_libs/

# Run the model
python main.py --workdir="./ws" --config="./configs/v100_x8.py"
