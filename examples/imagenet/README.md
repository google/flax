

# How to run

```
python train.py --batch_size=[batch_size] --model_dir=[model_dir]
```


# Using mixed precision training on NVIDIA V100 GPUs

```
python train.py --loss_scaling=256. --half_precision=True \
    --batch_size=[256*num_gpus] --model_dir=[model_dir]
```
