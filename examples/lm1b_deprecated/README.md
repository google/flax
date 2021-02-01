## NOTE
This example uses the deprecated `flax.nn` API. https://github.com/google/flax/issues/567 tracks the update
to [Linen](https://github.com/google/flax/tree/master/flax/linen).

## Language modeling
Trains a Transformer-based model (Vaswani *et al.*, 2017) on the One Billion Word Benchmark (lm1b; Chelba *et al.*, 2013).

This example uses linear learning rate warmup and inverse square root learning rate schedule.

### Requirements
* TensorFlow dataset `lm1b/subwords32k` will be downloaded and prepared automatically, if necessary.

### Supported setups
The model should run with other configurations and hardware, but explicitely tested on the following.

| Hardware | Batch size | Training time | Perplexity  | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| 8 x Nvidia V100 (16GB)  | 2048  |  1d 4h 32m  | 33.24 | [2020-03-14](https://tensorboard.dev/experiment/gG67xEXDTLywlagjVHQetw/) |

### How to run

#### 8 x Nvidia V100 (16GB)
`python train.py --batch_size=2048 --model_dir=./lm1b_bs=2048`
