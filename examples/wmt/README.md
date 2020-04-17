## Machine Translation
Trains a Transformer-based model (Vaswani *et al.*, 2017) on the WMT Machine
Translation dataset.

This example uses linear learning rate warmup and inverse square root learning
rate schedule.

### Requirements
* TensorFlow dataset `wmt17_translate/de-en` will be downloaded and prepared
  automatically, if necessary.  A subword tokenizer vocabulary will also be
  generated and saved on the first training run.

 ### How to run
  `python train.py --batch_size=256 --model_dir=./wmt_256`
