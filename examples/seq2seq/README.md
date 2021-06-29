## seq2seq addition

This example trains a simple LSTM on a sequence-to-sequence addition task using
an encoder-decoder architecture. The data is generated on the fly.

Colab lets you edit the source files and interact with the model:

https://colab.research.google.com/github/google/flax/blob/master/examples/seq2seq/seq2seq.ipynb

### Example output

From Colab run that also generated [tfhub.dev]

```
INFO:absl:[1900] accuracy=0.992188, loss=0.009365
INFO:absl:DECODE: 48+57 = 105 (CORRECT)
INFO:absl:DECODE: 13+59 = 72 (CORRECT)
INFO:absl:DECODE: 83+948 = 1031 (CORRECT)
INFO:absl:DECODE: 91+280 = 371 (CORRECT)
INFO:absl:DECODE: 65+270 = 335 (CORRECT)
```

[tfhub.dev]: https://tensorboard.dev/experiment/h81jpOlgS5iBJv4MVdznRQ/#scalars&_smoothingWeight=0

### How to run

`python train.py`
