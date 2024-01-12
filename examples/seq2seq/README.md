## seq2seq addition

This example trains a simple LSTM on a sequence-to-sequence addition task using
an encoder-decoder architecture. The data is generated on the fly.

Colab lets you edit the source files and interact with the model:

https://colab.research.google.com/github/google/flax/blob/main/examples/seq2seq/seq2seq.ipynb

### Example output

From Colab run that also generated [tfhub.dev]

```
INFO:absl:[1800] accuracy=1.0, loss=0.0020284138154238462
INFO:absl:DECODE: 14+381 = 395 (CORRECT)
INFO:absl:DECODE: 68+91 = 159 (CORRECT)
INFO:absl:DECODE: 0+807 = 707 (INCORRECT) correct=807
INFO:absl:DECODE: 95+532 = 627 (CORRECT)
INFO:absl:DECODE: 6+600 = 606 (CORRECT)
```

[tfhub.dev]: https://tensorboard.dev/experiment/TwvKVBqzTaKWgEbyebillw/#scalars&_smoothingWeight=0

### How to run

`python train.py`

The total runtime for 1200 steps on CPU (3.5GHz Intel Core i7, 16GB memory) is
about 4 minutes.
