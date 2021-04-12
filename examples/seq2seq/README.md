## seq2seq addition
This example trains a simple LSTM on a sequence-to-sequence addition task using
an encoder-decoder architecture. The data is generated on the fly.

### Example output

```
I1001 16:55:20.002528 140215911658880 train.py:367] train step: 5000, loss: 0.0004, accuracy: 100.00
I1001 16:55:20.498533 140215911658880 train.py:341] DECODE: 83+769 = 852 (CORRECT)
I1001 16:55:20.498713 140215911658880 train.py:341] DECODE: 96+401 = 497 (CORRECT)
I1001 16:55:20.498786 140215911658880 train.py:341] DECODE: 12+322 = 334 (CORRECT)
I1001 16:55:20.498831 140215911658880 train.py:341] DECODE: 67+80 = 147 (CORRECT)
I1001 16:55:20.498878 140215911658880 train.py:341] DECODE: 67+972 = 1039 (CORRECT)
```

### How to run

`python train.py`
