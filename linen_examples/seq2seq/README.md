## seq2seq addition
This example trains a simple LSTM on a sequence-to-sequence addition task using
an encoder-decoder architecture. The data is generated on the fly.

### Example output

```
I0314 18:33:34.921972 139788256962368 train.py:280] train step: 9800, loss: 0.0004, accuracy: 100.00
I0314 18:33:35.791534 139788256962368 train.py:249] DECODE: 25+45   = 70   (CORRECT)
I0314 18:33:35.791721 139788256962368 train.py:249] DECODE: 27+92   = 119  (CORRECT)
I0314 18:33:35.791795 139788256962368 train.py:249] DECODE: 51+420  = 471  (CORRECT)
I0314 18:33:35.791843 139788256962368 train.py:249] DECODE: 49+450  = 499  (CORRECT)
I0314 18:33:35.791887 139788256962368 train.py:249] DECODE: 48+853  = 901  (CORRECT)
```

### How to run

`python train.py`
