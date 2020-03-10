# Basic VAE Example

This is an implementation of the paper [Auto-Encoding with Variational Bayes](http://arxiv.org/abs/1312.6114) by D.P.Kingma and M.Welling.
This code follows [pytorch/examples/vae](https://github.com/pytorch/examples/blob/master/vae/README.md).

```bash
pip install -r requirements.txt
python main.py
```

## Examples;

If you run the code by above command, you can get some generated images:

![generated_mnist](https://github.com/makora9143/flax/blob/examples/vae/examples/vae/example.png)

Also, you can obtain ELBO of a test set as `107.0544 ± 0.2496` (5 times of trials)