# Basic VAE Example

This is an implementation of the paper [Auto-Encoding with Variational Bayes](http://arxiv.org/abs/1312.6114) by D.P.Kingma and M.Welling.
This code follows [pytorch/examples/vae](https://github.com/pytorch/examples/blob/master/vae/README.md).

```bash
pip install -r requirements.txt
python train.py
```

## Examples

If you run the code by above command, you can get some generated images:

![generated_mnist](./sample.png)

and reconstructions of test set digits:

![reconstruction_mnist](./reconstruction.png)

The test set loss after 10 epochs should be around `104`.
