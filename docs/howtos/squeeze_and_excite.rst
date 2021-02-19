Squeeze-and- Excitation (SE) Block
=============================

We will show you how to...

* Implement a Squeeze-and- Excitation (SE) block

Squeeze-and- Excitation blocks `(Hu at el., 2017) <https://arxiv.org/abs/1709.01507>`_ seek to
improve the quality of representations learned by a network by modelling the dependencies between channels of convolutional features.
The first step in the block is the Squeeze operation: all channels are aggregated by taking their means, changing the 
tensor's shape from (H, W, C) to (1, 1, C). Then it goes through an Excitation operation: the mean vector passes through a 2 layer MLP
before being  used in a self-gating mechanism, where it modulates each channel. The output still has shape (H, W, C) and so can just be passed
on to the next layer.
.. code-block:: python


  class SEBlock(nn.Module):
    """The Squeeze-and-Excitation block."""
    hidden_size: int
    act: Callable = nn.relu 
    axis: Tuple[int, int] = (1, 2), 

    @nn.compact
    def __call__(self, x):
        y = x.mean(axis=axis, keepdims=True)
        y = nn.Dense(
            y, features=self.hidden_size,
            name='reduce')
        y = act(y)
        y = nn.Dense(
            y, features=x.shape[-1],
            name='expand')
        return nn.sigmoid(y) * x
