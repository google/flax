Squeeze and Excite Block
=============================

We will show you how to...

* Implement a Squeeze and Excite Block

.. code-block:: python


  class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
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
