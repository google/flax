# Where to find the docs

The FLAX documentation can be found here:
https://flax.readthedocs.io/en/latest/

# How to build the docs

1. Clone the `flax` repository with `git clone https://github.com/google/flax.git`.
2. In the main `flax` folder, install the required dependencies using `pip install -r docs/requirements.txt`.
3. Install [`pandoc`](https://pandoc.org): `pip install pandoc`.
4. [Optional] If you need to make any local changes to the docs, create and switch to a branch. Make your changes to the docs in that branch.
5. To build the docs, in the `flax/docs` folder run the make script: `make html`. Alternatively, install [`entr`](https://github.com/eradman/entr/), which helps run arbitrary commands when files change. Then run `find ../ ! -regex '.*/[\.|\_].*' | entr -s 'make html'`.
6. If the build is successful, you should get the `The HTML pages are in _build/html.` message. You can preview the docs in `flax/docs/_build/html`.

# How to run embedded code tests

We use `doctest` blocks for embedded code in documents, that are also
tested. Learn more at https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html

To run tests locally, run `make doctest`

# How to write code documentation

Our documentation is written in reStructuredText for Sphinx. It is a
meta-language that is compiled into online documentation. For more details, 
check out
[Sphinx's documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
As a result, our docstrings adhere to a specific syntax that has to be kept in
mind. Below we provide some guidelines.

To learn how to contribute to Jupyter Notebooks or other formats in Flax docs,
refer to the dedicated
[Contributing](https://flax.readthedocs.io/en/latest/contributing.html) page.

## How much information to put in a docstring

Docstring should be informative. We prefer to err on the side of too much
documentation than too little. For instance, providing a one-line explanation
to a new `Module` which implements new functionality is not sufficient.

Furthermore, we highly encourage adding examples to your docstrings, so users
can directly see how code can be used.

## How to write inline tested code

We use [doctest](https://docs.python.org/3/library/doctest.html) syntax for
writing examples in documentation. These examples are ran as tests as part of
our CI process. In order to write `doctest` code in your documentation, please
use the following notation:

```bash
# Example code::
#
#   def sum(a, b):
#     return a + b
#
#   sum(0, 1)
```

The `Example code` string at the beginning can be replaced by anything as long
as there are two semicolons and a newline following it, and the code is
indented.

## How to use "code font"

When writing code font in a docstring, please use double backticks. Example:

```bash
# This returns a ``str`` object.
```

Note that argument names and objects like True, None or any strings should
usually be put in `code`.

## How to create cross-references/links

It is possible to create cross-references to other classes, functions, and
methods. In the following, `obj_typ` is either `class`, `func`, or `meth`.

```bash
# First method:
# <obj_type>:`path_to_obj`

# Second method:
# :<obj_type>:`description <path_to_obj>`
```

You can use the second method if the `path_to_obj` is very long. Some examples:

```bash
# Create: a reference to class flax.linen.Module.
# :class:`flax.linen.Module`

# Create a reference to local function my_func.
# :func:`my_func`

# Create a reference "Module.apply()" to method flax.linen.Module.apply.
# :meth:`Module.apply() <flax.linen.Module.apply>`  #
```

To creata a hyperlink, use the following syntax:
```bash
# Note the double underscore at the end:
# `Link to Google <http://www.google.com>`__
```

### How to specify arguments for classes and methods

*  Class attributes should be specified using the `Attributes:` tag.
*  Method argument should be specified using the `Args:` tags.
*  All attributes and arguments should have types.

Here is an example from our library:

```python
class DenseGeneral(Module):
  """A linear transformation with flexible axes.
    Attributes:
      features: int or tuple with number of output features.
      axis: int or tuple with axes to apply the transformation on. For instance,
        (-2, -1) will apply the transformation to the last two axes.
      batch_dims: tuple with batch axes.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
  """
  features: Union[int, Iterable[int]]
  axis: Union[int, Iterable[int]] = -1
  batch_dims: Iterable[int] = ()
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  precision: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.
    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    ...
```