## Gaussian processes training example

Demonstrates using distribution type objects inside NN-Layers

For a simple example of GP regression
inside of a `flax.nn.Model` run

```
> python basic_gp.py
```

For an example of fitting a variational Gaussian
process run

```shell script
> python basic_svgp.py
```

The additional files
* `distributions.py` contains basic `Distributions` classes
as `@flax.struct` decorated data classes allowing them to be passed
as the output of `nn.Module` layer.
* `kernels.py` defines functional definitions of the kernel function, 
as well `KernelProvider`s taking the form of `nn.Module` layers handling
initialisation of the kernels and tracking of parameters.
* `gaussian_processes.py` defines a dataclasses for a `GaussianProcess`
and `VariationalGaussianProcess`.
* `likelihoods.py` defines simple likelihood classes to specify observation models.
* Finally `utils.py` contains additional utility functions shared accross the above
modules.

