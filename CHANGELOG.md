Changelog
----------

We only include user-facing changes here. See our Git history or issue tracker for a full
list.

vNEXT
-----

v0.2.1
------
 - Support both pre- and post-omnistaging changes in JAX core.
 - Add single input conv and conv_transpost support (for use with vmap)
 - Export new initializers
 - Add Adafactor optimizer


v0.2
----
 - Added JAX trace-level checks for transforms.
 - BatchNorm added axis_index_groups for control in parallel training.
 - Optimizers broken out into separate directory with base class and implementations.
 - traverse_util added flatten_dict and unflatten_dict utility methods for nested dicts.


v0.1
----

 - Add ConvTranspose Module to nn.linear
 - Rename the following optional arguments to nn.linear.Conv:
     `lhs_dilation` -> `input_dilation`,
     `rhs_dilation` -> `kernel_dilation`
 - Change default layer names from numbers '0', '1', etc. to
   include the Module class name, e.g. 'Dense_0', 'LayerNorm_1'.

