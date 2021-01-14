Changelog
----------

vNext
------

(Add your change to a random empty line to avoid merge conflicts)
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 -
 - Added OptimizedLSTM: ~33% faster than the original LSTM when using <=1024 units
 - Bug Fix `Scope.variable` mutability check, before a variable could only be initialized
   if the 'params' collection was mutable.
 - Linen `Module` instances are now Frozen after `setup` has been called.
   Previously mutations after setup could be dropped silently. Now the stateless requirement
   is enforced by raising a TypeError in `__setattr__` after `setup`.
 - Pytrees of dicts and lists are transformed into FrozenDict and tuples during attribute assignment.
   This avoids undetected submodules and inner state. 

v0.3
-----
Linen is now out of Alpha (flax.nn is being deprecated)!

 - `flax.core.apply` and linen `Module.apply` will now only return the variables
   collections that were specified as mutable.
 - Fixed handling of multiple separate subclasses of a Module.
 - We now allow assignment of mixed Module pytrees in setup.
 - Refactored collection creation to fail early when modifying an undefined collection as
   before an non-existing non-mutable collection would just be silently ignored.
 - Added the silu activation function.
 - Add offset argument to Adafactor optimizer for fine-tuning schedules.
 - Relaxed limit on calling methods on unbound modules.
 - Relaxed parameter attribute check
 - Added centered version of RMSProp.
 - Added GCE getting started kit.
 - Renamed -gpu_type to -accelerator_type.
 - Fixed bug in MultiOptimizer causing it to throw away empty dictionary

### Improvements
 - Made FrozenDict constructor freeze correctly.
 - Made freeze a synonym of the FrozenDict constructor
 - Optimize freezing FrozenDicts by sharing immutable internal state.
 - We simplified __setattr__ handling of trees with Modules.
 - Minor improvements in dtype handling, broadcast option for dropout.
 - Added a dtype specification to Embed layer, made Adafactor use float32
   state consistently, and added a broadcasting option to the Dropout layer.
 - Improved frozen dict performance.
 - (Massive) docs improvements
 - End to end benchmarks added.
 - Examples were updated to Linen.

v0.2.2
----
 - Added Reinforcement Learning example (examples/ppo).
 - Fix Adafactor bug that prevented factorization.
 - Fix scan broadcast issue in functional core.
 - Fix initialization RNGs to work with omnistaging for jitted inits.
 - Replaces usage of 'param' kind to 'params' collection.
 - Fix LARS optimizer for zero param initialization.
 - Added various examples in Linen API. See [README.md](https://github.com/google/flax/blob/master/flax/linen/README.md) for more information.
 - Full JAX omnistaging compatibility.

v0.2
----
 - Added JAX trace-level checks for transforms.
 - BatchNorm added axis_index_groups for control in parallel training.
 - Optimizers broken out into separate directory with base class and implementations.
 - traverse_util added flatten_dict and unflatten_dict utility methods for nested dicts.

v0.1
----

### API Changes
 - Add ConvTranspose Module to nn.linear
 - Rename the following optional arguments to nn.linear.Conv:
     `lhs_dilation` -> `input_dilation`,
     `rhs_dilation` -> `kernel_dilation`
 - Change default layer names from numbers '0', '1', etc. to
   include the Module class name, e.g. 'Dense_0', 'LayerNorm_1'.

