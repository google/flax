Changelog
----------

vNext
------
(Add your change to a random empty line to avoid merge conflicts)
-
-
-
-
- removed GeGLU simplistic activation, it should be implemented manually.
-
-
-
- removed FLAX_LAZY_RNG flag support for old non-lazy PRNG derivation mode
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

0.8.2
-----
- fixed rng guide outputs by @chiamp in https://github.com/google/flax/pull/3685
- enforce mask kwarg in norm layers by @chiamp in https://github.com/google/flax/pull/3663
- added kwargs to self.param and self.variable by @chiamp in https://github.com/google/flax/pull/3675
- added nnx normalization tests by @chiamp in https://github.com/google/flax/pull/3689
- added NNX init_cache docstring example by @chiamp in https://github.com/google/flax/pull/3688
- added nnx attention equivalence test by @chiamp in https://github.com/google/flax/pull/3687
- Fix bug that assumed frozen-dict keys were strings. by @copybara-service in https://github.com/google/flax/pull/3692
- added nnx rmsnorm by @chiamp in https://github.com/google/flax/pull/3691
- updated nnx compute_stats by @chiamp in https://github.com/google/flax/pull/3693
- fixed intercept_methods docstring by @chiamp in https://github.com/google/flax/pull/3694
- [nnx] Add Sphinx Docs by @cgarciae in https://github.com/google/flax/pull/3678
- Fix pointless docstring example of nn.checkpoint / nn.remat. by @levskaya in https://github.com/google/flax/pull/3703
- added default params rng to .apply by @chiamp in https://github.com/google/flax/pull/3698
- [nnx] add partial_init by @cgarciae in https://github.com/google/flax/pull/3674
- make make_rng default to 'params' by @chiamp in https://github.com/google/flax/pull/3699
- Add SimpleCell. by @carlosgmartin in https://github.com/google/flax/pull/3697
- fix Module.module_paths docstring by @cgarciae in https://github.com/google/flax/pull/3709
- Guarantee the latest JAX version on CI by @cgarciae in https://github.com/google/flax/pull/3705
- Replace deprecated API `jax.tree.map` by @copybara-service in https://github.com/google/flax/pull/3715
- Use `jax.tree_util.tree_map` instead of deprecated `jax.tree.map`. by @copybara-service in https://github.com/google/flax/pull/3714
- [nnx] simplify readme by @cgarciae in https://github.com/google/flax/pull/3707
- [nnx] add demo.ipynb by @cgarciae in https://github.com/google/flax/pull/3680
- Fix Tabulate's compute_flops by @cgarciae in https://github.com/google/flax/pull/3721
- [nnx] simplify TraceState by @cgarciae in https://github.com/google/flax/pull/3724
- Add broadcast of `strides` and `kernel_dilation` to `nn.ConvTranspose` by @IvyZX in https://github.com/google/flax/pull/3731
- [nnx] Fix State.__sub__ by @cgarciae in https://github.com/google/flax/pull/3704
- [nnx] always fold_in on fork + new ForkedKeys return type by @cgarciae in https://github.com/google/flax/pull/3722
- [nnx] explicit Variables by @cgarciae in https://github.com/google/flax/pull/3720
- Improves fingerprint definition for Modules in nn.jit. by @copybara-service in https://github.com/google/flax/pull/3736
- Flax: avoid key reuse in tests by @copybara-service in https://github.com/google/flax/pull/3740
- added Einsum layer by @chiamp in https://github.com/google/flax/pull/3710
- nn.jit: automatic fingerprint definition for dataclass attributes by @cgarciae in https://github.com/google/flax/pull/3737
- [NVIDIA] Use custom grad accumulation for FP8 params by @kaixih in https://github.com/google/flax/pull/3623
- removed nnx dataclass by @chiamp in https://github.com/google/flax/pull/3742
- [nnx] cleanup graph_utils by @cgarciae in https://github.com/google/flax/pull/3728
- Fix doctest and unbreak head by @IvyZX in https://github.com/google/flax/pull/3753
- [nnx] add pytree support by @cgarciae in https://github.com/google/flax/pull/3732
- fixed intercept_methods docstring by @chiamp in https://github.com/google/flax/pull/3752
- Add ConvLSTMCell to docs. by @carlosgmartin in https://github.com/google/flax/pull/3712
- [nnx] remove flagslib by @cgarciae in https://github.com/google/flax/pull/3733
- Fix tests after applying JAX key-reuse checker. See: by @copybara-service in https://github.com/google/flax/pull/3748

0.8.1
-----
- Added default collection in `make_rng`.
- Added `InstanceNorm` and renamed `channel_axes` to `feature_axes`.
- Added norm equivalence tests.
- Added `Module.module_paths` and doc.
- make `Sequential.__call__` compact.
- Added `nn.compact_name_scope` v3.
- Add explicit control over frozen/slots setting in `flax.struct.dataclass`.
- Replacing `jax.tree_util.tree_map` with mapping over leafs.
- Fixed docs and docstrings.

0.8.0
-----
- Added [NNX](https://github.com/google/flax/tree/main/flax/nnx#nnx), a neural network library for JAX that provides a simple yet powerful module system that adheres to standard Python semantics. Its aim is to combine the robustness of Linen with a simplified, Pythonic API akin to that of PyTorch.
- Added `nn.compact_name_scope` decorator that enables methods to act as compact name scopes as with regular Haiku methods. This makes porting Haiku code easier.
- Add copy() method to Module.  This is a user-friendly version of the internal clone() method with better
  defaults for common use cases.
- Added [`BatchApply`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#batchapply) class.
- Added `sow_weights` option in attention layer.
- Added [`MultiHeadAttention`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.MultiHeadAttention.html) alias.
- Added kwargs support for `nn.jit`.
- Deprecated `normalize` activation function, in favor of `standardize`.
- Added `GeGLU` activation function.
- Added `Enum` support for `tabulate` function.
- Added simple argument-only lifted `nn.grad` function.

0.7.5
-----
- Report forward and backward pass FLOPs of modules and submodules in `linen.Module.tabulate` and `summary.tabulate` (in new `flops` and `vjp_flops` table columns). Pass `compute_flops=True` and/or `compute_vjp_flops=True` to include these columns.
- Re-factored `MultiHeadDotProductAttention`'s call method signature, by adding
`inputs_k` and `inputs_v` args and switching `inputs_kv`, `mask` and `determistic`
to keyword arguments. See more details in [#3389](https://github.com/google/flax/discussions/3389).
- Use new typed PRNG keys throughout flax: this essentially involved changing
  uses of `jax.random.PRNGKey` to `jax.random.key`.
  (See [JEP 9263](https://github.com/jax-ml/jax/pull/17297) for details).
  If you notice dispatch performance regressions after this change, be sure
  you update `jax` to version 0.4.16 or newer.
- Added `has_improved` field to EarlyStopping and changed the return signature of
`EarlyStopping.update` from returning a tuple to returning just the updated class.
See more details in [#3385](https://github.com/google/flax/pull/3385)

0.7.4
-----
New features:
- Add QK-normalization to MultiHeadDotProductAttention
- Allow apply's method argument to accept submodules
- Add module path to nn.module.
- [JAX] Generate new type of PRNG keys

Bug fixes:
- Directly call original method if method interceptor stack is empty.
- fix stackoverflow when loading pickled module
- Improve kw_only_dataclass.
- Allow pass-through implementation of state dict
- Promote dot_general injections from a function to a module.

0.7.2
-----
New features:
- make `flax.core.copy` `add_or_replace` optional
- Add `use_fast_variance` option to `GroupNorm` and `BatchNorm` to allow disabling it.

Bug fixes:
- Use `field_specifiers` instead of `field_descriptors` in `@dataclass_transform`.
- Fix `nn.Module` typing.
- [JAX] Replace uses of `jax.experimental.pjit.with_sharding_constraint` with `jax.lax.with_sharding_constraint`.

0.7.1
-----
Breaking changes:
- Migrating Flax from returning FrozenDicts to returning regular dicts. More details can be found in this [announcement](https://github.com/google/flax/discussions/3191)

New features:
- Use pyink
- added dict migration guide to index
- add scan over layers section
- Expose options to customize rich.Table
- add support for initializing carry variables in scan
- Let Flax-Orbax to not port the shape of `target` arrays when they port the `target` shardings.

Bug fixes:
- Use import `orbax.checkpoint` which is a better import pattern.
- Use import `orbax.checkpoint as ocp` to avoid the verbosity of using 'orbax.checkpoint` every time.
- [linen] Add alternative, more numerically stable, variance calculation to `LayerNorm`.
- [linen] Minor cleanup to normalization code.
- Fix norm calculation bug for 0-rank arrays.
- [JAX] Remove references to jax.config.jax_array.
- [linen] Use `stack` instead of `concatenate` in `compute_stats`, to handle scalar stats case.
- [linen] More minor cleanup in normalization `compute_stats`.
- Fix warnings from atari gym.
- Refactor TypeHandler to operate over batches of values, rather than individual ones. This allows more flexibility for implementations that may operate more efficiently on batches.
- Fix carry slice logic
- make flax_basics guide use utility fns
- Fix checkpointing guide error at head
- Improve scan docs

0.7.0
-----
- RNNCellBase refactor.

0.6.11
-----
- Set Orbax-as-backend to be the default checkpointing method.
- Fix setup trigger issue under sharing and transforms.
- Add collection to self.scope.reserve(name, col) so that sow works with the same name in different collections.
- Minor improvements for Sequential.
- Improve the error message in MultiHeadDotProductAttention.
- Allow manually specifying the rng key for Dropout.
- RNN refactor.
- fixed missing separator for rng fold in.

0.6.10
-----
- Rudimentary quantization support: some layers can be parametrized with custom dot_general and conv_general_dilated.

0.6.9
-----
- Depend on `orbax-checkpoint` package instead of `orbax`.
- Refactored setup scripts to `project.toml`.
- Added pretty_repr utility fn.
- Fix get_partition_spec on replicated array.
- Updates imagenet.ipynb to use GPU Colab runtime, and fixed config.
- Upgrade checkpointing code to `jax.sharding`, and with more warnings.

0.6.8
-----
- The automatic checkpoint migration was temporarily rolled back due to legacy compatibility issues.
  - We still recommend you to use the [upgrade guide](https://flax.readthedocs.io/en/latest/guides/orbax_upgrade_guide.html) and migrate completely to the Orbax API to ensure stability.
  - Or alternatively, add `flax.config.update('flax_use_orbax_checkpointing', True)` to your project to avoid being impacted by the automatic migration process.
- Added utility functions to frozen_dict api.
- Migrated Flax away from `register_keypaths`.
- Fixes kwargs in convert_to_graphs_tuple_fn.
- Fixed examples in a few ways:
  - Bumped the TF version
  - Used latest checkpoint formats
  - Other misc fixes.

0.6.7
-----
- New checkpoints will be saved using Orbax! Please check out [upgrade guide](https://flax.readthedocs.io/en/latest/guides/orbax_upgrade_guide.html) and consider migrating completely to the Orbax API.
  - You could `flax.config.update('flax_use_orbax_checkpointing', False)` to temporarily disable this migration, but note that Flax legacy checkpointing will be removed 3 months from Mar 10, 2023.
- Migrating `FrozenDict` to regular dict: utility functions now work on both.
- Migrated Flax dataclass and `FrozenDict` to JAX pytree keypath API.
- Fixed pytype and improved typing for `Module`
- Fixed up uses of PyTree and PyTreeDef types.

0.6.6
-----
- 0.6.5 was yanked so this release contains all that was in 0.6.5 as well.
- Migrated regular dict to FrozenDict, currently controlled by a flag.
- Refactored and separate out name relaxation policy changes.
- Added RMS normalization layer.

0.6.5
-----
- Added logical partitioning helpers for using pjit with Flax.
- Add ``Module.lazy_init`` to avoid compute during Module initialization.

0.6.4
-----
New features:
- Our [ReadTheDoc](https://flax.readthedocs.io/en/latest/index.html) site is a lot more organized now! More improvements on the way.
- Flax auto-SPMD parallelism API to work seamlessly with `jax.pjit`: https://flax.readthedocs.io/en/latest/guides/flax_on_pjit.html
- Added new `zeros_init` and `ones_init` initializers.
- Adds standardize initializer.
- Allowed specifying method as a string.
- Allowed runtime overwrite of `flax.config` flags.

Bug fixes:
- Added missing `dataclass.fields` from `__repr__`.
- Renamed ConvLSTM to ConvLSTMCell.
- Fix some tiny inconsistencies between scope.py and module.py.
- Improved many many docstrings, comments and error messages.

0.6.3
-----
New features:
- Flax checkpointing now uses [Orbax](https://github.com/google/orbax) for more flexiblity and features.
- Added support for python 3.10 and removed support for 3.7.

Bug fixes:
- Fixed rng generation in DenseGeneral init.
- Improved support for Mac M1 chip.
- Bumped package versions for a bunch of examples.
- Improved many docstrings and error messages.

0.6.2
-----
New features:
- Add rng_collection argument to Dropout.
- Fix flax.linen.stochastic.Dropout.
- Add flag allow_partial_mpa_restoration in checkpointing.
- Use `gfile.remove` for files because it doesn't work on GCS files.
- Added guides for: Flax the Sharp Bits, Checkpointing, Extracting Gradients
- Improved existed documentation pages.
- Improved errors, error messages and tests.
- Removed codebase's trailing whitespaces.

Bug fixes:
- Fixes launch_gce.sh with imagenet example.
- Properly report AttributeErrors from descriptors.
- Fixes usages of `pmap`.
- Return None if no _parent_ref is set.
- Cap dynamic scale to float32 max.
- no-op when double wrapping with struct.dataclass.
- Allow variable_with_axes to have empty axes when axes is set to an empty tuple.
- Don't create reference cycles among Modules.

0.6.1
-----
- Adds axis_name and axis_index_groups to LayerNorm and GroupNorm. by @copybara-service in [#2402](https://github.com/google/flax/pull/2402)
- Plumb spmd_axis_name through transforms.vmap through to JAX vmap by @copybara-service in [#2398](https://github.com/google/flax/pull/2398)
- Support multiple inputs in flax lifted vjp/custom_vjp by @copybara-service in [#2399](https://github.com/google/flax/pull/2399)
- Improve tabulate by @cgarciae in [#2316](https://github.com/google/flax/pull/2316)
- Add path_aware_map function by @cgarciae in [#2371](https://github.com/google/flax/pull/2371)
- Add static_argnums to nn.checkpoint by @cgarciae in [#2457](https://github.com/google/flax/pull/2457)
- Adding "count_include_pad" argument to flax.linen.pooling.avg_pool by @dslisleedh in [#2451](https://github.com/google/flax/pull/2451)
- Add perturb() to allow capturing intermediate gradients by @IvyZX in [#2476](https://github.com/google/flax/pull/2476)

0.6.0
-----

- Removed deprecated optimizers in `flax.optim` package.
- Moved `flax.optim.dynamic_scale` to `flax.training.dynamic_scale`.
- Switched to using `jax.named_scope` for all profile naming, cut some pointless
  stack traces out.

0.5.3
-----
New features:
- Added `nn.switch` as a lifted version of `jax.lax.switch`.
- Added a method for detecting the use of "init" functions.
- Added checkpointing support for `jax.experimental.GlobalDeviceArray`, a useful array type for multiprocess/multihost computing.
- Added async option to `save_checkpoints()` on single-process scenario.
- Improved documentation pages.

Bug fixes:
- Fixed variable aliasing in put_variable
- Fixed missing passthrough of nn.scan unroll arg
- Fixed the MNIST example

0.5.2
-----
- Fixes missing PyYAML dependency.

0.5.1
-----
New features:
- Added `nn.tabulate` and `Module.tabulate` to generate rich representations of the network structure.

0.5.0
-----
- Added `flax.jax_utils.ad_shard_unpad()` by @lucasb-eyer
- Implemented [default dtype FLIP](https://github.com/google/flax/blob/main/docs/flip/1777-default-dtype.md).
  This means the default dtype is now inferred from inputs and params rather than being hard-coded to float32.
  This is especially useful for dealing with complex numbers because the standard Modules will no longer truncate
  complex numbers to their real component by default. Instead the complex dtype is preserved by default.


Bug fixes:
- Fix support for JAX's experimental_name_stack.

Breaking changes:
- In rare cases the dtype of a layer can change due to [default dtype FLIP](https://github.com/google/flax/blob/main/docs/flip/1777-default-dtype.md). See the "Backward compatibility" section of the proposal for more information.

0.4.2
-----

New features:
- Add lifted conditional `nn.cond`.
- Improved error messages: parameters not found, loading checkpoints.
- Replace `jax.tree_multimap` (deprecated) with `jax.tree.map`.
- Add the "Module Lifecycle" design note.
- Add support for JAX dynamic stack-based named_call

Bug fixes:
- Handle rate==1.0 edgecase in Dropout.
- Fix bug where Linen Module state is reused.
- Bug fixes and generalizations of nn.partitioning API.

0.4.1
-----

New features:
- Added locally-connected (unshared CNN) layer `flax.linen.ConvLocal`.
- Improved seq2seq example: Factored our model and input pipeline code.
- Added Optax update guide and deprecated `flax.optim`.
- Added `sep` argument to `flax.traverse_util.flatten_dict()`.
- Implemented Sequential module, in `flax.linen.combinators`.

0.4.0
------
Breaking changes:
- flax.deprecated.nn is removed. Please pin to flax==0.3.6 if you are still using it.
- PixelCNN++ example is removed. It was not working well on TPU.
- linen Normalization layers no longer downcast double and complex floats tofloat32
  when computing the mean and variance.

New features:
- Added `flax.linen.custom_vjp` for custom derivatives inside a `Module`.
- Add `param_dtype` attribute to standard Linen Modules for specifying parameter dtypes.


0.3.6
------
Breaking changes:
- Move `flax.nn` to `flax.deprecated.nn`.

New features:
- Add experimental checkpoint policy argument. See `flax.linen.checkpoint`
- Add lifted versions of jvp and vjp.
- Add lifted transformation for mapping variables. See `flax.linen.map_variables`.


0.3.5
------

Breaking changes:
 - You can no longer pass an int as the `kernel_size` for a `flax.linen.Conv.
   Instead a type error is raised stating that
   a tuple/list should be provided. Stride and dilation arguments do support broadcasting a single int value now because this is not
   ambigious when the kernel rank is known.
 - `flax.linen.enable_named_call` and `flax.linen.disable_named_call` now work anywhere instead of only affecting Modules constructed after the enable/disable call. Additionally, there is now `flax.linen.override_named_call` that provided a context manager to locally disable/enable named_call.
 - NamedTuples are no longer converted to tuples on assignment to a `linen.Module`.

New features:
 - Flax internal stack frames are now removed from exception state traces.
 - Added `flax.linen.nowrap` to decorate method that should not be transformed
   because they are stateful.
 - Flax no longer uses implicit rank broadcasting. Thus, you can now use Flax with `--jax_numpy_rank_promotion=raise`.

Bugfixes:
 - linen Modules and dataclasses made with `flax.struct.dataclass` or `flax.struct.PyTreeNode` are now correctly recognized as dataclasses by static analysis tools like PyLance. Autocomplete of constructors has been verified to work with VSCode.
 - Fixed a bug in FrozenDict which didn't allow copying dicts with reserved names.
 - Fix the serialization of named tuples. Tuple fields are no longer stored in the state dict and the named tuple class is no longer recreated ([bug](https://github.com/google/flax/issues/1429)).
 - Mixed precision training with float16 now works correctly with the attention layers.
 - auto-generated linen Module `__hash__`, `__eq__`, `__repr__` no longer fail by default on non-init attributes.



0.3.4
------

Possibly breaking changes:
 - When calling `init` the 'intermediates' collection is no longer mutable.
   Therefore, intermediates will no longer be returned from initialization by default.
 - Don't update batch statistics during initialization.
 - When not using any non-determinism (e.g., dropout), it is not longer necessary to specify the `deterministic` argument in `MultiHeadDotProductAttention`.


Other changes:
 - Rewrote various examples to use Optax instead of Flax optimizers (e.g., Imagenet, SST2).
 - Added an NLP text classification example (on the SST-2 dataset) to
   [`examples/sst2`](https://github.com/google/flax/tree/main/examples/sst2).
   that uses a bidirectional LSTM (BiLSTM) to encode the input text.
 - Added `flax.training.train_state` to simplify using Optax optimizers.
 - `mutable` argument is now available on `Module.init` and `Module.init_with_outputs`
 - Bug fix: Correctly handle non-default parameters of Linen Modules with nested inheritance.
 - Expose `dot_product_attention_weights`, allowing access to attention weights.
 - `BatchNorm` instances will behave correctly during init when called multiple times.
 - Added a more extensive "how to contribute" guide in `contributing.md`.
 - Add proper cache behavior for [`lift.jit`](https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.jit.html#flax.linen.jit),
fixing cache misses.
 - Fix bug in Embed layer: make sure it behaves correctly when embedding is np.array.
 - Fix `linen.Module` for deep inheritance chains.
 - Fix bug in DenseGeneral: correctly expand bias to account for batch & noncontracting dimensions.
 - Allow Flax lifted transforms to work on partially applied Modules.
 - Make `MultiOptimizer` use `apply_gradient` instead of `apply_param_gradient`.

0.3.3
------

Possible breaking changes:
 - Bug Fix: Disallow modifying attributes in Modules after they are initialized.
 - Raise an error when saving a checkpoint which has a smaller step than the
   latest checkpoint already saved.
 - MultiOptimizer now rejects the case where multiple sub optimizers update the
   same parameter.

Other changes:
 - Added custom error classes to many Linen errors. See:
   https://flax.readthedocs.io/en/latest/flax.errors.html
 - Adds `Module.bind` for binding variables and RNGs to an interactive Module.
 - Adds `nn.apply` and `nn.init` for transforming arbitrary functions that take a `linen.Module` as their first argument.
 - Add option to overwrite existing checkpoints in `save_checkpoint`.
 - Remove JAX omnistaging check for forward compatibility.
 - Pathlib compatibility for checkpoint paths.
 - `is_leaf` argument in `traverse_util.flatten_dict`

0.3.2
------

`flax.nn` deprecation message no longer appears if you import flax directly.

NOTE: You must now explicitly import `flax.nn` if you want to use the old
      pre-Linen `flax.nn.Module`.

0.3.1
------

Many improvements to Linen, and the old `flax.nn` is officially deprecated!

Notably, there's a clean API for extracting intermediates from modules
defined using `@nn.compact`, a more ergonomic API for using Batch Norm and Dropout in modules
defined using `setup`, support for `MultiOptimizer` with Linen, and multiple safety, performance
and error message improvements.

Possible breaking changes:
 - Call setup lazily. See #938 for motivation and more details.
 - Linen `Module` instances are now frozen after `setup` has been called.
   Previously mutations after setup could be dropped silently. Now the stateless requirement
   is enforced by raising a TypeError in `__setattr__` after `setup`.
 - Pytrees of dicts and lists are transformed into FrozenDict and tuples during
   attribute assignment.
   This avoids undetected submodules and inner state.
 - Bug Fix `flax.core.apply` and `Module.apply`. Now it returns a tuple
   containing the output and a frozen empty
   collection when `mutable` is specified as an empty list.
 - `broadcast_dims` is now an attribute to `Dropout` instead of a `__call__`
   argument.
 - `use_running_average` and `deterministic` no longer have a default. They
   should be passed explicitly
 - Bug Fix `Scope.variable` mutability check, before a variable could only be
   initialized if the 'params' collection was mutable.

Other Improvements:
 - Re-introduced the `lm1b` language modeling example
 - Recognizes batch free inputs in pooling layers. (for use with vmap)
 - Add Adadelta optimizer
 - Fully deprecate all "pre-Linen" `flax.nn` classes and methods.
 - Some Module arguments can now be passed either as dataclass attribute or
   as argument to `__call__`. See [design note](https://flax.readthedocs.io/en/latest/guides/arguments.html)
 - Add `sow` method to `Module` and `capture_intermediates` argument to `Module.apply`.
   See [howto](https://flax.readthedocs.io/en/latest/howtos/extracting_intermediates.html) for usage patterns.
 - Support passing in modules directly as attributes to other modules, and
   deal with them correctly both in top-level modules and in submodules.
 - Don't require the `variable` argument to `Module.apply` to be a FrozenDict
 - Add support for dict/FrozenDict when using `ModelParamTraversal`
   As a result `MultiOptimizer` can be used properly with linen modules.
 - Added OptimizedLSTM: ~33% faster than the original LSTM when using <=1024 units
 - Fix dtype handling for Adam and LAMB optimizers in 64bit mode.
 - Added `is_mutable()` method to `Variable` and `is_mutable_collection()` to `flax.linen.Module`.
 - Add `axis_name` arg to `flax.linen.vmap`
 - Enable broadcast in `flax.linen.scan`
 - Fix behavior when inner module classes were defined in another module
 - Add automatic giant array chunking in msgpack checkpoints.
 - Log info message when a checkpoint is not found in the directory.

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
 - Added various examples in Linen API. See [README.md](https://github.com/google/flax/blob/main/flax/linen/README.md) for more information.
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
