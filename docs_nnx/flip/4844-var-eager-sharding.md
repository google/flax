- Start Date: 2025-09-12
- FLIP PR: [#4844](https://github.com/google/flax/pull/4844)

# Summary
[summary]: #summary

Simplify the creation of sharded NNX models. When a sharding annotation is provided, all `nnx.Variable` creation will **require a mesh context** and automatically be sharded as annotated.

# Motivation

To create a sharded model, user should only need to do this:

```python
mesh = jax.make_mesh(((2, 4)), ("data", "model"))
with jax.set_mesh(mesh):
  model = YourModelWithShardingAnnotations()
```

Instead of the current boilerplate combo of `nnx.jit`, `nnx.get_partition_spec`, `with_sharding_constraint` and `nnx.update`:

```python
@nnx.jit
def create_sharded_model():
  model = YourModelWithShardingAnnotations() # Unsharded at this moment.
  state = nnx.state(model)                   # The model's state, a pure pytree.
  pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(model, sharded_state)           # The model is sharded now!
  return model

mesh = jax.make_mesh(((2, 4)), ("data", "model"))
with mesh:
  sharded_model = create_sharded_model()
```

# Backward compatibility

User can turn off this feature in two ways:

### Global config flag

Run `flax.config.update('flax_always_shard_variable', False)` before running any NNX model initialization.

### Variable-specific flag

Create a specific variable with metadata `eager_sharding=False`, such as: `nnx.Param(..., eager_sharding=False)`.


# Implementation
[implementation]: #implementation

When an `nnx.Variable` is created, check for the metadata `sharding_names`, and if present, check if under a valid global mesh context of was supplied with a valid mesh. If no, throw error; if yes, call `jax.lax.with_sharding_constraint` to apply sharding constraint on the value.

Note that this only works in auto sharding mode. User should use JAX-level APIs to annotate shardings for explicit mode.

