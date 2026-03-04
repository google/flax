# Tree Mode NNX

Mar 4, 2026,  
Cristian Garcia, Flax Team 

## Motivation

Current NNX APIs allow for general graph structures and graph transformations, while powerful, these capabilities are beyond what JAX offers and supporting them results in both internal complexity, harder to reason about code, and a larger set of APIs a user must learn. We wish to tackle all these issues by simplifying NNX.

## Proposal

To do this we propose the  introduction of **Tree Mode NNX**: a reimplementation of the NNX APIs that only handles trees, assumes referential transparency, and has a more limited support for state updates. Concretely, this means:

* Tree structure assumed and enforced on all APIs (no sharing)  
* Automatic state updates only for Variables in NNX transforms.  
* Modules treated as stateless pytrees (no graph updates).  
* Full JAX transform compatibility (remove StateAxes, DiffState, StateSharding).

## Implementation

Tree mode will be implemented on top of the current APIs by introducing a `graph` argument, when `True` graph support is enabled (current version), when `False` only trees are supported and internals rely on `jax.tree.*` APIs.

```py
def split(..., graph: bool | None = None)
...
def jit(..., graph: bool | None = None)
...
```

If `graph` is not provided the default will be taken from the `nnx_graph_mode` config flag.

```py
import flax

print(f'{flax.config.nnx_graph_mode = }')
flax.config.update('nnx_graph_mode', True/False)
# or env var
# NNX_GRAPH_MODE=true/false
```

The goal will be to have the default value for `nnx_graph_mode` be `False` by default, thus enabling tree mode for new projects. Users that don’t want to migrate can use this flag to make sure their code continues to work in graph mode. For convenience we also provide the `set_graph_mode` function, which has the benefit of being used both as a global switch or a context manager.

```py
nnx.set_graph_mode(False)...with nnx.set_graph_mode(True):
  ...
```

### Tree mode transforms

The tree mode transforms are highly simplified compared to the graph transforms, they are easier to implement and optimize. Given a user function f, most tree mode transforms follow a simple pattern:

```py
@jax_transform
def transformed_f(*args):
  updates, snapshot = updates_and_snapshot(args)
  out = f(*args)
  check_no_aliases(updates, out)
  updates = mask_variable_updates(updates, snapshot)
  return out, updates

def transform_wrapper(*args):
  out, updates = transformed_f(*args)
  apply_variable_updates(args, updates)
  return out
```

The transformed function tracks input Variable `updates`, applies  f, and masks Variable updates (no updates for Variables that didn’t change). It also checks that there are no Variable aliases between the inputs and outputs (no shared references), and returns the user output plus Variable updates. The wrapper function calls the transformed function, applies the Variable updates to the input Variables, and returns the user output.

## Backward compatibility

To make porting and maintaining current graph mode code easier, all graph mode APIs will be accessible via the `nnx.graph` module.

```py
nnx.graph.split = partial(nnx.split, graph=True)
nnx.graph.jit = partial(nnx.jit, graph=True)
...
```

They’ll be implemented as partials that set `graph=True`. The benefit of having these is that it will make fixing entire codebases trivial as one can perform simple find and replace operations e.g:

`nnx.split` → `nnx.graph.split`  
`nnx.jit` → `nnx.graph.jit`  
…
