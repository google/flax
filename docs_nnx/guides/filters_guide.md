---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# Using `Filter`s

Flax NNX uses [`Filter`s](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/filterlib.html) extensively as a way to create [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) groups in APIs, such as [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split), [`nnx.state()`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.state), and many of the [Flax NNX transformations (transforms)](https://flax.readthedocs.io/en/latest/guides/jax_and_nnx_transforms.html).

In this guide you will learn:

* What is a `Filter`?
* Why are types, such as [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) or [`nnx.BatchStat`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.BatchStat), treated as `Filter`s?
* What is the `Filter` domain specific language (DSL)?
* How is [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) grouped / filtered?

In the following example [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) and [`nnx.BatchStat`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.BatchStat) are used as `Filter`s to split the model into two groups: one with the parameters and the other with the batch statistics:

```{code-cell} ipython3
from flax import nnx

class Foo(nnx.Module):
  def __init__(self):
    self.a = nnx.Param(0)
    self.b = nnx.BatchStat(True)

foo = Foo()

graphdef, params, batch_stats = nnx.split(foo, nnx.Param, nnx.BatchStat)

print(f'{params = }')
print(f'{batch_stats = }')
```

Let's dive deeper into `Filter`s.

+++

## The `Filter` Protocol

In general, Flax `Filter`s are predicate functions of the form:

```python

(path: tuple[Key, ...], value: Any) -> bool

```

where:

- `Key` is a hashable and comparable type;
- `path` is a tuple of `Key`s representing the path to the value in a nested structure; and
- `value` is the value at the path.

The function returns `True` if the value should be included in the group, and `False` otherwise.

Types are not functions of this form. They are treated as `Filter`s because, as you will learn in the next section, types and some other literals are converted to _predicates_. For example, [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) is roughly converted to a predicate like this:

```{code-cell} ipython3
def is_param(path, value) -> bool:
  return isinstance(value, nnx.Param) or (
    hasattr(value, 'type') and issubclass(value.type, nnx.Param)
  )

print(f'{is_param((), nnx.Param(0)) = }')
print(f'{is_param((), nnx.VariableState(type=nnx.Param, value=0)) = }')
```

Such function matches any value that is an instance of [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) or any value that has a `type` attribute that is a subclass of [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param). Internally Flax NNX uses `OfType` which defines a callable of this form for a given type:

```{code-cell} ipython3
is_param = nnx.OfType(nnx.Param)

print(f'{is_param((), nnx.Param(0)) = }')
print(f'{is_param((), nnx.VariableState(type=nnx.Param, value=0)) = }')
```

## The `Filter` DSL

To help users avoid having to create functions mentioned in the previous section, Flax NNX exposes a small domain specific language ([DSL](https://en.wikipedia.org/wiki/Domain-specific_language)), formalized as the [`nnx.filterlib.Filter`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/filterlib.html) type. The `Filter` DSL allows users to pass types, booleans, ellipsis, tuples/lists, etc, and converts them to the appropriate predicate internally.

Here is a list of all the callable `Filter`s included in Flax NNX, and their corresponding DSL literals (when available):


| Literal | Callable | Description |
|--------|----------------------|-------------|
| `...` or `True` | `Everything()` | Matches all values |
| `None` or `False` | `Nothing()` | Matches no values |
| `type` | `OfType(type)` | Matches values that are instances of `type` or have a `type` attribute that is an instance of `type` |
| | `PathContains(key)` | Matches values that have an associated `path` that contains the given `key` |
| `'{filter}'` <span style="color:gray">str</span> | `WithTag('{filter}')` | Matches values that have string `tag` attribute equal to `'{filter}'`. Used by `RngKey` and `RngCount`. |
| `(*filters)` <span style="color:gray">tuple</span> or `[*filters]` <span style="color:gray">list</span> | `Any(*filters)` | Matches values that match any of the inner `filters` |
| | `All(*filters)` | Matches values that match all of the inner `filters` |
| | `Not(filter)` | Matches values that do not match the inner `filter` |


Let's check out the DSL in action by using [`nnx.vmap`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.vmap) as an example. Consider the following:

1) You want to vectorize all parameters;
2) Apply `'dropout'` `Rng(Keys|Counts)` on the `0`th axis; and
3) Broadcast the rest.

To do this, you can use the following `Filter`s to define a `nnx.StateAxes` object that you can pass to `nnx.vmap`'s `in_axes` to specify how the `model`'s various sub-states should be vectorized:

```{code-cell} ipython3
state_axes = nnx.StateAxes({(nnx.Param, 'dropout'): 0, ...: None})

@nnx.vmap(in_axes=(state_axes, 0))
def forward(model, x):
  ...
```

Here `(nnx.Param, 'dropout')` expands to `Any(OfType(nnx.Param), WithTag('dropout'))` and `...` expands to `Everything()`.

If you wish to manually convert literal into a predicate, you can use [`nnx.filterlib.to_predicate`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/filterlib.html#flax.nnx.filterlib.to_predicate):

```{code-cell} ipython3
is_param = nnx.filterlib.to_predicate(nnx.Param)
everything = nnx.filterlib.to_predicate(...)
nothing = nnx.filterlib.to_predicate(False)
params_or_dropout = nnx.filterlib.to_predicate((nnx.Param, 'dropout'))

print(f'{is_param = }')
print(f'{everything = }')
print(f'{nothing = }')
print(f'{params_or_dropout = }')
```

## Grouping `State`s

With the knowledge of `Filter`s from previous sections at hand, let's learn how to roughly implement [`nnx.split`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split). Here are the key ideas:

* Use `nnx.graph.flatten` to get the [`GraphDef`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.GraphDef) and [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State) representation of the node.
* Convert all the `Filter`s to predicates.
* Use `State.flat_state` to get the flat representation of the state.
* Traverse all the `(path, value)` pairs in the flat state and group them according to the predicates.
* Use `State.from_flat_state` to convert the flat states to nested [`nnx.State`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/state.html#flax.nnx.State)s.

```{code-cell} ipython3
from typing import Any
KeyPath = tuple[nnx.graph.Key, ...]

def split(node, *filters):
  graphdef, state = nnx.graph.flatten(node)
  predicates = [nnx.filterlib.to_predicate(f) for f in filters]
  flat_states: list[dict[KeyPath, Any]] = [{} for p in predicates]

  for path, value in state.flat_state().items():
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i][path] = value
        break
    else:
      raise ValueError(f'No filter matched {path = } {value = }')

  states: tuple[nnx.GraphState, ...] = tuple(
    nnx.State.from_flat_path(flat_state) for flat_state in flat_states
  )
  return graphdef, *states

# Let's test it.
foo = Foo()

graphdef, params, batch_stats = split(foo, nnx.Param, nnx.BatchStat)

print(f'{params = }')
print(f'{batch_stats = }')
```

**Note:*** It's very important to know that **filtering is order-dependent**. The first `Filter` that matches a value will keep it, and therefore you should place more specific `Filter`s before more general `Filter`s.

For example, as demonstrated below, if you:

1) Create a `SpecialParam` type that is a subclass of [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param), and a `Bar` object (subclassing [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)) that contains both types of parameters; and
2) Try to split the [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param)s before the `SpecialParam`s

then all the values will be placed in the [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param) group, and the `SpecialParam` group will be empty because all `SpecialParam`s are also [`nnx.Param`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Param)s:

```{code-cell} ipython3
class SpecialParam(nnx.Param):
  pass

class Bar(nnx.Module):
  def __init__(self):
    self.a = nnx.Param(0)
    self.b = SpecialParam(0)

bar = Bar()

graphdef, params, special_params = split(bar, nnx.Param, SpecialParam) # wrong!
print(f'{params = }')
print(f'{special_params = }')
```

And reversing the order will ensure that the `SpecialParam` are captured first:

```{code-cell} ipython3
graphdef, special_params, params = split(bar, SpecialParam, nnx.Param) # correct!
print(f'{params = }')
print(f'{special_params = }')
```
