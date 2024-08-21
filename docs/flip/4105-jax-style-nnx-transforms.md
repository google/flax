# JAX-style NNX Transforms

- Authors: Cristian Garcia, Anselm Levskaya
- Date: Jun/2024
- FLIP PR: #4107
- Status: Implementing

## Motivation

NNX allows users to utilize Modules at the top level due to their eager initialization and self-contained state. This naturally leads users to want to use them with transforms and soon start playing with NNX transforms. Since NNX Modules resemble PyTrees in that they contain Arrays, new users often attempt to apply JAX conventions, for example:

```py
@nnx.vmap(in_axes=(1, 0))
def f(m1: Module, m2: Module):
  ...
```

However, this can be misleading. Currently, NNX transforms follow Linen's convention of treating input Modules as a single unit (all Modules are split together to preserve shared references) and provide APIs for transforming that State separately. The previous example effectively translates to:

```py
# this is what is really happening
@nnx.vmap(in_axes=(IGNORE, IGNORE), state_axes={BatchStat: None, ...: 0})
def f(m1: Module, m2: Module):
  ...
```

Note that `IGNORE` is not a real symbol, but represents the fact that any value placed here won't affect the outcome, as Modules are replaced by empty PyTree placeholders (similar to `None`). The `state_axes` parameter controls how the State is vectorized through a mapping of high-level `Filter`s to their desired axes. In this example, `...` (ellipsis) is a filter that accepts everything, so by default all States are vectorized on the 0th axis.

To express their original intention, users must resort to more complex custom filters that guess the index of each Module in the monolith. While this is straightforward in simple cases, users generally need to calculate the index (Modules appear in the order specified by `jax.tree.leaves` over the `args`):

```py
select_m1 = lambda path, value: path[0] == 0
select_m2 = lambda path, value: path[0] == 1

# To select modules individually, you must create a filter (which can be tricky)
@nnx.vmap(state_axes={select_m1: 1, select_m2: 0})
def f(m1: Module, m2: Module):
  ...
```

## What if JAX conventions Just Worked™?

This proposal aims to align NNX transforms with user's expectations based on their JAX experience, making the syntax work as intuitively as possible. The original example would function **as if** `m1` and `m2` were PyTrees vectorized in axes `1` and `0` respectively:

```py
@nnx.vmap(in_axes=(1, 0))
def f(m1: Module, m2: Module):
  ...
```

The primary advantage of this approach is that for `vmap` and `scan`, we could eliminate the `state_axes` and `split_rngs` arguments, relying solely on the `in_axes` API. This syntax alone would likely suffice for 80-90% of use cases, as users tend to manage state in predictable ways.

### The Lift symbols

To enable more fine-grained state control within each Module, we introduce the `Lift` API. By using special types containing State Filters in place of a tree prefix, state lifting can now be done **structurally**. This allows different Filters to be applied to different Modules in the arguments without the need for complex path-based filters. Ideally, each transform would support its own Lift type, adding the desired behavior through existing JAX APIs.

For example, in `vmap`, we could allow `StateAxes` instances (vmap's Lift type) to be accepted by `in/out_axes` to control how substates are handled by mapping state `Filter`s to an axis specifier:

```py
state_axes = StateAxes({Param: 1, BatchStat: None})

@nnx.vmap(in_axes=(state_axes, 0))
def f(m1: Module, m2: Module):
  ...
```

In this case, `m1`'s `Param`s are vectorized in axis `1` while its `BatchStat`s are broadcasted, and `m2`'s entire state is vectorized in axis `0`.

For `nnx.grad`, we could allow `DiffState` to be used in the `argnums` parameter to specify both the position of the argument to be differentiated and a Filter specifying the differentiable State of the Module:

```py
grads = nnx.grad(loss_fn, argnums=(DiffState(0, LoRAParam),))(model, x, y)
```

## Rng Handling

To simplify RNG state handling, we propose removing the separate `split_rngs` parameter in `vmap` and `scan`. Instead, we suggest introducing a new `nnx.split_rngs` API that would manage RNG handling before and after the transformation. This approach provides more explicit control to the user and aligns better with JAX transform behavior.

## Consistent Aliasing

To ensure the correctness of transformations with objects that obey reference semantics, we must enforce consistent lifting/lowering specifications for all aliases of a reference. Transforms must adhere to two rules:

1. All aliases of a reference must receive the **exact same** lifting/lowering specification.
2. Captured references are not allowed on the output of transformed functions.

For example:

```py
@nnx.vmap(in_axes=(m1_axes, m2_axes, m1_axes), out_axes=m2_axes)
def f(m1, m2, m1_alias):
  return m2

m2 = f(m1, m2, m1)
```

Here, `m1` has two input aliases as it is passed as the first and third input to `f`, but this is acceptable because `m1_axes` is assigned to both in `in_axes`. `m2` is passed as the second input and has an output alias, which is also acceptable because `m2_axes` is assigned in both `in_axes` and `out_axes`.

Let's examine some examples of programs that should be **rejected** based on these criteria:

### Inconsistent input aliases

Consider a function with two arguments `m1` and `m2` being vectorized in axis `0` and `1` respectively. Passing the same Module as both arguments would be inconsistent:

```py
@nnx.vmap(in_axes=(0, 1))
def f(m1: Module, m2: Module):
  ...

f(m, m)  # This should be rejected
```

### Inconsistent input / output aliases

Now consider an identity function `g` under `vmap` with `in_axes=0` and `out_axes=1`. In JAX, this would result in transposing the arrays in the inputs:

```py
@nnx.vmap(in_axes=0, out_axes=1)
def g(m: Module):
  return m
```

While this appears correct, in NNX this behavior is not well-defined because shared mutable references behave as auxiliary outputs. Under the hood, `g` is converted into a function that has the inputs as an extra first output, and `out_axes` is set to the same values as `in_axes` for that output:

```py
@nnx.vmap(in_axes=0, out_axes=(0, 1))
def g_real(m: Module):
  return m, m
```

This return structure reveals an inconsistency: we're attempting to lower `m` with both `out_axes=0` and `out_axes=1`.

### Inconsistent aliases in nested structures

Similar issues can arise in less obvious cases, such as when `m` is contained within another structure:

```py
@nnx.vmap(in_axes=0, out_axes=1)
def f(m: Module):
  return SomeModule(m)
```

This means we must traverse the entire graph of both inputs and outputs to check for consistent assignments. The same problem occurs when passing shared reference inputs/outputs with different specifications:

```py
shared = Shared()
m1, m2 = Foo(shared), Foo(shared)

@nnx.vmap(in_axes=(0, 1))
def f(m1, m2):  # shared is passed through both
  ...
```

### Captured Modules cannot be outputs

Finally, let's consider the second consistent aliasing rule, which states that captured Modules cannot be outputs. The main issue here is that NNX needs to split all input references together to track changes, but captured Modules bypass this process. Treating them as new references would result in **implicit cloning**:

```py
m = SomeModule()

@nnx.vmap(out_axes=0, axis_size=5)
def f():
  return m

assert m is not f()  # implicit cloning
```

To preserve reference identity, we must disallow captured Modules as outputs. In practice, we can detect captured Modules using the trace level context machinery used to restrict stateful updates on Modules from a different level.

## Recap

In this document, we have:

* Discussed issues with the current implementation that make it unintuitive for JAX users.
* Proposed refactoring NNX transforms to allow users to use regular JAX semantics when interacting with objects, removing extra arguments introduced by NNX transforms.
* Introduced the use of Lift types in JAX APIs to compensate for the lack of a "prefix" notion in NNX objects, enabling independent lifting of Module substates.
* Proposed a new `nnx.split_rngs` API to replace the `split_rngs` arguments in `vmap` and `scan`, making RNG handling an explicit operation and giving users more control.
* Analyzed edge cases resulting from aliasing shared mutable references and proposed enforcing **consistent aliasing** on all transforms with semantics over the inputs.