- Start Date: 06.07.21
- FLIP PR: [#0000](https://github.com/google/flax/pull/0000)
- FLIP Issue: [#1436](https://github.com/google/flax/issues/1436)

Authors: @alextp, @bastings, @zpolina (alphabetically)

Table of contents:

- [Summary]
- [Motivation & background]
- [Questions]
- [Alternative approaches]
- [Rollout Plan]
- [Implementation]


# Summary
[Summary]: #summary

We are interested in enabling the use of hooks (‘interceptors’) into calls to Flax modules to add functionality for various purposes. For example, hooks would make it possible to change the input of a Module call or to perturb its output. Hooks can also be used to access or modify the gradients that pass through a Module and expose those.

# Motivation & background
[Motivation & background]: #motivation-and-background

The main motivation for adding hooks support is to avoid forking existing off-the-shelf models and create code duplication when it can be avoided. It is also not always feasible (or easy) to make changes to existing models as many of their layers are defined in upstream libraries not under control of model authors, and it can also reduce readability if lots of optional functionality needs to be hardcoded in.

### Use cases
Capture intermediate outputs of a neural network.
Capture the gradient at each layer (Module) of a neural network.
Perturb/prune models or explain their behavior by exposing gradients. Often the perturbation parameters are learned, and injected into particular layers of the model. For example:
- Differential Masking ([ArXiv](https://arxiv.org/abs/2004.14992)) ([GitHub example with a hook](https://github.com/nicola-decao/diffmask/blob/2385532d4859d6b1608892afe0f827c726aeda75/diffmask/utils/getter_setter.py#L76))
- Parameter space noise for exploration ([ArXiv](https://arxiv.org/pdf/1706.01905.pdf))  ([GitHub example with a hook](https://github.com/iffiX/machin/blob/75b271fb0384a986a6012e17764ceaad65550915/machin/frame/noise/param_space_noise.py#L110))
- SparseML ([Documentation](https://docs.neuralmagic.com/sparseml/))  ([GitHub example with a hook](https://github.com/neuralmagic/sparseml/blob/7c72143a6a8501c7271aa66695be9c0a35c3abb5/src/sparseml/pytorch/optim/mask_pruning.py#L652))
- Class Activation Mapping ([ArXiv](https://arxiv.org/abs/1512.04150))  ([GitHub example with a hook](https://github.com/Project-MONAI/MONAI/blob/12f267c98eabdcd566dff11a3daf931201f04da4/monai/visualize/class_activation_maps.py#L77))

Debugging or temporarily changing or verifying behavior. For example, to change the precision of BatchNorm as done here.

### Interface

We propose the following change to the Flax APIs:
```python
from typing import Protocol, Optional
class Interceptor(Protocol):
  @abstractmethod
  def __call__(self, mdl: Module, fun, *args, **kwargs) -> Any:
    pass

class Module:
  …
  def apply(..., intercept_method : Optional[Interceptor]): ...
```

The interceptor is called on any invocation of any method on Module instances in the subtree of the module in which apply is called. The return value of the interceptor is returned instead of the return value of the original module. Note that the call signature restricts that only one interceptor is usable at any time. Interceptors take precedence over capture_intermediates (that is, the return value of the interceptor is captured).

### Example

Note that the hook is a generalization of `capture_intermediates`, in the sense that capture_intermediates can be implemented as an interceptor as in the following example for the case of `capture_intermediates=True`:

```python
def intercept_method(mdl, fun, *args, **kwargs):
  if fun.__name__ == '__call__':
    y = fun(mdl, *args, **kwargs)
    mdl.sow('intermediates', fun.__name__, y)
    return y
  return fun(mdl, *args, **kwargs)
```
 
It might be convenient for users if capture_intermediates stays, but it could be implemented as an intercept method.
Alternative: a filter-based approach
An alternative to the above function signature would be to adopt the "filters" that `capture_intermediates` uses, and let each hook (in a list) define when it fires based on the filter.

```python
class Bar(nn.Module):
  def test(self, x):
    return x + 1

class Foo(nn.Module):
  @nn.compact
  def __call__(self, x):
    return Bar().test(x) + 1

def intercept_method(mdl, fun, *args, **kwargs):
  # Add 1 to the output of Bar.test().
  if isinstance(mdl, Bar):
    return fun(mdl, *args, **kwargs) + 1
  return fun(mdl, *args, **kwargs)

output = Foo().apply({}, 1, intercept_method=intercept_method)
self.assertEqual(output, 4)
```

### Other frameworks

- [**PyTorch** allows](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_forward_hook#torch.nn.Module.register_forward_hook) a user to register hooks using `register_forward_hook` and `register_forward_pre_hook`. It also has a `register_full_backward_hook` that is called during the backward pass.

- In the JAX ecosystem, Haiku has an `intercept_methods` [function](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.experimental.intercept_methods).

# Questions
[Questions]: #questons

- **Code readability (How do hooks affect readability of the code. Will users be surprised if they use someone else's model that uses hooks?)**. The interceptor shouldn’t be any more confusing than `capture_intermediates`, and given proper documentation and set of examples users shouldn’t have any trouble understanding how to use it.  The interceptor is defined via the `intercept_method` that will unlikely stay unnoticed to the user when they read the code. Arguably this is actually easier code to understand compared to, for example, a forked model that has modified internal methods, that can be easily left unnoticed, and it might not be clear which version of the module you're working with. Additionally given how common this is in PyTorch, many users might be already familiar with the concept.

- **Compatibility of hooks and transforms (nn.scan and nn.vmap/nn.pmap).** Vmap and pmap logic happens at much lower level then the apply or any of the module logic, including our newly added interceptor. For example, by the time the compilation to XLA (needed for pmap) happens there is no difference for the XLA interpreter between hooked and unhooked versions of the module implementation. Vmap is pushing the loop into the primitive operations and vectorizes them, making the code more efficient, which is generally independent of how the nn.Module was constructed. We [implemented a test that](https://github.com/google/flax/pull/1356) shows that a Module with hooks works with vmap and the test passes as expected.

- **Composition of multiple hooks.** 
	- Our current implementation supports only a single hook for simplicity, but that single hook could contain different functionality based on the Module it is operating on.
	- Alternatives are possible here: 
		- We could allow a list of hooks, each with a filter that determines whether to apply it. If two hooks apply to a certain condition, they could be applied in the given order.

- Should it be possible to add hooks at multiple levels (so not only at nn.apply)?
If this is about having hooks inside model code: we can't prevent anyone from doing so (just like we cant prevent someone from calling capture_intermediates inside a model), but it would not result in an error: the "internal" apply call would turn into a pure JAX function and whatever hooks were defined for it will be part of that, and no longer in the Flax call stack.

# Alternative approaches
[Alternative approaches]: #alternative-approaches

- **Monkey patching**.
  - Creates intransparent side effects. The issue with this approach is that if two packages attempt to monkeypatch in this manner, only the changes done by one package would be used. 
  - If someone is using your code and the monkey patching is in place then they might mistakenly keep using the monkey patched code when they add new code inside the same module.
  - Here is the pitfalls of it from Wikipedia [article](https://en.wikipedia.org/wiki/Monkey_patch#Pitfalls).
  - Another reason not to do this is that if a user wants to change the behavior of many different modules, these would all need to be monkey patched, which is much more cumbersome than to specify a hook.


- **Aspect Oriented Programming (AOP)**.
We are pursuing two main goals: not to ask users of existing modules to fork them if they want to change one line in a single sub-module and we also don’t want developers of the modules to have some restrictions imposed on them, when they want their code to be used by many users, especially the ones who want to have small perturbations to the models. AOP solution means asking anyone who wants their code to be more flexible to implement it in the “Aspect” format with predefined “Advices”. That approach is completely not flexible from the users perspective, who would need to rely on the model implementers to have this solution offered to them.


# Rollout Plan
[Rollout plan]: #rollout-plan

- Finalize discussions on this FLIP
- Add additional tests? (nn.scan maybe?)
- Create a doc with examples.
- Update all documentation (including README, Flax guided tour, HOWTOs, ...) to explain how to use the intercept method.
 
# Implementation
[Implementation]: #implementation

We coded a proposal in this PR: https://github.com/google/flax/pull/1356

This will be updated based on discussions on e.g., 
- how to deal with multiple hooks
- let `capture_intermediates` be implemented as a hook in the background (without changing the API for it).

