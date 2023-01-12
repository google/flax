# The Flax philosophy

In no particular order:

* Library code should be easy to read and understand.
* Prefer duplicating code over a bad abstraction.
* Generally, prefer duplicating code over adding options to functions.
* Comment-driven design: If it's hard to document your code, consider
  changing the design.
* Unit test-driven design: If it's hard to test your code, consider
  changing the design.
* People start projects by copying an existing implementation — make
  base implementations excellent.
* If we expose an abstraction to our developers, we own the mental
  overhead.
* Developer-facing functional programming abstractions confuse some users,
  expose them where the benefit is high.
* "Read the manual" is not an appropriate response to developer confusion.
  The framework should guide developers
  towards good solutions, such as through assertions and error messages.
* An unhelpful error message is a bug.
* "Debugging is twice as hard as writing the code in the first
  place. Therefore, if you write the code as cleverly as possible, you
  are, by definition, not smart enough to debug it." — Brian Kernighan

## Design principles

Flax is a neural network library built on [JAX](https://jax.readthedocs.io) that has been adopted by a
growing set of users, most notably in the JAX submissions for the MLPerf
0.7 benchmark. Our experience over the last year (and many conversations
with users and JAX core devs) has guided a redesign of the API called
[Linen](https://github.com/google/flax/blob/main/flax/linen/README.md) ([`flax.linen`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html)) in response to the following basic design questions.

### How does a neural network library benefit from being built on JAX and leverage JAX’s unique strengths?

The world already has TensorFlow and PyTorch, and there’s little need to
build a clone of either. We believe that the composable
function-transformation approach that JAX takes opens up new frontiers
for making neural net code more maintainable, more scalable and more
performant than existing libraries.While we strive to offer an API
familiar to those experienced with Keras/Sonnet/PyTorch, Linen is
fundamentally a functional system for defining neural nets in JAX. Just
a few examples of what we believe a JAX-targeted library can enable:

- Write models as “single-example” code and introduce batching
  automatically with [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html).
- Automatically handle ragged batches in NLP and other masking issues.
- Create efficient compile-time and runtime models by utilizing
  rematerialized `scan` for massive convolutional networks.
- Remove memory headaches by enabling easy rematerialization,
  reversibility, and model-parallel data sharding.

### How does one interoperate with JAX transformations?

Arguably, the entire point of a neural net library is to offer an
implicit variable management API to save the user from having to
manually thread thousands of variables through a complex tree of
functions. However, JAX operates on pure functions. To handle both
current and future JAX transforms (configured and composed in any way),
Linen Modules are directly “functionalized”, that is, automatically cast
in-place as explicit functions of the form:

$$f \left( v_{in}, x \right) \rightarrow v_{out}, y$$

Where $v_{in}$ is the variable collections and [PRNG](https://jax.readthedocs.io/en/latest/jep/263-prng.html) state used by
the model, $v_{out}$ the mutated output variable collections,
$x$ the input data and $y$ the output data. Applying JAX
transformations then simply reduces to specifying any argument-specific
transform options to the various variable collections and PRNG state.
This unleashes the flexibility and strength of [JAX transformations](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) – for
example, one can achieve either device-parallel training or per-device
ensembling by using [`jax.pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) in different ways, without any explicit
library support. Moreover, **within [Modules](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module)**, we expose lightweight
wrappers around the complex JAX transforms such as [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) and [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
that annotate how each variable collection is to be transformed by JAX.
Importantly, we handle the nontrivial cases of creating new variables
and transformed variables under mapping and loop transforms correctly
for initialization and application.

### How are parameters represented, and how do we handle general “differentiable algorithms” that update stateful variables?

We follow the JAX functional conventions of storing data in “pytrees”:
JAX arrays contained in nested tuples, lists, dictionaries. Because
researchers inevitably manually interact with this data, we use nested
dictionaries with meaningful default keys and offer several utilities
(traversals, etc.) for handling them directly. Linen uses an accelerated
version of a Python frozen dictionary that caches its JAX-flattened form
to speed up [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html)ted function call overheads.

Flax generalizes the operation of a neural net by allowing models to
accept collections of several different “kinds”: parameters, batch-norm
stats, autoregressive caches, debug information, fine-grained
hyperparameters, etc. Each collection is stored in a nested dictionary
of the same structure as the model. Importantly, we do *not* conflate
these various kinds under the single vague rubric of “state”, but keep
different logical types of variables separate that can be treated
differently under JAX transformations and under mutations (e.g. training
vs prediction). Similarly, we allow for multiple separate named PRNG
chains inside [Modules](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module) for separate treatment of randomness for different
applications such as initialization, dropout, sampling, etc.

At every stage the data associated with a neural net is not kept in a
custom object hierarchy, but left in an explicit, Python and JAX native
form that is easy to introspect and modify. Users have utilized this to
map TF and PyTorch checkpoints to Flax, to implement submodel-specific
loss terms, and to perform fast model surgery, etc. For saving this
data, most Flax examples store these nested dictionaries via the
efficient “msgpack” binary format – but as variables are simply Python
dicts, you can use any (non-JAX-aware) serialization library directly.

### How does one interoperate with purely functional JAX code?

To be broadly useful to the JAX ecosystem, users shouldn’t need to
heavily refactor their code in order to add “trainability” for a given
numerical task. _“The library should not get in the way.”_ Utilizing
purely functional code from within Linen is trivial: [Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module)
implementations are just JAX code with named variables. Using Linen
Modules inside otherwise purely functional code can be as simple as
using a single top-level Module transformation to allow initialization
and pure application of any JAX program that might contain various
trainable sections.
