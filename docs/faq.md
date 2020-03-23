# The Flax FAQ

This FAQ is compiled from various questions asked on github issues, mailing list, chatrooms, and from personal conversations. 

---

**Question: How to initialize a Flax model?**

**Context:** Modules seem to have just one function `apply` that takes as 
arguments both the input and additional parameters (e.g. size of some layer). 
How to initialize and use a model? The MNIST example has a CNN with no 
additional parameters, so initialization happens by only calling 
`init_by_shape`.

**Answer:** You'll typically use a `init_by_shape` call to init models as it 
runs `apply` in "init mode", which doesn't perform any actual computation -
it just traces all the shapes and inits submodules.

---

**Question: The Model abstraction is very lightweight, is it necessary?**

**Context:** Why not having a separate function for defining `params` and a pair 
of `init`/`apply` methods?

**Answer:** Modules have an `init` and `call` function that can be used if you
need use them. So you can do things like `Dense.call(params, X, ...)`. `Model` just
wraps parameters and the `apply` function together in a way that's JAX-aware,
so you can just pass a model instance into JAX-transformed functions
without thinking about `static_argnums`.

--- 

**Question: Does Flax name the tensors like Tensorflow? 
(e.g. `orthogonal_conv/kernel:0`)?**

**Answer:** Parameters are just numpy arrays so they don't really have a name. 
We do use a path notation in a few places which is based on the nested dict 
structure that models use:
```python
{
  'conv': {
    'kernel': ... # path is /conv/kernel
}
```

---

**Question: How to create multiple nested modules?**

**Answer:** You can directly write submodules and just nest them in a higher 
module - Flax takes care of submodule initialization for you based on tracing 
shapes, etc. for initializing them.  For instance, the Conv and Batchnorm layers 
inside the resnet model are submodules themselves.

---

**Question: How to control which devices Flax uses to execute code?**

**Answer:** `@jax.jit` has a device argument and will use the the first TPU device by 
default, if available. `pmap` by default runs on all accelerators, so you have 
to use `pmap` if you want to use all available TPU cores.

---

**Question: How does Flax know which devices are available? 
Do we pass a target somewhere?**

**Answer:** The devices used by Flax are determined by JAX and the transformations 
that you use. By default JAX executes op-by-op and uses an accelerator when 
available. Using `@jax.jit(..., device=...)` you can control which device is 
used. If you want to parallelize over multiple device you should use 
`@jax.pmap(..., devices=...)`.

--- 

**Question: What is `@functools.partial()` for?**

**Answer:** Example: `functools.partial(jax.jit, static_argnums=(1, 2, 3))`. 
`jax.jit` is a function decorator that causes compilation of the underlying 
JAX code. We might like to just say `@jax.jit(static_argnums=(1,2,3))`, but if 
we were to use it this way `jax.jit` would have to be a function that returns a 
function wrapper, rather than being a function wrapper itself. 
The way around this is to use python's built-in functional tool 
`functools.partial` which applies a subset of a functions arguments 
(`static_args` in this case) and allows you then to apply to the rest.  

Another way of writing this would be:
```python
def create_model(...):
   ....blah blah function code....
# redefine it as jitted function:
create_model = jax.jit(create_model, static_argnums=(1, 2, 3))
```
This partial application is typical JAX functional style.

The models.ResNet.partial call is a little different but similar in spirit - it's just setting the model hyperparameters before we initialize it below.  (The reason we jit the model creation is that under the hood there's a lot of initialization calculations going on that involve a lot of RNG number crunching, so we want it compiled.)

--- 

**Question: What is `nn.stateful()` for?**

**Answer:** Example: `with nn.stateful() as init_state`. 

Flax uses with scopes to manage state (like batchnorm statistics) and JAX's 
functional RNGs for stochastic layers (like dropout).  It's a bit weird 
compared to pytorch state and RNG but pretty simple to use at the top level and 
helps to deal with transforming model code to a pure functional form.  
In JAX state has to be considered explicitly, you're ultimately defining a 
function like y, `new_state = model_function(x, old_state, params)` -- Flax 
in fact uses scoping tricks to handle state management because most of the 
time you're dealing with models and layers that don't have any state, 
so it keeps most end-user code simpler.

---

**Question: Is there something similar to `tf.keras.Sequential`?**

**Answer:** We don't have a Sequential combinator in flax at the moment, 
so you have to manually write the chain of layer function calls. 
In Flax sequential(Foo1, Foo2, Foo3) just becomes something like:
```python
class Foo(nn.Module):
  def apply(self, x):
    x = Foo1(x)
    x = Foo2(x)
    x = Foo3(x)
    return x
```
(or can be made into 3 lines by applying all the function calls directly in 
one line). The benefit is that if you then want to add something between 
Foo2 and Foo3 you don't need to rewrite the module -- you can just "hack away".

---

**Question: When should I use Module.shared() and when not?**

**Answer:** Iterating over a submodule in a module function may lead to errors 
if Module.shared() is not used:

```python
class Test(nn.Module):
  def apply(self, x):
    return nn.Dense(x, features=5, name='dense')
  
  @nn.module_method
  def apply2(self, x):
    for _ in range(5):
      x = nn.Dense(x, features=5, name='dense')
    return x
```

The api guards you against accidentally sharing parameters. So you want to do 
something like this:
```python
@nn.module_method
  def apply2(self, x):
    dense = nn.Dense.shared(features=5, name='dense')
    for _ in range(5):
      x = dense(x)
    return x
```

---

**Question: How to perform computations for a Flax module only occasionally?**

**Context:** I am trying to create a new Flax module that instantiates a 
standard kernel but then performs some transformations to map it onto the
 orthogonal/stiefel manifold. My initial implementation computed the 
 transformation within the apply function, but I want to perform these 
 transformations only occasionally (upon initial creation and after every 
 gradient update). How would you enable this within Flax? 

**Answer:** Can you do the orthogonal projection directly in your train loop? 
You could simply fetch the parameters `optimizer.target.params` and apply the 
projection there, then replace `optimizer.target` with the new parameters. 
You could write something like this in your train function: 

```python
orthogonalized_model = jax.tree_map(orthogonalize_param, optimizer.target)
optimizer = optimizer.replace(target=orthogonalized_model)
```

---

**Question: How to get a submodule's parameters?**

**Answer:**

Assuming you're doing this within a module,

```python
nn.Embed(.., name='vocab')
embedding_matrix = self.get_param('vocab')['embedding']
```

Or alternatively
```python
class MyEmbed(nn.Embed):
  @nn.module_method
  def get_embedding(self):
    return self.get_param('embedding')

embed_layer = MyEmbed.shared()
...
embed_layer.get_embedding()
```

---

**Question: How to filter parameters and pass them to specific optimizers?**
 
**Answer:** You can use the `MultiOptimizer` to filter parameters and pass them 
to specific optimizers. Here is the example from `MultiOptimizer`:

```python
kernels = optim.ModelParamTraversal(lambda path, _: 'kernel' in path)
biases = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
kernel_opt = optim.Momentum(learning_rate=0.01)
bias_opt = optim.Momentum(learning_rate=0.1)
opt_def = MultiOptimizer((kernels, kernel_opt), (biases, bias_opt))
optimizer = opt_def.create(model)
```

--- 

**Question: Why does Batchnorm not expose the dtypes like other layers?**

**Answer:**  We haven't fully figured out yet how we want to deal with reduced 
precision in cases like batchnorm.

--- 

**Question: Do Flax models have static or dynamic shapes?**

**Answer:** They have static shapes. A model is created from an initial shape, 
and it is not directly possible to change this. (This is a limitation of XLA
and thus of JAX)

--- 
