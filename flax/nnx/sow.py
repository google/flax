"""A `sow` transform, implemented as a jaxpr interpreter.

`sow(value, *, name)` tags a pytree during tracing. `capture(f)` runs `f` and
returns `(f_output, collected)` where `collected` maps each name to a tuple of
the pytrees sown under it, in call order.

`sow` carries a JAX effect so it survives DCE even when its result is unused
(so you can log metrics with a bare `sow(loss, name="loss")` statement), and a
trivial identity lowering so a sow-containing function still runs under plain
`jax.jit` (sows are silently discarded when not harvested).

Control flow: sows inside `jit`/`closed_call` are collected inline. A sow inside
a `scan` fires once per iteration and is collected as a single pytree stacked
along the scan axis (leading dim = length). A sow inside `cond` yields the taken
branch's value; every branch must sow the same names/shapes (enforced by jax's
own branch-matching check). A sow inside a `remat` is collected by inlining the
body when it is undifferentiated (which is how jax lowers it anyway) and, once
differentiated, by rebinding `remat_p` with the sown values as extra outputs so
the optimization barrier survives — note this makes them residuals, i.e. you keep
in memory exactly what you harvest. `while_loop` is unsupported (a dynamic trip
count can't produce a fixed-size collection).
"""
from contextvars import ContextVar
from functools import partial

import jax
from jax._src import core, effects, pjit
from jax._src.custom_derivatives import custom_jvp_call_p, custom_vjp_call_p
from jax.interpreters import ad, batching, mlir
from jax.tree_util import tree_flatten, tree_structure, tree_unflatten


class SowEffect(effects.Effect):
  pass


sow_effect = SowEffect()
effects.lowerable_effects.add_type(SowEffect)
effects.control_flow_allowed_effects.add_type(SowEffect)
effects.remat_allowed_effects.add_type(SowEffect)

sow_p = core.Primitive("sow")
sow_p.multiple_results = True
sow_p.def_impl(lambda *xs, **__: list(xs))
sow_p.def_effectful_abstract_eval(lambda *avals, **__: (list(avals), {sow_effect}))
mlir.register_lowering(sow_p, lambda ctx, *args, **__: args)  # identity; sow discarded
# sow is a linear identity, so all transform rules are pass-throughs.
ad.deflinear2(sow_p, lambda cts, *_, **__: cts)
batching.primitive_batchers[sow_p] = lambda args, dims, **params: (
    sow_p.bind(*args, **params), dims)

_prefix: ContextVar[str] = ContextVar("sow_prefix", default="")

def sow(value, *, name):
  """Tag `value` (any pytree) with `name`, any hashable. Identity outside
  `capture`. `capture_prefix` only applies to string names; a non-string name is
  taken as already-absolute (nnx's `Module.sow` keys on a tuple that carries its
  own module path)."""
  leaves, treedef = tree_flatten(value)
  if isinstance(name, str):
    name = _prefix.get() + name
  out = sow_p.bind(*leaves, name=name, tree=treedef)
  return tree_unflatten(treedef, out)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _perturb(value, name):
  return value


_perturb.defvjp(lambda value, name: (value, None),
                lambda name, _res, g: (sow(g, name=name),))


def perturb(value, *, name):
  """Identity on the forward pass; sows the incoming cotangent under `name` on
  the backward pass. Drop it into a function to harvest a gradient:

      def loss(x):
        return jnp.sum(perturb(x, name="grad_x") ** 2)

      grad, collected = capture(jax.grad(loss))(x)
      # collected["grad_x"][0] holds d(loss)/d(perturbed value)

  Like `sow`, but for cotangents instead of forward values — the custom_vjp's
  backward rule stages the sow into the gradient jaxpr at trace time, so
  `capture(jax.grad(...))` picks it up (see test_sow_in_custom_vjp_backward)."""
  return _perturb(value, name)

# Call primitives whose sub-jaxpr we inline (sows collected directly). For
# custom_jvp/vjp we inline the primal call_jaxpr; their differentiation rules
# already staged any backward-pass sow into the enclosing jaxpr at trace time.
_CALL_PRIMS = {
    pjit.jit_p: "jaxpr",
    core.closed_call_p: "call_jaxpr",
    custom_jvp_call_p: "call_jaxpr",
    custom_vjp_call_p: "call_jaxpr",
}

def _contains_sow(jaxpr: core.Jaxpr) -> bool:
  return any(
      eqn.primitive is sow_p or any(map(_contains_sow, core.jaxprs_in_params(eqn.params)))
      for eqn in jaxpr.eqns
  )

def _merge(dst: dict, src: dict) -> None:
  for name, vals in src.items():
    dst.setdefault(name, []).extend(vals)

def _flatten_collected(collected: dict):
  """collected {name: [pytree,...]} -> (flat leaves, meta) in a stable order."""
  leaves, meta = [], []
  for name in sorted(collected, key=str):  # names need not be orderable, just stable
    for pytree in collected[name]:
      lvs, treedef = tree_flatten(pytree)
      meta.append((name, treedef, len(lvs)))
      leaves.extend(lvs)
  return leaves, meta

def _unflatten_collected(leaves, meta) -> dict:
  out: dict = {}
  i = 0
  for name, treedef, n in meta:
    out.setdefault(name, []).append(tree_unflatten(treedef, leaves[i:i + n]))
    i += n
  return out

def _run(jaxpr: core.Jaxpr, consts, *args):
  """Interpret a jaxpr, returning (outputs, {name: [pytree, ...]})."""
  env: dict = {}
  collected: dict = {}

  def read(v):
    return v.val if isinstance(v, core.Literal) else env[v]

  env.update(zip(jaxpr.constvars, consts))
  env.update(zip(jaxpr.invars, args))

  for eqn in jaxpr.eqns:
    invals = [read(v) for v in eqn.invars]
    prim = eqn.primitive
    if prim is sow_p:
      pytree = tree_unflatten(eqn.params["tree"], invals)
      collected.setdefault(eqn.params["name"], []).append(pytree)
      outs = invals  # pass through
    elif prim in _CALL_PRIMS:
      sub = eqn.params[_CALL_PRIMS[prim]]  # ClosedJaxpr or open Jaxpr
      jx, cs = (sub.jaxpr, sub.consts) if isinstance(sub, core.ClosedJaxpr) else (sub, [])
      outs, sub_coll = _run(jx, cs, *invals)
      _merge(collected, sub_coll)
    else:
      if any(_contains_sow(sj) for sj in core.jaxprs_in_params(eqn.params)):
        raise NotImplementedError(
            f"sow inside {prim} is not supported (handled: top level, jit, "
            f"closed_call, custom_jvp/vjp, scan, cond, remat; not: while_loop)"
        )
      ans = prim.bind(*invals, **eqn.params)
      outs = ans if prim.multiple_results else [ans]
    env.update(zip(eqn.outvars, outs))

  return [read(v) for v in jaxpr.outvars], collected

def capture(f):
  """Transform `f` to return `(f_output, {name: (pytree, ...)})`."""
  def wrapped(*args, **kwargs):
    cj, out_shape = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
    flat_args = tree_flatten((args, kwargs))[0]
    out_flat, collected = _run(cj.jaxpr, cj.consts, *flat_args)
    out = tree_unflatten(tree_structure(out_shape), out_flat)
    return out, {name: tuple(vals) for name, vals in collected.items()}

  return wrapped
