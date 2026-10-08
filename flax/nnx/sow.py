"""A `sow` transform, implemented as a jaxpr interpreter.

`sow(value, *, name)` tags a pytree during tracing. `capture(f)` runs `f` and
returns `(f_output, collected)` where `collected` maps each name to a tuple of
the pytrees sown under it, in call order.

`sow` carries a JAX effect so it survives DCE even when its result is unused
(so you can log metrics with a bare `sow(loss, name="loss")` statement), and a
trivial identity lowering so a sow-containing function still runs under plain
`jax.jit` (sows are silently discarded when not harvested).
"""
from contextvars import ContextVar
from functools import partial

import jax
from jax import lax
from jax._src import core, effects, pjit
from jax._src.custom_derivatives import custom_jvp_call_p, custom_vjp_call_p
from jax._src.lax.control_flow.loops import scan_p
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

_prefix: ContextVar[tuple] = ContextVar("sow_prefix", default=())

def sow(value, *, name):
  """Tag `value` (any pytree) with `name`, any hashable. Identity outside
  `capture`."""
  leaves, treedef = tree_flatten(value)
  name = (*_prefix.get(), name)
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

def _contains_sow(jaxpr: core.Jaxpr, cache: dict) -> bool:
  hit = cache.get(id(jaxpr))
  if hit is None:
    hit = cache[id(jaxpr)] = any(
        eqn.primitive is sow_p
        or any(_contains_sow(sj, cache) for sj in core.jaxprs_in_params(eqn.params))
        for eqn in jaxpr.eqns
    )
  return hit

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

# Dispatch rules: prim -> rule(context, eqn, invals) -> (outs, collected).
_DISPATCH: dict = {}

def _call_rule(param, context, eqn, invals):
  """Inline a call primitive's sub-jaxpr (sows collected directly)."""
  sub = eqn.params[param]  # ClosedJaxpr or open Jaxpr
  jx, cs = (sub.jaxpr, sub.consts) if isinstance(sub, core.ClosedJaxpr) else (sub, [])
  return _run(jx, cs, *invals, context=context)

_DISPATCH[pjit.jit_p] = partial(_call_rule, "jaxpr")
_DISPATCH[core.closed_call_p] = partial(_call_rule, "call_jaxpr")

# For custom_jvp/vjp we inline the primal call_jaxpr; their differentiation
# rules already staged any backward-pass sow into the enclosing jaxpr at trace
# time.
_DISPATCH[custom_jvp_call_p] = partial(_call_rule, "call_jaxpr")
_DISPATCH[custom_vjp_call_p] = partial(_call_rule, "call_jaxpr")

def _sow_rule(context, eqn, invals):
  pytree = tree_unflatten(eqn.params["tree"], invals)
  return invals, {eqn.params["name"]: [pytree]}  # invals pass through

_DISPATCH[sow_p] = _sow_rule

def _scan_rule(context, eqn, invals):
  if not _contains_sow(eqn.params["jaxpr"].jaxpr, context["contains_sow_cache"]):
    return _bind(context, eqn, invals)
  return _scan_with_sow(context, eqn.params, invals)

_DISPATCH[scan_p] = _scan_rule

def _bind(context, eqn, invals):
  prim = eqn.primitive
  if any(_contains_sow(sj, context["contains_sow_cache"])
         for sj in core.jaxprs_in_params(eqn.params)):
    handled = ", ".join(sorted(p.name for p in _DISPATCH if p is not sow_p))
    raise NotImplementedError(
        f"sow inside {prim} is not supported (handled: top level, {handled})"
    )
  ans = prim.bind(*invals, **eqn.params)
  return (ans if prim.multiple_results else [ans]), {}

def _run(jaxpr: core.Jaxpr, consts, *args, context: dict | None = None):
  """Interpret a jaxpr, returning (outputs, {name: [pytree, ...]}).

  `context` carries per-interpretation state shared by dispatch rules (e.g.
  "contains_sow_cache"); it is created on the outermost call."""
  if context is None:
    context = {"contains_sow_cache": {}}
  env: dict = {}
  collected: dict = {}

  def read(v):
    return v.val if isinstance(v, core.Literal) else env[v]

  env.update(zip(jaxpr.constvars, consts))
  env.update(zip(jaxpr.invars, args))

  for eqn in jaxpr.eqns:
    invals = [read(v) for v in eqn.invars]
    outs, sub_coll = _DISPATCH.get(eqn.primitive, _bind)(context, eqn, invals)
    _merge(collected, sub_coll)
    env.update(zip(eqn.outvars, outs))

  return [read(v) for v in jaxpr.outvars], collected

def _scan_with_sow(context, params, invals):
  """Sows inside a scan body become extra `ys`, so scan stacks them along the
  scan axis. Each per-iteration sow appears as one stacked pytree entry."""
  body = params["jaxpr"]
  # ft_in.update(args).unpack() splits operands into (consts, carry, xs) — the
  # same call scan's own impl uses (loops.py `_scan_impl`).
  consts, init, xs = map(list, params["ft_in"].update(invals).unpack())
  ncarry = len(init)
  meta_box: list = []

  def new_body(carry, x):
    outs, coll = _run(body.jaxpr, body.consts, *consts, *carry, *x, context=context)
    leaves, meta = _flatten_collected(coll)
    meta_box.append(meta)
    return outs[:ncarry], (outs[ncarry:], leaves)

  final_carry, (stacked_ys, stacked_leaves) = lax.scan(
      new_body, init, xs, length=params["length"],
      reverse=params["reverse"], unroll=params["unroll"])
  return [*final_carry, *stacked_ys], _unflatten_collected(stacked_leaves, meta_box[0])

def _nest(collected: dict) -> dict:
  """{(a, b): [v, ...]} -> {a: {b: (v, ...)}}."""
  out: dict = {}
  for path, vals in collected.items():
    d = out
    for key in path[:-1]:
      d = d.setdefault(key, {})
      if not isinstance(d, dict):
        raise ValueError(f"sow name {path} conflicts with a shorter name")
    if path[-1] in d:
      raise ValueError(f"sow name {path} conflicts with a longer name")
    d[path[-1]] = tuple(vals)
  return out

def capture(f):
  """Transform `f` to return `(f_output, nested)`, where `nested` maps each
  component of a sown name to a sub-dict, bottoming out in tuples of pytrees."""
  def wrapped(*args, **kwargs):
    cj, out_shape = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
    flat_args = tree_flatten((args, kwargs))[0]
    out_flat, collected = _run(cj.jaxpr, cj.consts, *flat_args)
    out = tree_unflatten(tree_structure(out_shape), out_flat)
    return out, _nest(collected)

  return wrapped
