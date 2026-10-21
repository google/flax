import math
import functools
import time
from pathlib import Path

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding
import numpy as np

NUM_LANES = 128


@functools.partial(jax.named_call, name="ragged_decode_kernel")
def ragged_decode_kernel_fwd(
    # prefetch scalars:
    start_ref,  # [bs]
    length_ref,  # [bs]
    # inputs:
    q_ref,  # [heads, head_dim]
    k_ref,  # [block_kv, head_dim]
    v_ref,  # [block_kv, head_dim]
    k_scale_ref,  # [heads]
    v_scale_ref,  # [heads]
    # outputs:
    o_ref,  # [heads, head_dim]
    # scratch memory:
    o_scratch_ref,  # [heads, head_dim]
    l_scratch_ref,  # [heads, TPU_MIN_SIZE]
    m_scratch_ref,  # [heads, TPU_MIN_SIZE]
    # parameters:
    kv_seq_len: int,
    block_kv: int,
    scale: float | None = None,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
):
    mask_value = jnp.finfo(o_scratch_ref.dtype).min
    b, i = pl.program_id(0), pl.program_id(1)
    start, length = start_ref[b], length_ref[b]
    scale = scale if scale is not None else jnp.sqrt(q_ref.shape[-1])

    @pl.when(i == 0)
    def init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    def repeat(x, new_size=q_ref.shape[-1], axis=-1):
        assert new_size % NUM_LANES == 0 and new_size % x.shape[axis] == 0
        return pltpu.repeat(x, new_size // x.shape[axis], axis % x.ndim)

    block_start, block_end = i * block_kv, (i + 1) * block_kv
    should_compute = (start < length) & ((block_start < length) & (block_end >= start))

    @pl.when(should_compute)
    def compute():
        q, k = q_ref[...], k_ref[...]
        if k_scale_ref is not None and not scale_qk_not_k:
            k = k * k_scale_ref[...].astype(jnp.float32).reshape(k.shape[:-1] + (1,)).astype(jnp.bfloat16)
        # don't use transpose in pallas kernels if you can
        # e.g., don't use: qk = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
        contracting_dims = ((1,), (1,))
        batch_dims = ((), ())
        qk = jax.lax.dot_general(q, k, (contracting_dims, batch_dims), preferred_element_type=jnp.float32)
        if k_scale_ref is not None and scale_qk_not_k:
            qk = qk * k_scale_ref[...]

        qk *= scale
        indices = i * block_kv + jax.lax.broadcasted_iota(jnp.int32, qk.shape, dimension=1)
        mask = (indices >= start) & (indices < length)
        qk += jnp.where(mask, 0, mask_value)
        m_curr = repeat(jnp.max(qk, axis=-1)[:, None], NUM_LANES)
        s_curr = jnp.exp(qk - repeat(m_curr, qk.shape[-1]))
        l_curr = repeat(jnp.sum(s_curr, axis=-1)[:, None], NUM_LANES)
        v = v_ref[...]
        if v_scale_ref is not None and not scale_s_not_v:
            v = v * v_scale_ref[...].astype(jnp.float32).reshape(v.shape[:-1] + (1,)).astype(jnp.bfloat16)
        elif v_scale_ref is not None and scale_s_not_v:
            s_curr = s_curr * v_scale_ref[...]
        o_curr = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)

        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        o_prev = o_scratch_ref[...]
        m_next = jnp.maximum(m_prev, m_curr)
        alpha, beta = jnp.exp(m_prev - m_next), jnp.exp(m_curr - m_next)
        l_next = l_prev * alpha + l_curr * beta
        l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

        o_next = (repeat(l_prev * alpha) * o_prev + repeat(beta) * o_curr) / repeat(l_next_safe)
        m_scratch_ref[...] = m_next
        l_scratch_ref[...] = l_next_safe
        o_scratch_ref[...] = o_next

    @pl.when(i == (kv_seq_len // block_kv) - 1)
    def done():
        o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)


def ragged_decode_fwd(
    q: jax.Array,  # [bs, q_heads, head_dim]
    k: jax.Array,  # [bs, kv_seq_len, head_dim]
    v: jax.Array,  # [bs, kv_seq_len, head_dim]
    starts: jax.Array | None = None,  # [bs]
    lengths: jax.Array | None = None,  # [bs]
    k_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    v_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    block_qheads: int = 16,
    block_kv: int = 256,
    scale: float | None = None,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
):
    scale = math.sqrt(q.shape[-1]) if scale is None else scale
    bs, q_heads, head_dim = q.shape
    bs_k, kv_seq_len, head_dim_k = k.shape
    bs_v, kv_seq_len_v, head_dim_v = v.shape
    assert bs == bs_k == bs_v and head_dim == head_dim_k == head_dim_v and kv_seq_len == kv_seq_len_v

    if starts is None:
        starts = jnp.zeros((bs,), dtype=jnp.int32)
    if lengths is None:
        lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

    assert starts.ndim == 1 and starts.size == bs
    assert lengths.ndim == 1 and lengths.size == bs
    assert kv_seq_len % block_kv == 0

    def kv_prefetch_map(b, i, starts_ref, lengths_ref):
        # return b, i, 0
        start, length = starts_ref[b], lengths_ref[b]
        s_idx = i * block_kv
        last_batch, seq_done = b == bs - 1, s_idx > length
        start_next = starts_ref[b + (~last_batch)]
        first_start_i, next_start_i = start // block_kv, start_next // block_kv

        b = jnp.where(seq_done & (~last_batch), b + 1, b)
        i = jnp.where(seq_done, jnp.where(last_batch, i, next_start_i), jnp.maximum(first_start_i, i))
        return b, i, 0

    def kv_scale_prefetch_map(b, i, starts_ref, lengths_ref):
        b_, i_, _ = kv_prefetch_map(b, i, starts_ref, lengths_ref)
        return b_, 0, i_

    in_specs = [
        pl.BlockSpec((None, q_heads, head_dim), lambda b, i, *_: (b, 0, 0)),
        pl.BlockSpec((None, block_kv, head_dim), kv_prefetch_map),
        pl.BlockSpec((None, block_kv, head_dim), kv_prefetch_map),
    ]
    if k_scale is not None:
        in_specs.append(pl.BlockSpec((None, 1, block_kv), kv_scale_prefetch_map))
        k_scale = k_scale.reshape(k_scale.shape[:-1] + (1, k_scale.shape[-1]))
        k_scale = k_scale.astype(jnp.bfloat16)
    else:
        in_specs.append(None)
    if v_scale is not None:
        in_specs.append(pl.BlockSpec((None, 1, block_kv), kv_scale_prefetch_map))
        v_scale = v_scale.reshape(v_scale.shape[:-1] + (1, v_scale.shape[-1]))
        v_scale = v_scale.astype(jnp.bfloat16)
    else:
        in_specs.append(None)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(bs, kv_seq_len // block_kv),
        in_specs=in_specs,
        out_specs=[pl.BlockSpec((None, q_heads, head_dim), lambda b, i, *_: (b, 0, 0))],
        scratch_shapes=[
            pltpu.VMEM((q_heads, head_dim), dtype=jnp.float32),
            pltpu.VMEM((q_heads, NUM_LANES), dtype=jnp.float32),
            pltpu.VMEM((q_heads, NUM_LANES), dtype=jnp.float32),
        ],
    )
    kernel = functools.partial(
        ragged_decode_kernel_fwd,
        block_kv=block_kv,
        kv_seq_len=kv_seq_len,
        scale=scale,
        scale_qk_not_k=scale_qk_not_k,
        scale_s_not_v=scale_s_not_v,
    )
    (attn,) = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=(q,))(starts, lengths, q, k, v, k_scale, v_scale)
    return attn


################################################################################


def ragged_decode_fwd_ref(
    q: jax.Array,  # [bs, q_heads, head_dim]
    k: jax.Array,  # [bs, kv_seq_len, head_dim]
    v: jax.Array,  # [bs, kv_seq_len, head_dim]
    starts: jax.Array | None = None,  # [bs]
    lengths: jax.Array | None = None,  # [bs]
    k_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    v_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    block_qheads: int = 16,
    block_kv: int = 256,
    scale: float | None = None,
):
    scale = math.sqrt(q.shape[-1]) if scale is None else scale
    bs, q_heads, head_dim = q.shape
    bs_k, kv_seq_len, head_dim_k = k.shape
    bs_v, kv_seq_len_v, head_dim_v = v.shape

    if starts is None:
        starts = jnp.zeros((bs,), dtype=jnp.int32)
    if lengths is None:
        lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

    qk = jnp.einsum("bqh,bTh->bqT", q, k) * scale
    if k_scale is not None:
        qk = qk * k_scale[..., None, :]
    indices = jnp.arange(k.shape[-2])
    mask = (indices >= starts[:, None]) & (indices < lengths[:, None])
    qk = jnp.where(mask[:, None, :], qk, jnp.finfo(qk.dtype).min)
    s = jax.nn.softmax(qk, axis=-1) * (jnp.sum(mask, -1) > 0)[:, None, None]
    if v_scale is not None:
        s = s * v_scale[..., None, :]
    return jnp.einsum("bqT,bTh->bqh", s, v)


def _simple_quantize(x: jax.Array, axis: int | tuple[int, ...], scale_dtype=jnp.float16):
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    axis = tuple(z % x.ndim for z in axis)
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
    quant = jnp.round(x / scale).astype(jnp.int8)
    return quant, scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])


def test_main():
    bs, q_heads, kv_heads, kv_seq_len, head_dim = 16, 32, 4, 8192, 128
    # bs, q_heads, kv_heads, kv_seq_len, head_dim = 16, 64, 8, 8192, 128
    # bs, q_heads, kv_heads, kv_seq_len, head_dim = 16, 8, 8, 8192, 128
    dtype = jnp.bfloat16
    # dtype = jnp.float32
    mesh = jax.make_mesh((jax.device_count(),), ("x",))

    @functools.partial(jax.jit, static_argnames=("which", "block_kv", "scale_qk_not_k", "scale_s_not_v"))
    def fn(
        q,
        k,
        v,
        starts,
        lengths,
        which: str = "pallas",
        block_kv: int = 2048,
        scale_qk_not_k: bool = True,
        scale_s_not_v: bool = True,
    ):
        k, k_scale = k if isinstance(k, tuple) else (k, None)
        v, v_scale = v if isinstance(v, tuple) else (v, None)
        q_ = q.reshape(q.shape[:1] + (k.shape[1], -1) + q.shape[2:])
        qkv_spec = P(None, "x", None, None)
        in_specs = 3 * (qkv_spec,) + 2 * (P(),)
        in_specs += (P(None, "x", None) if k_scale is not None else None,)
        in_specs += (P(None, "x", None) if v_scale is not None else None,)
        out_specs = qkv_spec

        @functools.partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
        def _fn(q, k, v, starts, lengths, k_scale, v_scale):
            in_axes = (1, 1, 1, None, None)
            in_axes += (1 if k_scale is not None else None,)
            in_axes += (1 if v_scale is not None else None,)
            if which == "pallas":
                return jax.vmap(
                    lambda *args: ragged_decode_fwd(
                        *args, block_kv=block_kv, scale_qk_not_k=scale_qk_not_k, scale_s_not_v=scale_s_not_v
                    ),
                    in_axes=in_axes,
                    out_axes=1,
                )(q, k, v, starts, lengths, k_scale, v_scale)
            else:
                return jax.vmap(ragged_decode_fwd_ref, in_axes=in_axes, out_axes=1)(
                    q, k, v, starts, lengths, k_scale, v_scale
                )

        return _fn(q_, k, v, starts, lengths, k_scale, v_scale).reshape(q.shape)

    keyit = iter(random.split(random.key(17), 1024))
    q = random.normal(next(keyit), (bs, q_heads, head_dim), dtype=dtype)
    k = random.normal(next(keyit), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype)
    v = random.normal(next(keyit), (bs, kv_heads, kv_seq_len, head_dim), dtype=dtype)
    mesh = jax.make_mesh((jax.device_count(),), P("x"))
    repl_sharding = NamedSharding(mesh, P())
    q = jax.device_put(q / jnp.linalg.norm(q, axis=-1)[..., None], NamedSharding(mesh, P(None, "x", None)))
    k = jax.device_put(k / jnp.linalg.norm(k, axis=-1)[..., None], NamedSharding(mesh, P(None, "x", None, None)))
    v = jax.device_put(v / jnp.linalg.norm(v, axis=-1)[..., None], NamedSharding(mesh, P(None, "x", None, None)))

    k = _simple_quantize(k, axis=-1, scale_dtype=jnp.bfloat16)
    v = _simple_quantize(v, axis=-1, scale_dtype=jnp.bfloat16)

    # starts = random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32)
    # lengths = random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32)
    starts = jnp.zeros((bs,), dtype=jnp.int32)
    lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)
    #lengths = 256 * jnp.ones((bs,), dtype=jnp.int32)
    starts, lengths = jax.device_put(starts, repl_sharding), jax.device_put(lengths, repl_sharding)

    ret = fn(q, k, v, starts, lengths, which="pallas")
    ret_ref = fn(q, k, v, starts, lengths, which="ref")
    err = jnp.linalg.norm((ret - ret_ref).astype(jnp.float32), axis=-1)
    err = err / jnp.linalg.norm(ret_ref.astype(jnp.float32), axis=-1)
    err = jnp.mean(err, -1)
    print(f"{err = }")

    n_trials = 100
    block_kv = 2048
    for which in ["pallas", "ref"]:
        for _ in range(2):  # precompile
            fn(q, k, v, starts, lengths, which=which, block_kv=block_kv).block_until_ready()
        profile_root = Path("~/profiles").expanduser()
        with jax.profiler.trace(str(profile_root / f"ragged_{which}")):
            for scale_qk_not_k in [True, False]:
                for scale_s_not_v in [True, False]:
                    args = q, k, v, starts, lengths
                    kw = dict(
                        which=which, block_kv=block_kv, scale_qk_not_k=scale_qk_not_k, scale_s_not_v=scale_s_not_v
                    )
                    with jax.profiler.TraceAnnotation(f"{scale_qk_not_k = } {scale_s_not_v = }"):
                        fn(*args, **kw).block_until_ready()
                        fn(*args, **kw).block_until_ready()
        t = time.perf_counter()
        for _ in range(n_trials):
            fn(q, k, v, starts, lengths, which=which, block_kv=block_kv).block_until_ready()
        t = time.perf_counter() - t
        print(f"{which = } takes {t / n_trials:.4e} s")

    # from tune_jax import tune
    # fn_ = jax.jit(tune(lambda *args, block_kv=None: fn(*args, which="pallas", block_kv=block_kv),
    #     hyperparams={"block_kv": [128, 256, 1024, 2048, 8192]}, example_args=(q, k, v, starts, lengths)))
    # fn_(q, k, v, starts, lengths).block_until_ready()
    # fn_(q, k, v, starts, lengths).block_until_ready()
    # with jax.profiler.trace(str(profile_root / "ragged_pallas_tuned")):
    #    fn_(q, k, v, starts, lengths).block_until_ready()
    #    fn_(q, k, v, starts, lengths).block_until_ready()
    # t = time.perf_counter()
    # for _ in range(n_trials):
    #    fn_(q, k, v, starts, lengths).block_until_ready()
    # t = time.perf_counter() - t
    # print(f"Pallas takes {t / n_trials:.4e} s")


################################################################################


def test_kv_prefetch_map(block_kv: int = 256):
    bs, kv_seq_len = 16, 8192
    # starts_ref = jnp.zeros(bs, dtype=jnp.int32)
    # lengths_ref = kv_seq_len * jnp.ones(bs, dtype=jnp.int32)
    keyit = iter(random.split(random.key(17), 1024))
    starts_ref = random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32)
    lengths_ref = random.randint(next(keyit), (bs,), 0, kv_seq_len, dtype=jnp.int32)

    def kv_prefetch_map(b, i, starts_ref, lengths_ref):
        start, length = starts_ref[b], lengths_ref[b]
        s_idx = i * block_kv
        last_batch, seq_done = b == bs - 1, s_idx > length
        start_next = starts_ref[b + (~last_batch)]
        first_start_i, next_start_i = start // block_kv, start_next // block_kv

        b = jnp.where(seq_done & (~last_batch), b + 1, b)
        i = jnp.where(seq_done, jnp.where(last_batch, i, next_start_i), jnp.maximum(first_start_i, i))
        return b, i, 0

    b_idx = np.zeros((bs, kv_seq_len // block_kv), dtype=np.int32)
    i_idx = np.zeros((bs, kv_seq_len // block_kv), dtype=np.int32)
    for b in range(bs):
        for i in range(kv_seq_len // block_kv):
            block_start, block_end = i * block_kv, (i + 1) * block_kv
            b_, i_, _ = kv_prefetch_map(b, i, starts_ref, lengths_ref)
            should_compute = (starts_ref[b] < lengths_ref[b]) & (
                (block_start < lengths_ref[b]) & (block_end >= starts_ref[b])
            )
            if should_compute:
                assert b == b_ and i == i_, f"{b = } == {b_ = } and {i = } == {i_ = }"
            b_idx[b, i] = b_
            i_idx[b, i] = i_
    b_flat, i_flat = b_idx.reshape(-1), i_idx.reshape(-1)
    changes = np.mean(1.0 * ((np.diff(b_flat) > 0) | (np.diff(i_flat) > 0)))
    print(f"Changes %: {changes:.2%}")

    with np.printoptions(linewidth=300):
        print(b_idx)
        print("-" * 80)
        print(i_idx)


if __name__ == "__main__":
    test_main()
    # test_kv_prefetch_map()
