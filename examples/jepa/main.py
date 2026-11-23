import jax
import jax.numpy as jnp
from collections import namedtuple

import flax

from examples.jepa.data import make_series
from examples.jepa.bfgs import make_bfgs
from examples.jepa.base import MLP, _init_mlp
from examples.jepa.eval import linear_probe


JEPA = namedtuple(
    'JEPA',
    'f_fun make_f_fun stats theta0 theta_tgt0 n_enc encode '
    'momentum var_weight ctx_steps tgt_steps repr_dim '
    'f fg bfgs evaluate dtype enc_module pred_module '
    'to_variables to_params')


def gen_jepa(X_data, ctx_steps=8, tgt_steps=3, repr_dim=4,
             enc_hidden=8, pred_hidden=12,
             var_weight=0.1, momentum=0.9, key=None, seed=0):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X_data = jnp.asarray(X_data)
    dtype = jnp.result_type(X_data.dtype, jnp.float32)
    X_data = X_data.astype(dtype)

    if X_data.ndim != 3:
        raise ValueError("JEPA expects X_data with shape (n, L, d).")
    if X_data.shape[1] < ctx_steps + tgt_steps:
        raise ValueError("sequence length must be >= ctx_steps + tgt_steps.")

    d = int(X_data.shape[2])
    enc_dims = (d, enc_hidden, repr_dim)
    pred_dims = (repr_dim, pred_hidden, tgt_steps * repr_dim)
    k1, k2 = jax.random.split(key)

    enc_module = MLP(layer_dims=enc_dims, dtype=dtype)
    pred_module = MLP(layer_dims=pred_dims, dtype=dtype)
    th_enc, enc_apply, n_enc, _, unravel_enc = _init_mlp(
        enc_module, k1, d, dtype)
    th_pred, pred_apply, _, _, unravel_pred = _init_mlp(
        pred_module, k2, repr_dim, dtype)

    theta0 = jnp.concatenate([th_enc, th_pred])
    theta_tgt0 = jnp.array(th_enc)

    def to_variables(theta, theta_tgt=None):
        enc_vars = unravel_enc(theta[:n_enc])
        pred_vars = unravel_pred(theta[n_enc:])
        if theta_tgt is None:
            return enc_vars, pred_vars
        return enc_vars, pred_vars, unravel_enc(theta_tgt)

    def to_params(theta):
        enc_vars, pred_vars = to_variables(theta)
        return enc_vars['params'], pred_vars['params']

    def encode(theta_enc, frames):
        return enc_apply(theta_enc, frames)

    def _stats(theta, Xb, theta_tgt):
        ctx = Xb[:, :ctx_steps, :]
        tgt = Xb[:, ctx_steps:ctx_steps + tgt_steps, :]

        z_ctx = enc_apply(theta[:n_enc], ctx)                 # (m, T, r)
        z_pool = jnp.mean(z_ctx, axis=1)                      # (m, r)
        z_pred = pred_apply(theta[n_enc:], z_pool).reshape(-1, tgt_steps, repr_dim)
        z_tgt = jax.lax.stop_gradient(enc_apply(theta_tgt, tgt))

        mse = jnp.mean((z_pred - z_tgt) ** 2)
        flat = z_pred.reshape(-1, repr_dim)
        std = jnp.sqrt(jnp.var(flat, axis=0) + 1e-4)
        var_reg = jnp.mean((1.0 - std) ** 2)
        return mse, var_reg, std

    def _loss(theta, Xb, theta_tgt):
        mse, var_reg, _ = _stats(theta, Xb, theta_tgt)
        return mse + var_weight * var_reg

    f_jit = jax.jit(_loss)
    fg_jit = jax.jit(jax.value_and_grad(_loss))
    hess_jit = jax.jit(jax.hessian(_loss))
    stats_jit = jax.jit(_stats)

    @jax.jit
    def evaluate(theta, Xb, X_full, theta_tgt):
        mse, _, std = stats_jit(theta, Xb, theta_tgt)
        return mse, jnp.mean(std), f_jit(theta, X_full, theta_tgt)

    def f_fun(theta, theta_tgt):
        return f_jit(theta, X_data, theta_tgt)

    def make_f_fun(Xb, theta_tgt):
        Xb = jnp.asarray(Xb).astype(dtype)
        return lambda theta: f_jit(theta, Xb, theta_tgt)

    return JEPA(f_fun=f_fun, make_f_fun=make_f_fun, stats=stats_jit,
                theta0=theta0, theta_tgt0=theta_tgt0, n_enc=n_enc,
                encode=encode, momentum=momentum, var_weight=var_weight,
                ctx_steps=ctx_steps, tgt_steps=tgt_steps, repr_dim=repr_dim,
                f=f_jit, fg=fg_jit,
                bfgs=make_bfgs(fg_jit, hess_jit, int(theta0.size)),
                evaluate=evaluate, dtype=dtype,
                enc_module=enc_module, pred_module=pred_module,
                to_variables=to_variables, to_params=to_params)

def train_jepa(model, theta, theta_tgt, X,
               epochs=3, batch_size=64, epsilon=5e-2,
               seed=0, verbose=True, patience=10,
               min_delta=1e-8, monitor='batch'):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")
    X = jnp.asarray(X)
    n = X.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    eps = jnp.asarray(epsilon)
    history = []
    early_stopped = False

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated, no_decrease, prev_full = 0, 0, float('inf')

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            Xb = X[idx]

            theta, l0, l1 = model.bfgs(theta, Xb, theta_tgt, eps)
            mse, stdm, l_full = model.evaluate(theta, Xb, X, theta_tgt)
            l0, l1, mse, stdm, l_full = map(
                float, jax.device_get((l0, l1, mse, stdm, l_full)))

            theta_tgt = (model.momentum * theta_tgt
                         + (1.0 - model.momentum) * theta[:model.n_enc])

            decreased = (l_full < prev_full - min_delta) if monitor == 'full' \
                        else (l1 < l0 - min_delta)
            if monitor == 'full':
                prev_full = l_full
            no_decrease = 0 if decreased else no_decrease + 1
            if abs(l1 - l0) > 1e-12:
                n_updated += 1

            history.append(dict(epoch=epoch, batch=b, n=int(idx.shape[0]),
                                loss_before=l0, loss_after=l1,
                                loss_full=l_full, mse=mse, std=stdm))
            if verbose:
                print('epoch {} | batch {:>2d} | n={:<3d} | jepa loss '
                      '{:.6f} -> {:.6f} | full {:.6f} | pred mse {:.6f} '
                      '| repr std {:.4f}'
                      .format(epoch, b, int(idx.shape[0]), l0, l1, l_full,
                              mse, stdm))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d} | {} consecutive '
                          'batches without loss decrease\n'
                          .format(epoch, b, patience))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full loss {:.6f} | updated {}/{} batches\n'
                  .format(epoch, l_full, n_updated, n_batches))
        if early_stopped:
            break

    return theta, theta_tgt, history


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    SEED, EPOCHS, BATCH_SIZE, EPSILON = 42, 3, 64, 5e-2
    CTX, TGT, D = 8, 3, 4

    X = make_series(n_windows=320, ctx_steps=CTX, tgt_steps=TGT, d=D, seed=7)
    X = (X - X.mean(axis=(0, 1))) / X.std(axis=(0, 1))
    print('data shape (n, L, d):', X.shape)

    model = gen_jepa(X, ctx_steps=CTX, tgt_steps=TGT, repr_dim=4,
                     enc_hidden=8, pred_hidden=12,
                     var_weight=0.1, momentum=0.99, seed=SEED)
    theta, theta_tgt = model.theta0, model.theta_tgt0
    print('flax {} | online params: {} (encoder {} + predictor {}) '
          '| target params: {} | dtype {}'
          .format(flax.__version__, theta.size, model.n_enc,
                  theta.size - model.n_enc, theta_tgt.size, model.dtype))

    enc_vars, pred_vars, tgt_vars = model.to_variables(theta, theta_tgt)
    enc_params, pred_params = model.to_params(theta)
    print('encoder params pytree:',
          jax.tree_util.tree_map(lambda a: a.shape, enc_params))
    print('predictor params pytree:',
          jax.tree_util.tree_map(lambda a: a.shape, pred_params))
    print('standard flax apply shape:',
          model.enc_module.apply(enc_vars, X[:2, :CTX, :]).shape)

    f0_init, g_init = model.fg(theta, jnp.asarray(X[:BATCH_SIZE]), theta_tgt)
    print('init full loss: {:.6f} | grad shape: {}'
          .format(float(model.f_fun(theta, theta_tgt)), g_init.shape))

    theta, theta_tgt, hist = train_jepa(
        model, theta, theta_tgt, X,
        epochs=EPOCHS, batch_size=BATCH_SIZE, epsilon=EPSILON,
        seed=SEED, verbose=True, patience=10, monitor='batch')

    print('final full loss: {:.6f}'.format(float(model.f_fun(theta, theta_tgt))))
    per_epoch = {}
    for h in hist:
        per_epoch[h['epoch']] = h['loss_full']
    print('epoch-end full loss:',
          [round(per_epoch[e], 6) for e in sorted(per_epoch)])

    r2_repr, r2_rand, r2_raw = linear_probe(model, theta, model.theta0, X)
    print('linear probe R^2 (predict next {} frames) | trained repr: {:.4f} '
          '| random-init repr: {:.4f} | raw last frame: {:.4f}'
          .format(TGT, r2_repr, r2_rand, r2_raw))
