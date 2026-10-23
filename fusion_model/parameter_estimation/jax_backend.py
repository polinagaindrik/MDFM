"""
JAX backend: cost, exact gradient and residual Jacobian of cost_withS
====================================================================

Same model (fusion_model2, constant temperature per experiment), observables (MiBi, MALDI,
NGS), noise models ('absolute' / 'relative'), aggregations (cost_sum_and_geometric_mean,
GaussianNeg2LogLik, GeometricMeanNeg2LogLik, SelectedDataTypes) and s_max penalty as
cost_function.cost_withS / residuals_withS, but written in JAX and solved with diffrax, so that
derivatives are exact (automatic differentiation) instead of finite differences:

  * gradient of the cost (L-BFGS-B):          reverse mode, a few ODE solves per gradient
  * Jacobian of the residuals (least squares): forward mode, all parameter directions in one
                                               vectorized solve
  independent of the number of parameters (finite differences: one solve per parameter).

All experiments are solved together (vmap). Runs on CPU, or on a GPU if JAX finds one
(pip install "jax[cuda12]"). Requires: pip install jax diffrax.

Usage:  jc = JaxCost(calibr_setup); jc.cost(p); jc.value_and_grad(p); jc.residual_jacobian(p)
or through local_optimization(..., backend='jax') / minimize_options={'backend': 'jax'}.
"""
import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # several processes may share a GPU
import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)  # the fit needs double precision

from .cost_function import (cost_sum_and_geometric_mean, relative_mask, _noise_model, SelectedDataTypes,
                            _n_total_points)


# ----------------------------------------------------------------------
# aggregation of the per-data-type sums of squares, in JAX
# ----------------------------------------------------------------------
def _jax_aggregation(agg):
    """JAX function of the list [SSR_ngs, SSR_maldi, SSR_mibi] equal to the numpy aggregation agg."""
    use = None
    if isinstance(agg, SelectedDataTypes):
        agg, use = agg.aggregation_func, list(agg.use)
    name = type(agg).__name__
    if agg is cost_sum_and_geometric_mean:
        def f(ssr):
            return jnp.exp(jnp.mean(jnp.log(jnp.stack(ssr))))
    elif name == "GeometricMeanNeg2LogLik":
        n = [float(v) for v in agg.n_data]
        n_bar = float(agg.n_bar)

        def f(ssr):
            return sum(n_bar * jnp.log(jnp.maximum(s, 1e-300) / nk) for s, nk in zip(ssr, n) if nk > 0)
    elif name == "GaussianNeg2LogLik":
        n = [float(v) for v in agg.n_data]

        def f(ssr):
            return sum(nk * jnp.log(jnp.maximum(s, 1e-300) / nk) for s, nk in zip(ssr, n) if nk > 0)
    else:
        raise NotImplementedError(f"aggregation {name} has no JAX version (supported: cost_sum_and_geometric_mean, "
                                  f"GaussianNeg2LogLik, GeometricMeanNeg2LogLik, SelectedDataTypes of these)")
    if use is None:
        return f
    return lambda ssr: f([s for s, u in zip(ssr, use) if u])


class JaxCost:
    """
    cost_withS and its derivatives for one calibr_setup (data, bounds layout, objective, noise model,
    s_max_penalty). Parameter vector layout as in cost_withS: [x0 (n_exps x n_cl) | ODE | S].
    rtol, atol: tolerances of the adaptive ODE solver (diffrax), solver: 'Tsit5' or 'Kvaerno5' (stiff).
    """
    def __init__(self, calibr_setup, rtol=1e-8, atol=1e-8, solver="Tsit5", max_steps=100000):
        days, (obs_mibi, obs_maldi, obs_ngs) = calibr_setup["data_array"]
        df_mibi = calibr_setup["dfs"][0]
        self.exps = sorted(set(s.split("_")[0] for s in df_mibi.columns))
        self.n_cl = int(np.shape(obs_maldi)[2])
        self.n_media = int(calibr_setup["n_media"])
        self.n_exps = len(self.exps)
        self.days = np.asarray(days, dtype=float)
        self.temps = np.array([float(calibr_setup["exp_temps"][e]) for e in self.exps])
        self.T_x = np.asarray(calibr_setup["T_x"], dtype=float)
        self.noise_model = _noise_model(calibr_setup)
        self.obs = [np.asarray(o, dtype=float) for o in (obs_ngs, obs_maldi, obs_mibi)]
        self.kinds = ["composition", "composition", "counts"]
        if self.noise_model == "relative":
            self.masks = [relative_mask(o, k) for o, k in zip(self.obs, self.kinds)]
        else:
            self.masks = [~np.isnan(o) for o in self.obs]
        self.mibi_max = np.nanmax(self.obs[2], axis=(0, 1))
        self.agg = _jax_aggregation(calibr_setup["aggregation_func"])
        self.penalty_w = float(calibr_setup.get("s_max_penalty") or 0.)
        self.n_total = _n_total_points(calibr_setup["aggregation_func"])
        self.rtol, self.atol, self.max_steps = rtol, atol, max_steps
        self.solver = {"Tsit5": diffrax.Tsit5, "Kvaerno5": diffrax.Kvaerno5, "Dopri8": diffrax.Dopri8}[solver]()
        n_x0 = self.n_cl * self.n_exps
        self.slices = (slice(0, n_x0), slice(n_x0, -self.n_cl * self.n_media))

        self._cost = jax.jit(self._cost_fn)
        self._vg = jax.jit(jax.value_and_grad(self._cost_fn))
        self._res = jax.jit(lambda p: self._residuals_fn(p, adjoint="reverse"))
        self._jac = jax.jit(jax.jacfwd(lambda p: self._residual_vector(p, adjoint="forward")))
        self._res_vec = jax.jit(lambda p: self._residual_vector(p, adjoint="forward"))

    # --- model -------------------------------------------------------------
    def _rhs(self, t, y, args):
        ode, temp = args
        n = self.n_cl
        lambd = 10.0 ** ode[:n] * temp ** ode[n:2 * n]
        alph = ode[2 * n:3 * n] * temp ** ode[3 * n:4 * n]
        nmax = 10.0 ** ode[4 * n] * temp ** ode[4 * n + 1]
        kij = ode[4 * n + 2:].reshape(n, n) * 5e-8
        lag, act, res = y[:n], y[n:2 * n], y[-1]
        b = kij @ act
        growth = alph * res * act / (1.0 + b)
        return jnp.concatenate([-lag * res * lambd, lag * res * lambd + growth,
                                jnp.array([-(1.0 / nmax) * jnp.sum(growth)])])

    def _solve_one(self, x0_log, temp, ode, adjoint):
        n = self.n_cl
        y0 = jnp.concatenate([10.0 ** x0_log, jnp.ones(n + 1)])
        adj = diffrax.ForwardMode() if adjoint == "forward" else diffrax.RecursiveCheckpointAdjoint()
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self._rhs), self.solver, t0=0.0, t1=float(self.days[-1]),
                                  dt0=None, y0=y0, args=(ode, temp), saveat=diffrax.SaveAt(ts=self.days),
                                  stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                                  max_steps=self.max_steps, adjoint=adj, throw=False)
        ys = sol.ys  # (n_days, 2n+1)
        return (ys[:, :n] + ys[:, n:2 * n]).T  # bacterial counts (n_cl, n_days)

    def _observables(self, p, adjoint):
        n, m = self.n_cl, self.n_media
        x0 = p[self.slices[0]].reshape(self.n_exps, n)
        ode = p[self.slices[1]]
        s_x = p[-n * m:].reshape(m, n)
        counts = jax.vmap(lambda x, T: self._solve_one(x, T, ode, adjoint))(x0, jnp.asarray(self.temps))
        # counts: (n_exps, n_cl, n_days)
        f_ngs = jnp.asarray(self.T_x).reshape(1, n, 1) * counts
        m_ngs = f_ngs / jnp.sum(f_ngs, axis=1, keepdims=True)
        f_maldi = s_x.reshape(1, m, n, 1) * counts[:, None, :, :]          # (exps, media, n_cl, days)
        f_maldi = jnp.where(f_maldi <= 0.005, 0.0, f_maldi)
        m_maldi = f_maldi / jnp.sum(f_maldi, axis=2, keepdims=True)
        m_mibi = jnp.sum(s_x.reshape(1, m, n, 1) * counts[:, None, :, :], axis=2)  # (exps, media, days)
        return m_ngs, m_maldi, m_mibi

    # --- residuals ---------------------------------------------------------
    def _residual_groups(self, p, adjoint):
        models = self._observables(p, adjoint)
        out = []
        for obs, mod, mask, kind, scale in zip(self.obs, models, self.masks, self.kinds,
                                               (1.0, 1.0, self.mibi_max)):
            obs_safe = np.where(mask, obs, 1.0)
            if self.noise_model == "relative":
                d = jnp.log(obs_safe) - jnp.log(jnp.maximum(jnp.where(mask, mod, 1.0), 1e-300))
                if kind == "composition":
                    n_present = np.maximum(mask.sum(axis=-2, keepdims=True), 1)
                    d = d - jnp.sum(jnp.where(mask, d, 0.0), axis=-2, keepdims=True) / n_present
                r = d
            else:
                r = (obs_safe - jnp.where(mask, mod, 0.0)) / scale
            out.append(r[mask])
        return out

    def _penalty_residual(self, p):
        s_x = p[-self.n_cl * self.n_media:]
        return jnp.max(s_x) - 1.0

    def _residuals_fn(self, p, adjoint):
        groups = self._residual_groups(p, adjoint)
        if self.penalty_w:
            groups = groups + [jnp.atleast_1d(self._penalty_residual(p))]
        return groups

    def _residual_vector(self, p, adjoint):
        return jnp.concatenate(self._residuals_fn(p, adjoint))

    def _cost_fn(self, p):
        groups = self._residual_groups(p, "reverse")
        c = self.agg([jnp.sum(r ** 2) for r in groups])
        if self.penalty_w:
            p_sq = self._penalty_residual(p) ** 2
            if self.n_total is not None:
                c = c + self.n_total * self.penalty_w * p_sq
            else:
                c = c * jnp.exp(self.penalty_w * p_sq)
        return c

    # --- public API (numpy in, numpy out) -----------------------------------
    def cost(self, p):
        c = float(self._cost(jnp.asarray(p, dtype=float)))
        return c if np.isfinite(c) else 1e10

    def value_and_grad(self, p):
        c, g = self._vg(jnp.asarray(p, dtype=float))
        c, g = float(c), np.asarray(g, dtype=float)
        if not np.isfinite(c) or not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(g)
        return c, g

    def residuals(self, p):
        """Residual groups as residuals_withS (incl. the penalty group): (list of arrays, owners=None)."""
        return [np.asarray(r) for r in self._res(jnp.asarray(p, dtype=float))]

    def residual_vector(self, p):
        return np.asarray(self._res_vec(jnp.asarray(p, dtype=float)))

    def residual_jacobian(self, p):
        """d residual_vector / d p, shape (n_residuals, n_params), exact (forward mode)."""
        return np.asarray(self._jac(jnp.asarray(p, dtype=float)))

    def group_sizes(self):
        return [int(m.sum()) for m in self.masks] + ([1] if self.penalty_w else [])


_CACHE = {}


def cost_withS_jax(param, calibr_setup, jac_spasity=None):
    """cost_withS evaluated with the JAX backend (same value up to the ODE solver accuracy; the JAX solve is
    more accurate). Used instead of cost_withS whenever the fits use backend='jax', so that the optimum and
    all profile points are computed with the same solver."""
    return jax_cost_for(calibr_setup).cost(param)


def _register():
    from . import least_squares_fit as lsf
    from .cost_function import residuals_withS
    lsf.RESIDUAL_FUNCS[cost_withS_jax] = residuals_withS


_register()


def jax_cost_for(calibr_setup, **kwargs):
    """JaxCost cached per calibr_setup object and its bounds / objective / noise model / penalty."""
    key = (id(calibr_setup), id(calibr_setup.get("aggregation_func")), calibr_setup.get("noise_model"),
           calibr_setup.get("s_max_penalty"), tuple(map(tuple, calibr_setup["param_bnds"])),
           tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = JaxCost(calibr_setup, **kwargs)
    return _CACHE[key]
