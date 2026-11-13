"""
JAX backend: cost, exact gradient and residual Jacobian of cost_withS
====================================================================

Same model (fusion_model2, constant temperature per experiment), observables (plate counts, MALDI,
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
                            _n_total_points, maldi_n_of, multinomial_counts, multinomial_deviance_residuals,
                            ngs_dm_of, maldi_dm_of, dm_saturated, dm_deviance_residuals, _dm_rate, ode_params_for_model)


# ----------------------------------------------------------------------
# aggregation of the per-data-type sums of squares, in JAX
# ----------------------------------------------------------------------
def _jax_aggregation(agg):
    """JAX function of the list [SSR_ngs, SSR_maldi, SSR_platecount] equal to the numpy aggregation agg."""
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
    elif name == "KnownNoiseNeg2LogLik":
        n = [float(v) for v in agg.n_data]
        sig = [float(v) for v in agg.sigma]

        def f(ssr):
            return sum(s / sg ** 2 for s, nk, sg in zip(ssr, n, sig) if nk > 0)
    elif name == "MultinomialMaldiNeg2LogLik":
        n = [float(v) for v in agg.n_data]
        dev = list(agg.is_deviance)

        def f(ssr):
            return sum(s if d else nk * jnp.log(jnp.maximum(s, 1e-300) / nk)
                       for s, nk, d in zip(ssr, n, dev) if nk > 0)
    elif name == "GaussianNeg2LogLik":
        n = [float(v) for v in agg.n_data]

        def f(ssr):
            return sum(nk * jnp.log(jnp.maximum(s, 1e-300) / nk) for s, nk in zip(ssr, n) if nk > 0)
    else:
        raise NotImplementedError(f"aggregation {name} has no JAX version (supported: cost_sum_and_geometric_mean, "
                                  f"GaussianNeg2LogLik, GeometricMeanNeg2LogLik, KnownNoiseNeg2LogLik, SelectedDataTypes of these)")
    if use is None:
        return f
    return lambda ssr: f([s for s, u in zip(ssr, use) if u])


class JaxCost:
    """
    cost_withS and its derivatives for one calibr_setup (data, bounds layout, objective, noise model,
    s_max_penalty). Parameter vector layout as in cost_withS: [x0 (n_exps x n_cl) | ODE | S].
    rtol, atol: tolerances of the adaptive ODE solver (diffrax), solver: 'Tsit5' or 'Kvaerno5' (stiff).
    structured_jacobian: residual_jacobian solves the ODE sensitivities only for the directions that reach each
    experiment (its own x0 and the free ODE parameters) and adds S and the other x0 afterwards, without an ODE solve.
    Same Jacobian in the free columns; the columns of fixed ODE parameters are 0 (they are not used by the fits).
    calibr_setup['jax_options'] = {...} sets these options through jax_cost_for.
    """
    def __init__(self, calibr_setup, rtol=1e-8, atol=1e-8, solver="Tsit5", max_steps=100000,
                 structured_jacobian=False):
        days, (obs_platecount, obs_maldi, obs_ngs) = calibr_setup["data_array"]
        df_platecount = calibr_setup["dfs"][0]
        self.exps = sorted(set(s.split("_")[0] for s in df_platecount.columns))
        self.n_cl = int(np.shape(obs_maldi)[2])
        self.n_media = int(calibr_setup["n_media"])
        self.n_exps = len(self.exps)
        self.days = np.asarray(days, dtype=float)
        self.temps = np.array([float(calibr_setup["exp_temps"][e]) for e in self.exps])
        self.T_x = np.asarray(calibr_setup["T_x"], dtype=float)
        self.noise_model = _noise_model(calibr_setup)
        self.obs = [np.asarray(o, dtype=float) for o in (obs_ngs, obs_maldi, obs_platecount)]
        self.kinds = ["composition", "composition", "counts"]
        if self.noise_model == "relative":
            self.masks = [relative_mask(o, k) for o, k in zip(self.obs, self.kinds)]
        else:
            self.masks = [~np.isnan(o) for o in self.obs]
        # MALDI with the multinomial likelihood: all measured entries incl. zeros (undetected species)
        self.maldi_n = maldi_n_of(calibr_setup)
        if self.maldi_n is not None:
            self.masks[1] = np.isfinite(self.obs[1])
            self.maldi_k = np.where(self.masks[1], multinomial_counts(self.obs[1], self.maldi_n), 0.)
        # MALDI with the Dirichlet-multinomial likelihood (maldi_dm): as NGS below
        self.maldi_dm = maldi_dm_of(calibr_setup)
        if self.maldi_dm is not None:
            self.masks[1] = np.isfinite(self.obs[1])
            self.maldi_k = np.where(self.masks[1], multinomial_counts(self.obs[1], self.maldi_dm[0]), 0.)
            self.maldi_sat = dm_saturated(self.maldi_k, *self.maldi_dm)
            self.maldi_c = _dm_rate(*self.maldi_dm)
        # NGS with the Dirichlet-multinomial likelihood: all measured entries incl. zeros (undetected species)
        self.ngs_dm = ngs_dm_of(calibr_setup)
        if self.ngs_dm is not None:
            self.masks[0] = np.isfinite(self.obs[0])
            self.ngs_k = np.where(self.masks[0], multinomial_counts(self.obs[0], self.ngs_dm[0]), 0.)
            self.ngs_sat = dm_saturated(self.ngs_k, *self.ngs_dm)
            self.ngs_c = _dm_rate(*self.ngs_dm)
        self.platecount_max = np.nanmax(self.obs[2], axis=(0, 1))
        self.agg = _jax_aggregation(calibr_setup["aggregation_func"])
        self.penalty_w = float(calibr_setup.get("s_max_penalty") or 0.)
        self.n_total = _n_total_points(calibr_setup["aggregation_func"])
        self.lambda_tref = calibr_setup.get("lambda_tref")  # see cost_function.ode_params_for_model
        self.k_structure = calibr_setup.get("k_structure")  # reduced inhibition matrix, see cost_function
        self.k_groups = None if calibr_setup.get("k_groups") is None else tuple(calibr_setup["k_groups"])
        self.rtol, self.atol, self.max_steps = rtol, atol, max_steps
        self.solver = {"Tsit5": diffrax.Tsit5, "Kvaerno5": diffrax.Kvaerno5, "Dopri8": diffrax.Dopri8}[solver]()
        n_x0 = self.n_cl * self.n_exps
        self.slices = (slice(0, n_x0), slice(n_x0, -self.n_cl * self.n_media))
        # S fixed (cost_direct): parameter vectors without S get calibr_setup['s_x'] appended
        self.n_without_S = n_x0 + 4 * self.n_cl + 2 + self.n_cl * self.n_cl
        s_fixed = calibr_setup.get("s_x")
        self.s_fixed = None if s_fixed is None else np.ravel(np.asarray(s_fixed, dtype=float))

        self._cost = jax.jit(self._cost_fn)
        self._vg = jax.jit(jax.value_and_grad(self._cost_fn))
        self._res = jax.jit(lambda p: self._residuals_fn(p, adjoint="reverse"))
        # free ODE parameters; explicit end, because param_bnds has no S entries when S is fixed (cost_direct)
        bnds_ode = np.array(calibr_setup["param_bnds"], dtype=float)[n_x0:self.n_without_S]
        self.ode_free = np.where(bnds_ode[:, 1] > bnds_ode[:, 0])[0]
        self._jac = jax.jit(self._structured_jacobian if structured_jacobian
                            else jax.jacfwd(lambda p: self._residual_vector(p, adjoint="forward")))
        self._res_vec = jax.jit(lambda p: self._residual_vector(p, adjoint="forward"))

    # --- model -------------------------------------------------------------
    def _rhs(self, t, y, args):
        ode, temp = args
        n = self.n_cl
        lambd = 10.0 ** ode[:n] * temp ** ode[n:2 * n]
        mu = ode[2 * n:3 * n] * temp ** ode[3 * n:4 * n]
        nmax = 10.0 ** ode[4 * n] * temp ** ode[4 * n + 1]
        kij = ode[4 * n + 2:].reshape(n, n) * 5e-8
        lag, act, res = y[:n], y[n:2 * n], y[-1]
        b = kij @ act
        growth = mu * res * act / (1.0 + b)
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
        ode = ode_params_for_model(p[self.slices[1]], {"lambda_tref": self.lambda_tref, "k_structure": self.k_structure, "k_groups": self.k_groups, "n_cl": n}, xp=jnp)
        s_x = p[-n * m:].reshape(m, n)
        counts = jax.vmap(lambda x, T: self._solve_one(x, T, ode, adjoint))(x0, jnp.asarray(self.temps))
        return self._observables_of_counts(counts, s_x)

    def _observables_of_counts(self, counts, s_x):
        """Observables from the bacterial counts (n_exps, n_cl, n_days) and S (n_media, n_cl)."""
        n, m = self.n_cl, self.n_media
        f_ngs = jnp.asarray(self.T_x).reshape(1, n, 1) * counts
        m_ngs = f_ngs / jnp.sum(f_ngs, axis=1, keepdims=True)
        f_maldi = s_x.reshape(1, m, n, 1) * counts[:, None, :, :]          # (exps, media, n_cl, days)
        f_maldi = jnp.where(f_maldi <= 0.005, 0.0, f_maldi)
        m_maldi = f_maldi / jnp.sum(f_maldi, axis=2, keepdims=True)
        m_platecount = jnp.sum(s_x.reshape(1, m, n, 1) * counts[:, None, :, :], axis=2)  # (exps, media, days)
        return m_ngs, m_maldi, m_platecount

    # --- residuals ---------------------------------------------------------
    def _residual_groups(self, p, adjoint):
        return self._groups_of_models(self._observables(p, adjoint))

    def _groups_of_models(self, models):
        out = []
        for i, (obs, mod, mask, kind, scale) in enumerate(zip(self.obs, models, self.masks, self.kinds,
                                                              (1.0, 1.0, self.platecount_max))):
            obs_safe = np.where(mask, obs, 1.0)
            if i == 0 and self.ngs_dm is not None:
                r = dm_deviance_residuals(self.ngs_k, jnp.where(mask, mod, 1.0) * self.ngs_dm[1], self.ngs_sat, self.ngs_c,
                                          xp=jnp, lgamma=jax.scipy.special.gammaln)
            elif i == 1 and self.maldi_dm is not None:
                r = dm_deviance_residuals(self.maldi_k, jnp.where(mask, mod, 1.0) * self.maldi_dm[1], self.maldi_sat,
                                          self.maldi_c, xp=jnp, lgamma=jax.scipy.special.gammaln)
            elif i == 1 and self.maldi_n is not None:
                r = multinomial_deviance_residuals(self.maldi_k, jnp.where(mask, mod, 1.0) * self.maldi_n, xp=jnp)
            elif self.noise_model == "relative":
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

    def _structured_jacobian(self, p):
        """d residual_vector / d p: ODE sensitivities per experiment, then the chain rule through the observables."""
        n, m = self.n_cl, self.n_media
        raw = p[self.slices[1]]
        idx = jnp.asarray(self.ode_free)
        x0 = p[self.slices[0]].reshape(self.n_exps, n)

        def sens(x, temp):
            def f(theta):
                ode = ode_params_for_model(raw.at[idx].set(theta[n:]), {"lambda_tref": self.lambda_tref, "k_structure": self.k_structure, "k_groups": self.k_groups, "n_cl": n}, xp=jnp)
                c = self._solve_one(theta[:n], temp, ode, "forward")
                return c, c
            return jax.jacfwd(f, has_aux=True)(jnp.concatenate([x, raw[idx]]))

        dc, counts = jax.vmap(sens)(x0, jnp.asarray(self.temps))  # dc: (n_exps, n_cl, n_days, n + n_free_ode)
        s_flat = p[-n * m:]

        def res(c, s):
            groups = self._groups_of_models(self._observables_of_counts(c, s.reshape(m, n)))
            if self.penalty_w:
                groups = groups + [jnp.atleast_1d(jnp.max(s) - 1.0)]
            return jnp.concatenate(groups)

        r, lin = jax.linearize(res, counts, s_flat)
        zero_c, zero_s = jnp.zeros_like(counts), jnp.zeros_like(s_flat)
        j_ode = jax.vmap(lambda t: lin(t, zero_s), in_axes=-1, out_axes=-1)(dc[..., n:])
        es, sp = jnp.divmod(jnp.arange(self.n_exps * n), n)
        j_x0 = jax.vmap(lambda e, i: lin(zero_c.at[e].set(dc[e, :, :, i]), zero_s), out_axes=-1)(es, sp)
        j_s = jax.vmap(lambda d: lin(zero_c, d), out_axes=-1)(jnp.eye(s_flat.size))
        n_x0 = self.n_exps * n
        return (jnp.zeros((r.size, p.size)).at[:, :n_x0].set(j_x0).at[:, n_x0 + idx].set(j_ode)
                .at[:, p.size - s_flat.size:].set(j_s))

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
    # p is either the full vector [x0 | ODE | S] (cost_withS) or [x0 | ODE] with S fixed at
    # calibr_setup['s_x'] (cost_direct); derivatives are returned for the entries of p only.
    def _full(self, p):
        p = np.asarray(p, dtype=float)
        if self.s_fixed is not None and len(p) == self.n_without_S:
            return jnp.asarray(np.concatenate([p, self.s_fixed])), len(p)
        return jnp.asarray(p), len(p)

    def cost(self, p):
        c = float(self._cost(self._full(p)[0]))
        return c if np.isfinite(c) else 1e10

    def value_and_grad(self, p):
        pf, n = self._full(p)
        c, g = self._vg(pf)
        c, g = float(c), np.asarray(g, dtype=float)[:n]
        if not np.isfinite(c) or not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(g)
        return c, g

    def residuals(self, p):
        """Residual groups as residuals_withS (incl. the penalty group): (list of arrays, owners=None)."""
        return [np.asarray(r) for r in self._res(self._full(p)[0])]

    def residual_vector(self, p):
        return np.asarray(self._res_vec(self._full(p)[0]))

    def residual_jacobian(self, p):
        """d residual_vector / d p, shape (n_residuals, len(p)), exact (forward mode)."""
        pf, n = self._full(p)
        return np.asarray(self._jac(pf))[:, :n]

    def group_sizes(self):
        return [int(m.sum()) for m in self.masks] + ([1] if self.penalty_w else [])


_CACHE = {}


def cost_withS_jax(param, calibr_setup, jac_spasity=None):
    """cost_withS evaluated with the JAX backend (same value up to the ODE solver accuracy; the JAX solve is
    more accurate). Used instead of cost_withS whenever the fits use backend='jax', so that the optimum and
    all profile points are computed with the same solver."""
    return jax_cost_for(calibr_setup).cost(param)


def cost_direct_jax(param, calibr_setup, jac_spasity=None):
    """cost_direct (S fixed at calibr_setup['s_x'], param = [x0 | ODE]) with the JAX backend."""
    return jax_cost_for(calibr_setup).cost(param)


def _register():
    from . import least_squares_fit as lsf
    from .cost_function import residuals_withS
    lsf.RESIDUAL_FUNCS[cost_withS_jax] = residuals_withS
    lsf.RESIDUAL_FUNCS[cost_direct_jax] = lsf.residuals_direct


_register()


def _hashable(x):
    """Cache key of a count setting: a number, an array of N per composition, or a tuple (N, A)."""
    if isinstance(x, (tuple, list)):
        return tuple(_hashable(v) for v in x)
    if np.ndim(x) > 0:
        a = np.ascontiguousarray(x, dtype=float)
        return (a.shape, a.tobytes())
    return x


def jax_cost_for(calibr_setup, **kwargs):
    """JaxCost cached per calibr_setup object and its bounds / objective / noise model / penalty.
    Options of JaxCost: keyword arguments here, or calibr_setup['jax_options'] (default: none)."""
    kwargs = {**(calibr_setup.get("jax_options") or {}), **kwargs}
    s_x = calibr_setup.get("s_x")
    key = (id(calibr_setup), id(calibr_setup.get("aggregation_func")), calibr_setup.get("noise_model"),
           _hashable(calibr_setup.get("maldi_n")), _hashable(calibr_setup.get("ngs_dm")),
           _hashable(calibr_setup.get("maldi_dm")),
           None if s_x is None else tuple(np.ravel(np.asarray(s_x, dtype=float))),
           calibr_setup.get("s_max_penalty"), tuple(map(tuple, calibr_setup["param_bnds"])),
           tuple(np.ravel(calibr_setup.get("lambda_tref", ()))), calibr_setup.get("k_structure"),
           None if calibr_setup.get("k_groups") is None else tuple(calibr_setup["k_groups"]),
           tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = JaxCost(calibr_setup, **kwargs)
    return _CACHE[key]
