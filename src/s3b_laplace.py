#!/usr/bin/env python3
"""Stage 3b: MAP + Laplace inference for the hierarchical mixture, in JAX.

Same generative model, priors, and variants as s3_fit.py / model.py, but the
posterior is maximized (L-BFGS on the unconstrained parametrization) and the
covariance taken from the curvature at the mode. Each fit takes minutes, so
the full variant grid is cheap; the NUTS fits on the cluster remain as a
cross-check of this approximation on the primary specification.

  python s3b_laplace.py --phase fulltext --variant primary

Writes data/pi_<tag>.csv (same schema as the MCMC path) and
data/laplace_<tag>.json with the mode, key scalars, and diagnostics.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, quarter_label
from s3_fit import VARIANTS, load, neutral_drift_offset, SECTIONS
import model as M


# ---------------------------------------------------------------- densities
def nb_logpmf(k, mu, phi):
    return (gammaln(k + phi) - gammaln(phi) - gammaln(k + 1.0)
            + phi * jnp.log(phi / (phi + mu)) + k * jnp.log(mu / (phi + mu)))


def norm_lp(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - jnp.log(sd)


def halfnorm_lp_pos(x, sd):
    """log N+(x; 0, sd) for x > 0 (constant dropped)."""
    return -0.5 * (x / sd) ** 2


# ---------------------------------------------------------------- parameters
# Unconstrained vector layout (n_free = number of post-boundary quarters,
# n_c topics):
#   beta0, log_sd_b, b_raw[n_c-1]            (b sums to zero)
#   slope, curve
#   log_phi0, log_phi1
#   eta0, log_sd_eta, eta_inc[n_free-1]
#   log_sd_v, v_raw[n_c-1]
#   m0, log_sd_m, m_inc_raw[n_free-1]        (monotone: inc = softplus)
#   log_sd_u, u_raw[n_c-1]
#   rho_a, rho_b


def declared_anchor(df, K, free_from, n_t):
    """Per-quarter log excess of declared papers over the pre-2020 background.

    This is the spam-filter calibration: the assisted component's marker rate
    is pinned to the papers that SAY they used a model, with a smoothed,
    shrunk trajectory. Quarters with no declared papers inherit the nearest
    later value (declarations only begin in 2023). Returns log(anchor_t) for
    the free quarters, floored at log(1.05) so the components never merge
    exactly.
    """
    pre = df.year < 2020
    r_bg = K[pre.values].sum() / df.L.values[pre.values].sum()
    dcl = df.declared.values == 1
    logratio = np.full(n_t - free_from, np.nan)
    for i in range(n_t - free_from):
        m = dcl & (df.q.values == free_from + i)
        Lsum = df.L.values[m].sum()
        if m.sum() >= 5 and Lsum > 0:
            r = K[m].sum() / Lsum
            logratio[i] = np.log(max(r / r_bg, 1.05))
    # smooth with a 3-quarter rolling mean, then back/forward fill
    s = pd.Series(logratio).rolling(3, center=True, min_periods=1).mean()
    s = s.bfill().ffill()
    if s.isna().all():
        s[:] = np.log(1.05)
    return s.values


def build_logpost(data, drift, monotone, use_disclosure, gamma, free_from,
                  eta_anchor=None, anchor_sd=0.2):
    K = jnp.asarray(data["K"], dtype=jnp.float64)
    logL = jnp.log(jnp.asarray(data["L"], dtype=jnp.float64))
    t = np.asarray(data["t"], dtype=int)
    c = np.asarray(data["c"], dtype=int)
    declared = jnp.asarray(data["declared"], dtype=jnp.float64)
    n_t, n_c = int(data["n_quarters"]), int(data["n_topics"])
    n_free = n_t - free_from
    tq = np.arange(n_t, dtype=float)
    tc = (tq - free_from) / 4.0
    drift_off = jnp.asarray(data["drift_offset"], dtype=jnp.float64)

    pre = t < free_from
    idx_pre, idx_fr = np.where(pre)[0], np.where(~pre)[0]
    tf = jnp.asarray(t[idx_fr] - free_from)
    cf = jnp.asarray(c[idx_fr])
    c_pre = jnp.asarray(c[idx_pre])
    t_pre = jnp.asarray(t[idx_pre])
    K_pre, K_fr = K[idx_pre], K[idx_fr]
    logL_pre, logL_fr = logL[idx_pre], logL[idx_fr]
    D_fr = declared[idx_fr]

    sizes = dict(n_c=n_c, n_free=n_free)
    cuts, tot = {}, 0
    for name, ln in [("beta0", 1), ("log_sd_b", 1), ("b_raw", n_c - 1),
                     ("slope", 1), ("curve", 1),
                     ("log_phi0", 1), ("log_phi1", 1),
                     ("eta0", 1), ("log_sd_eta", 1), ("eta_inc", n_free - 1),
                     ("log_sd_v", 1), ("v_raw", n_c - 1),
                     ("m0", 1), ("log_sd_m", 1), ("m_inc_raw", n_free - 1),
                     ("log_sd_u", 1), ("u_raw", n_c - 1),
                     ("rho_a", 1), ("rho_b", 1)]:
        cuts[name] = (tot, tot + ln)
        tot += ln
    n_par = tot

    def unpack(th):
        p = {k: th[a] if b - a == 1 else th[a:b] for k, (a, b) in cuts.items()}
        return p

    def zerosum(raw):
        v = jnp.concatenate([raw, -jnp.sum(raw, keepdims=True)])
        return v

    def logpost(th):
        p = unpack(th)
        lp = 0.0

        beta0 = p["beta0"]
        sd_b = jnp.exp(p["log_sd_b"])
        b = zerosum(p["b_raw"]) * sd_b
        phi0 = jnp.exp(p["log_phi0"])
        phi1 = jnp.exp(p["log_phi1"])

        # priors (transform Jacobians for the log-scales included via +log sd)
        lp += norm_lp(beta0, -7.0, 3.0)
        lp += halfnorm_lp_pos(sd_b, 1.0) + p["log_sd_b"]
        lp += jnp.sum(norm_lp(p["b_raw"], 0.0, 1.0))
        lp += norm_lp(p["slope"], 0.0, 0.5) + norm_lp(p["curve"], 0.0, 0.05)
        lp += halfnorm_lp_pos(phi0, 5.0) + p["log_phi0"]
        lp += halfnorm_lp_pos(phi1, 5.0) + p["log_phi1"]

        # background drift
        g_lin = p["slope"] * tc + p["curve"] * tc ** 2
        if drift == "linear":
            g0 = g_lin
        elif drift == "frozen":
            g0 = jnp.where(tq >= free_from, 0.0, g_lin)
        elif drift == "tracked":
            g0 = jnp.where(tq >= free_from, drift_off, g_lin)
        else:
            raise ValueError(drift)

        # assisted excess: anchored to the declared papers (spam-filter
        # calibration) when an anchor is supplied, else a free random walk
        sd_eta = jnp.exp(p["log_sd_eta"])
        lp += halfnorm_lp_pos(sd_eta, 0.4) + p["log_sd_eta"]
        eta = jnp.concatenate([p["eta0"][None],
                               p["eta0"] + jnp.cumsum(p["eta_inc"] * sd_eta)])
        if eta_anchor is not None:
            anc = jnp.asarray(eta_anchor)
            lp += jnp.sum(norm_lp(eta, anc, anchor_sd))
            lp += norm_lp(p["eta0"], anc[0], anchor_sd)
            lp += jnp.sum(norm_lp(p["eta_inc"], 0.0, 1.0))
        else:
            lp += norm_lp(p["eta0"], jnp.log(0.7), 1.0)
            lp += jnp.sum(norm_lp(p["eta_inc"], 0.0, 1.0))
        delta = jnp.exp(eta)

        sd_v = jnp.exp(p["log_sd_v"])
        lp += halfnorm_lp_pos(sd_v, 0.3) + p["log_sd_v"]
        lp += jnp.sum(norm_lp(p["v_raw"], 0.0, 1.0))
        v = zerosum(p["v_raw"]) * sd_v

        # prevalence walk
        sd_m = jnp.exp(p["log_sd_m"])
        lp += halfnorm_lp_pos(sd_m, 0.6) + p["log_sd_m"]
        lp += norm_lp(p["m0"], -4.0, 2.0)
        if monotone:
            inc = jax.nn.softplus(p["m_inc_raw"]) * sd_m
            # HalfNormal(1) prior on softplus(raw) with its Jacobian
            sp = jax.nn.softplus(p["m_inc_raw"])
            lp += jnp.sum(halfnorm_lp_pos(sp, 1.0)
                          + jnp.log(jax.nn.sigmoid(p["m_inc_raw"])))
        else:
            inc = p["m_inc_raw"] * sd_m
            lp += jnp.sum(norm_lp(p["m_inc_raw"], 0.0, 1.0))
        m_t = jnp.concatenate([p["m0"][None], p["m0"] + jnp.cumsum(inc)])

        sd_u = jnp.exp(p["log_sd_u"])
        lp += halfnorm_lp_pos(sd_u, 0.5) + p["log_sd_u"]
        lp += jnp.sum(norm_lp(p["u_raw"], 0.0, 1.0))
        u = zerosum(p["u_raw"]) * sd_u

        lp += norm_lp(p["rho_a"], -4.0, 2.0) + norm_lp(p["rho_b"], 0.0, 1.0)

        # ------------------------ likelihood: known-negative era
        mu0_pre = jnp.exp(logL_pre + beta0 + b[c_pre] + g0[t_pre])
        lp += jnp.sum(nb_logpmf(K_pre, mu0_pre, phi0))

        # ------------------------ likelihood: free era (mixture, PU)
        mu0 = jnp.exp(logL_fr + beta0 + b[cf] + g0[free_from + tf])
        mu1 = mu0 * delta[tf] * jnp.exp(v[cf])
        lp0 = nb_logpmf(K_fr, mu0, phi0)
        lp1 = nb_logpmf(K_fr, mu1, phi1)

        logit_pi = m_t[tf] + u[cf]
        log_pi = -jax.nn.softplus(-logit_pi)
        log_1mpi = -jax.nn.softplus(logit_pi)

        if use_disclosure:
            logit_rho = p["rho_a"] + p["rho_b"] * (tf / 4.0)
            log_rho = -jax.nn.softplus(-logit_rho)
            log_1mrho = -jax.nn.softplus(logit_rho)
            lp1d = nb_logpmf(K_fr, mu1 * np.exp(gamma), phi1)
            ll_d1 = log_pi + log_rho + lp1d
            ll_d0 = jnp.logaddexp(log_1mpi + lp0, log_pi + log_1mrho + lp1)
            ll = jnp.where(D_fr == 1.0, ll_d1, ll_d0)
        else:
            ll = jnp.logaddexp(log_1mpi + lp0, log_pi + lp1)
        lp += jnp.sum(ll)
        return -lp                     # negative log posterior for minimize

    return logpost, unpack, cuts, n_par, sizes


def fit(args):
    spec = dict(VARIANTS[args.variant])
    basket = spec.pop("basket")
    free_from = spec.pop("free_from", M.FREE_FROM)
    drift = spec.pop("drift")
    monotone = spec.pop("monotone")
    use_disc = spec.pop("use_disclosure", True)
    gamma = spec.pop("gamma", 0.0)

    df, K = load(args.phase, basket, section=args.section)
    n_q = int(df.q.max()) + 1
    data = {"K": K, "L": df.L.values, "t": df.q.values, "c": df.topic.values,
            "declared": df.get("declared", pd.Series(np.zeros(len(df)))).values,
            "n_quarters": n_q, "n_topics": int(df.topic.max()) + 1,
            "drift_offset": neutral_drift_offset(df, n_q, free_from,
                                                 phase=args.phase,
                                                 section=args.section)}
    print(f"phase={args.phase} variant={args.variant} n={len(df)} "
          f"drift={drift} monotone={monotone}", flush=True)

    eta_anchor = None
    if not args.free_delta:
        eta_anchor = declared_anchor(df, K, free_from, n_q)
        print("delta anchor (exp):",
              np.round(np.exp(eta_anchor), 2).tolist(), flush=True)
    logpost, unpack, cuts, n_par, sizes = build_logpost(
        data, drift, monotone, use_disc, gamma, free_from,
        eta_anchor=eta_anchor)
    n_free = sizes["n_free"]

    f = jax.jit(logpost)
    g = jax.jit(jax.grad(logpost))

    x0 = np.zeros(n_par)
    x0[cuts["beta0"][0]] = -7.5
    x0[cuts["m0"][0]] = -3.0
    x0[cuts["eta0"][0]] = np.log(1.0)
    x0[cuts["m_inc_raw"][0]:cuts["m_inc_raw"][1]] = -1.0 if monotone else 0.0
    for name in ("log_sd_b", "log_sd_eta", "log_sd_v", "log_sd_m", "log_sd_u"):
        x0[cuts[name][0]] = -1.5
    x0[cuts["rho_a"][0]] = -4.0

    from scipy.optimize import minimize
    t0 = time.time()
    res = minimize(lambda x: float(f(jnp.asarray(x))),
                   x0, jac=lambda x: np.asarray(g(jnp.asarray(x))),
                   method="L-BFGS-B",
                   options={"maxiter": 60000, "maxfun": 120000, "ftol": 1e-11,
                            "gtol": 1e-6})
    print(f"optimum {res.fun:.2f} in {time.time()-t0:.0f}s, "
          f"converged={res.success} ({res.message})", flush=True)

    # Laplace covariance on the pi-relevant block via full Hessian
    th = jnp.asarray(res.x)
    H = np.asarray(jax.hessian(logpost)(th))
    jitter = 1e-8 * np.eye(n_par)
    try:
        cov = np.linalg.inv(H + jitter)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H + jitter)

    # transform draws -> pi trajectories (delta method via sampling the
    # Gaussian; cheap and exact enough for bands)
    rng = np.random.default_rng(0)
    ev = np.linalg.eigvalsh(H + jitter)
    n_bad = int((ev <= 0).sum())
    if n_bad:
        print(f"WARNING: {n_bad} non-positive Hessian eigenvalues "
              f"(min {ev.min():.2e}); using abs-eigenvalue PSD projection")
        w, Q = np.linalg.eigh(H + jitter)
        cov = (Q * (1.0 / np.abs(w))) @ Q.T
    draws = rng.multivariate_normal(np.asarray(th), cov, size=2000)

    a, b = cuts["m0"][0], cuts["m_inc_raw"][1]
    m0_d = draws[:, cuts["m0"][0]]
    sd_m_d = np.exp(draws[:, cuts["log_sd_m"][0]])
    incraw_d = draws[:, cuts["m_inc_raw"][0]:cuts["m_inc_raw"][1]]
    if monotone:
        inc_d = np.logaddexp(0.0, incraw_d) * sd_m_d[:, None]
    else:
        inc_d = incraw_d * sd_m_d[:, None]
    m_d = np.concatenate([m0_d[:, None], m0_d[:, None] + np.cumsum(inc_d, 1)], 1)
    pi_d = 1.0 / (1.0 + np.exp(-m_d))

    tag = f"{args.phase}_{args.variant}" + (f"_{args.section}" if args.section else "")
    qs = [quarter_label(free_from + i) for i in range(n_free)]
    tab = pd.DataFrame({"quarter": qs, "mean": pi_d.mean(0),
                        "lo": np.percentile(pi_d, 2.5, 0),
                        "hi": np.percentile(pi_d, 97.5, 0)})
    tab.to_csv(os.path.join(DATA, f"pi_{tag}.csv"), index=False)
    print(tab.round(4).to_string(index=False))

    p = unpack(th)
    eta = np.concatenate([[float(p["eta0"])],
                          float(p["eta0"])
                          + np.cumsum(np.asarray(p["eta_inc"])
                                      * np.exp(float(p["log_sd_eta"])))])
    out = {"variant": args.variant, "phase": args.phase,
           "neg_logpost": float(res.fun), "converged": bool(res.success),
           "n_bad_eigen": n_bad,
           "delta_by_quarter": {qs[i]: float(np.exp(eta[i]))
                                for i in range(n_free)},
           "beta0": float(p["beta0"]),
           "g0_slope": float(p["slope"]), "g0_curve": float(p["curve"]),
           "phi0": float(np.exp(p["log_phi0"])),
           "phi1": float(np.exp(p["log_phi1"])),
           "u_topic": np.asarray(np.concatenate(
               [np.asarray(p["u_raw"]), -np.sum(np.asarray(p["u_raw"]),
                                                keepdims=True)])
               * np.exp(float(p["log_sd_u"]))).tolist()}
    with open(os.path.join(DATA, f"laplace_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote pi_{tag}.csv and laplace_{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="fulltext",
                    choices=["abstracts", "fulltext"])
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--section", default=None, choices=SECTIONS + ["wholebody"])
    ap.add_argument("--free-delta", action="store_true",
                    help="disable the declared-papers anchor on delta_t")
    fit(ap.parse_args())
