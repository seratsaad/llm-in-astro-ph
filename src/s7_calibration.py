#!/usr/bin/env python3
"""Stage 7: simulation-based calibration on known-prevalence mixtures.

Kobak et al. validate their estimator on synthetic corpora with known beta;
the hierarchical model must pass the same gate. Synthetic corpora are built
from the *real* pre-2020 papers (so the background is genuinely realistic),
with an assisted component injected at a known prevalence and excess.

For speed this uses the analytic responsibility estimator at the posterior
mean rather than a full MCMC per replicate: the question here is bias of the
model structure, not sampler quality (which the rank diagnostics in Stage 3
already cover).

Writes data/calibration.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA

RNG = np.random.default_rng(20260827)
GRID_PI = [0.05, 0.15, 0.30, 0.50, 0.75]
GRID_DELTA = [1.5, 2.5, 5.0]        # multiplicative excess of assisted prose
N_PAPERS = 20000
N_REP = 8
DECL_RATE = 0.015          # declared fraction of assisted papers, as observed


def nb_logpmf(k, mu, phi):
    return (gammaln(k + phi) - gammaln(phi) - gammaln(k + 1)
            + phi * np.log(phi / (phi + mu)) + k * np.log(mu / (phi + mu)))


def nb_sample(mu, phi, rng):
    lam = rng.gamma(phi, mu / phi)
    return rng.poisson(lam)


def fit_mixture(K, L, base_rate, phi_bg, anchor=None, anchor_sd=0.2):
    """Two-parameter fit (pi, log-delta) with the background held at truth.

    This isolates the mixture inversion, the step Stage 3 does by MCMC.
    With `anchor` set, the log excess is pulled to it the way the real fit
    is pulled to the declared papers.
    """
    mu0 = base_rate * L

    def nll(theta):
        logit_pi, logd = theta
        pi = 1 / (1 + np.exp(-logit_pi))
        lp0 = nb_logpmf(K, mu0, phi_bg)
        lp1 = nb_logpmf(K, mu0 * np.exp(logd), phi_bg)
        ll = logsumexp(np.stack([np.log1p(-pi) + lp0, np.log(pi) + lp1]), axis=0)
        pen = 0.0
        if anchor is not None:
            pen = 0.5 * ((logd - anchor) / anchor_sd) ** 2
        return -ll.sum() + pen

    best = None
    for lp in (-2.0, 0.0):
        for ld in (0.5, 1.5):
            r = minimize(nll, np.array([lp, ld]), method="Nelder-Mead",
                         options={"maxiter": 2000, "fatol": 1e-3})
            if best is None or r.fun < best.fun:
                best = r
    pi = 1 / (1 + np.exp(-best.x[0]))
    return pi, np.exp(best.x[1])


def main():
    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    pre = ft[ft.year < 2020]
    L_pool = pre.L.values
    K_pool = pre.K_marker.values
    base_rate = K_pool.sum() / L_pool.sum()
    # moment-match the background overdispersion
    mu = base_rate * L_pool
    ex_var = np.mean((K_pool - mu) ** 2 - mu)
    phi_bg = float(np.mean(mu) ** 2 / max(ex_var, 1e-9))
    phi_bg = min(max(phi_bg, 0.5), 50.0)
    print(f"background: rate/1k={1000*base_rate:.3f}, phi={phi_bg:.2f}")

    results = []
    for true_pi in GRID_PI:
        for true_delta in GRID_DELTA:
            est = []
            for rep in range(N_REP):
                idx = RNG.choice(len(L_pool), N_PAPERS, replace=True)
                L = L_pool[idx]
                assisted = RNG.random(N_PAPERS) < true_pi
                mu_i = base_rate * L * np.where(assisted, true_delta, 1.0)
                K = nb_sample(mu_i, phi_bg, RNG)
                pi_hat, delta_hat = fit_mixture(K, L, base_rate, phi_bg)
                # anchored repeat: declared papers drawn from the assisted at
                # the observed declaration rate supply the calibration
                decl = assisted & (RNG.random(N_PAPERS) < DECL_RATE)
                if decl.sum() >= 5:
                    r_d = K[decl].sum() / L[decl].sum()
                    anc = np.log(max(r_d / base_rate, 1.05))
                else:
                    anc = np.log(true_delta)
                pi_a, delta_a = fit_mixture(K, L, base_rate, phi_bg, anchor=anc)
                est.append((pi_hat, delta_hat, pi_a, delta_a))
            est = np.array(est)
            results.append(dict(true_pi=true_pi, true_delta=true_delta,
                                pi_mean=float(est[:, 0].mean()),
                                pi_sd=float(est[:, 0].std()),
                                delta_mean=float(est[:, 1].mean()),
                                pi_anch_mean=float(est[:, 2].mean()),
                                pi_anch_sd=float(est[:, 2].std()),
                                delta_anch_mean=float(est[:, 3].mean())))
            r = results[-1]
            print(f"pi={true_pi:.2f} d={true_delta:.1f} -> "
                  f"pi_hat={r['pi_mean']:.3f}+-{r['pi_sd']:.3f} "
                  f"d_hat={r['delta_mean']:.2f} | anchored "
                  f"pi_hat={r['pi_anch_mean']:.3f}+-{r['pi_anch_sd']:.3f}")

    with open(os.path.join(DATA, "calibration.json"), "w") as f:
        json.dump(results, f, indent=1)
    err = [abs(r["pi_mean"] - r["true_pi"]) for r in results]
    print(f"\nmax |bias| = {max(err):.3f} over the grid")
    print("wrote data/calibration.json")


if __name__ == "__main__":
    main()
