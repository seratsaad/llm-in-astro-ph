"""The hierarchical mixture model, shared by the abstract and full-text phases.

Per paper i we observe the marker-token count K_i in a document of L_i
analysable tokens, submitted in quarter t_i, in broad topic c_i, optionally
carrying an explicit writing-assistance declaration D_i.

  background   K_i | A_i=0 ~ NegBin(mu0_i, phi0),
               log mu0_i = log L_i + beta0 + b_{c_i} + g0(t_i)

  assisted     K_i | A_i=1 ~ NegBin(mu1_i, phi1),
               log mu1_i = log mu0_i + delta_{t_i} + v_{c_i},  delta > 0

  prevalence   A_i ~ Bernoulli(pi_{t_i,c_i}),
               logit pi = m_t + u_c,  with pi == 0 before FREE_FROM.

A_i is marginalised analytically, so the sampler only ever sees continuous
parameters. delta_t is strictly positive, which fixes the label of the two
mixture components; without it the components are exchangeable.

The background drift g0(t) is the identification crux. It is linear in t with
slope estimated on the known-negative era and then *extrapolated*; the three
drift modes below bracket the assumption.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

# quarter index (2015Q1 == 0) from which prevalence is allowed to be non-zero
FREE_FROM = 20          # 2020Q1
ADOPTION = 31           # 2022Q4, ChatGPT public release


def build(data, drift="linear", monotone=True, use_disclosure=True,
          gamma=0.0, free_from=FREE_FROM, eta_anchor=None, anchor_sd=0.2):
    """data: dict with K, L, t, c, declared (all 1-D numpy), n_topics, n_quarters.

    drift:
      "linear"  g0 continues its known-negative-era linear trend (primary)
      "frozen"  g0 held at its value at free_from (all later drift -> prevalence)
      "tracked" g0 follows a supplied per-quarter neutral-vocabulary offset
    monotone: prevalence increments constrained non-negative.
    gamma: log ratio of marker rate in declared vs silent assisted papers.
    """
    K = data["K"].astype("int64")
    logL = np.log(data["L"].astype("float64"))
    t = data["t"].astype("int32")
    c = data["c"].astype("int32")
    n_t = int(data["n_quarters"])
    n_c = int(data["n_topics"])
    declared = data.get("declared")
    if declared is None:
        use_disclosure = False
    else:
        declared = declared.astype("int8")

    # quarter axis, centred on the known-negative era so the intercept is
    # interpretable and the extrapolation is a genuine extrapolation
    tq = np.arange(n_t, dtype="float64")
    tc = (tq - free_from) / 4.0                     # units of years past free_from

    free_idx = np.arange(free_from, n_t)
    n_free = len(free_idx)

    coords = {"quarter": np.arange(n_t), "topic": np.arange(n_c),
              "free_quarter": free_idx}

    with pm.Model(coords=coords) as m:
        # ---------------------------------------------------------- background
        beta0 = pm.Normal("beta0", mu=-7.0, sigma=3.0)
        sd_b = pm.HalfNormal("sd_b", 1.0)
        b_raw = pm.ZeroSumNormal("b_raw", sigma=1.0, dims="topic")
        b = pm.Deterministic("b_topic", b_raw * sd_b, dims="topic")

        slope = pm.Normal("g0_slope", mu=0.0, sigma=0.5)      # per year
        curve = pm.Normal("g0_curve", mu=0.0, sigma=0.05)     # gentle quadratic

        g_lin = slope * tc + curve * tc ** 2
        if drift == "linear":
            g0 = g_lin
        elif drift == "frozen":
            # g_lin is zero at t == free_from by construction, so clamping the
            # later quarters to zero holds the background at its 2020Q1 level
            g0 = pt.switch(pt.ge(pt.as_tensor(tq), free_from),
                           pt.zeros_like(g_lin), g_lin)
        elif drift == "tracked":
            offs = pt.as_tensor(data["drift_offset"].astype("float64"))
            g0 = pt.switch(pt.ge(pt.as_tensor(tq), free_from), offs, g_lin)
        else:
            raise ValueError(drift)
        g0 = pm.Deterministic("g0", g0, dims="quarter")

        phi0 = pm.HalfNormal("phi0", 5.0)

        log_mu0 = logL + beta0 + b[c] + g0[t]
        mu0 = pt.exp(log_mu0)

        # ----------------------------------------------------------- assisted
        # delta_t > 0: the log excess marker rate of assisted prose.
        eta0 = pm.Normal("eta0", mu=np.log(0.7), sigma=1.0)
        sd_eta = pm.HalfNormal("sd_eta", 0.4)
        eta_inc = pm.Normal("eta_inc", 0.0, 1.0, shape=n_free - 1)
        eta = pt.concatenate([[eta0], eta0 + pt.cumsum(eta_inc * sd_eta)])
        if eta_anchor is not None:
            # Spam-filter calibration, identical to the Laplace fit: every
            # quarter's log excess is pulled to the declared-paper trajectory.
            anc = np.asarray(eta_anchor, dtype="float64")
            pm.Potential("eta_anchor",
                         -0.5 * pt.sum(((eta - anc) / anchor_sd) ** 2)
                         - 0.5 * ((eta0 - anc[0]) / anchor_sd) ** 2)
        delta_free = pm.Deterministic("delta", pt.exp(eta), dims="free_quarter")

        sd_v = pm.HalfNormal("sd_v", 0.3)
        v_raw = pm.ZeroSumNormal("v_raw", sigma=1.0, dims="topic")
        v = pm.Deterministic("v_topic", v_raw * sd_v, dims="topic")

        phi1 = pm.HalfNormal("phi1", 5.0)

        # ---------------------------------------------------------- prevalence
        m0 = pm.Normal("m0", mu=-4.0, sigma=2.0)
        sd_m = pm.HalfNormal("sd_m", 0.6)
        if monotone:
            inc = pm.HalfNormal("m_inc", 1.0, shape=n_free - 1)
        else:
            inc = pm.Normal("m_inc", 0.0, 1.0, shape=n_free - 1)
        m_t = pt.concatenate([[m0], m0 + pt.cumsum(inc * sd_m)])
        pm.Deterministic("m", m_t, dims="free_quarter")

        sd_u = pm.HalfNormal("sd_u", 0.5)
        u_raw = pm.ZeroSumNormal("u_raw", sigma=1.0, dims="topic")
        u = pm.Deterministic("u_topic", u_raw * sd_u, dims="topic")

        pi_free = pm.Deterministic("pi", pm.math.sigmoid(m_t), dims="free_quarter")

        # ------------------------------------------------------- likelihood
        is_free = t >= free_from
        idx_free = np.where(is_free)[0]
        idx_fix = np.where(~is_free)[0]

        # known-negative era: background only
        if len(idx_fix):
            pm.NegativeBinomial("obs_pre", mu=mu0[idx_fix], alpha=phi0,
                                observed=K[idx_fix])

        tf = t[idx_free] - free_from
        cf = c[idx_free]
        mu0f = mu0[idx_free]
        Kf = K[idx_free]

        logit_pi = m_t[tf] + u[cf]
        # log sigmoid(x) = -softplus(-x);  log(1 - sigmoid(x)) = -softplus(x)
        log_pi = -pt.softplus(-logit_pi)
        log_1mpi = -pt.softplus(logit_pi)

        mu1_silent = mu0f * delta_free[tf] * pt.exp(v[cf])
        lp0 = pm.logp(pm.NegativeBinomial.dist(mu=mu0f, alpha=phi0), Kf)
        lp1 = pm.logp(pm.NegativeBinomial.dist(mu=mu1_silent, alpha=phi1), Kf)

        if use_disclosure:
            df = declared[idx_free]
            # rho: P(declare | assisted) by quarter, pooled, rising
            rho_a = pm.Normal("rho_a", -4.0, 2.0)
            rho_b = pm.Normal("rho_b", 0.0, 1.0)
            logit_rho = rho_a + rho_b * ((tf) / 4.0)
            log_rho = -pt.softplus(-logit_rho)
            log_1mrho = -pt.softplus(logit_rho)
            # declared papers may use markers at a different rate (selection)
            mu1_decl = mu1_silent * np.exp(gamma)
            lp1d = pm.logp(pm.NegativeBinomial.dist(mu=mu1_decl, alpha=phi1), Kf)

            # D=1: assisted and declared.   D=0: background, or assisted+silent.
            ll_d1 = log_pi + log_rho + lp1d
            ll_d0 = pt.logaddexp(log_1mpi + lp0, log_pi + log_1mrho + lp1)
            ll = pt.switch(pt.eq(pt.as_tensor(df), 1), ll_d1, ll_d0)
        else:
            ll = pt.logaddexp(log_1mpi + lp0, log_pi + lp1)

        pm.Potential("obs_free", pt.sum(ll))
    return m
