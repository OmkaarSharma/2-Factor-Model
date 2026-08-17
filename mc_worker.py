"""
Worker module for the parallel simulate-and-recover Monte Carlo.

Must live in a .py file, not a notebook cell: on Windows and macOS the
multiprocessing start method is 'spawn', so each child re-imports the module
that defines the worker function.  A function defined in a notebook cell is
not importable and the pool will hang or raise.
"""
import os
# tiny matrices -- BLAS threads only cause oversubscription across processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from dataclasses import dataclass

import ss_fit as F
from ss_flow import load_window


@dataclass(frozen=True)
class Params:
    kappa: float; sigma_chi: float; lambda_chi: float
    mu_xi: float;  sigma_xi: float;  mu_xi_star: float; rho: float

    @property
    def lambda_xi(self):  return self.mu_xi - self.mu_xi_star
    @property
    def half_life(self):  return np.log(2)/self.kappa


def A(T, p):
    T = np.asarray(T, float); k = p.kappa
    return (p.mu_xi_star*T - (1 - np.exp(-k*T))*p.lambda_chi/k
            + 0.5*((1 - np.exp(-2*k*T))*p.sigma_chi**2/(2*k)
                   + p.sigma_xi**2*T
                   + 2*(1 - np.exp(-k*T))*p.rho*p.sigma_chi*p.sigma_xi/k))


_CTX = {}                      # built once per worker process, not per rep


def _ctx(window):
    if window not in _CTX:
        _CTX[window] = F.make_ctx(load_window(window), A, Params)
    return _CTX[window]


def run_rep(args):
    """One replication: simulate a panel from theta, refit, return the estimates.

    args = (window, theta_tuple, seed, fresh_seed)
    Returns a (7 + n,) array, or None if the refit failed.
    """
    window, theta, seed, fresh = args
    th = np.asarray(theta, float)
    ctx = _ctx(window)
    p, s = F.untransform(th, Params)

    rng = np.random.default_rng(seed)
    y_sim, _ = F.simulate_panel(p, s, ctx, rng)
    c = dict(ctx); c['y'] = y_sim               # same tau grid, new prices

    try:
        o = F.estimate(c, n_starts=1,
                       th0=(None if fresh else th),
                       se=False, verbose=False,
                       nm_maxiter=(20000 if fresh else 0))
        return np.r_[[getattr(o['p'], f) for f in F.PNAMES], o['s']]
    except Exception:
        return None