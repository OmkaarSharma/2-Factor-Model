"""
Drop-in additions for the Schwartz-Smith notebook: loading the cleaned Phelix
panel and the geometric-delivery-average measurement equation.

Everything else in the notebook (transition, Joseph update, likelihood,
smoother) is unchanged. Replace only the call to build_measurement_grid.
"""
import numpy as np, pandas as pd
from pathlib import Path


def load_window(path):
    """Load one cleaned window. Returns a dict of arrays ready for the filter."""
    p = Path(path)
    d = {k: np.load(p / f"{k}.npy") for k in
         ("y", "y_raw", "gbar", "tau_start", "tau_end", "tau_mid",
          "U", "W", "dt", "settle", "volume", "open_int")}
    d["stale"]   = np.load(p / "stale.npy")
    d["dates"]   = pd.DatetimeIndex(np.load(p / "dates.npy"))
    d["tickers"] = pd.read_csv(p / "tickers.csv", index_col=0)
    d["n_T"], d["n"] = d["y"].shape
    assert not np.isnan(d["y"]).any()
    return d


# ---------------------------------------------------------------------------
# Geometric-delivery-average measurement equation
#
#   ln F_geo(t, i) = [ sum_k w_ik e^{-kappa u_ik} ] chi_t + xi_t
#                    + sum_k w_ik A(u_ik)
#
# Affine in the state, so the linear Kalman filter stays exact.  Reduces to the
# point-maturity form when the delivery period collapses to a single instant.
# ---------------------------------------------------------------------------
def build_measurement_grid_geo(p, U, W, A):
    """U, W: (n_T, n, n_days) delivery-day times (years) and hour weights.
    A: the notebook's A(T, p).  Returns d (n_T, n), Z (n_T, n, 2)."""
    Zc = (W * np.exp(-p.kappa * U)).sum(-1)            # (n_T, n)
    d  = (W * A(np.where(W > 0, U, 1.0), p)).sum(-1)   # padding contributes 0
    Z  = np.stack([Zc, np.ones_like(Zc)], axis=-1)
    return d, Z


def initial_state_geo(p, y_first, U_first, W_first, A, c0_xi=1.0):
    """chi at its stationary law; xi from the first cross-section with a proper
    but weak prior (sd 1 in log space).  Still skip t=0 from the likelihood."""
    dd = (W_first * A(np.where(W_first > 0, U_first, 1.0), p)).sum(-1)
    m0 = np.array([0.0, float(np.mean(np.asarray(y_first) - dd))])
    C0 = np.diag([p.sigma_chi**2 / (2 * p.kappa), c0_xi])
    return m0, C0


# ---------------------------------------------------------------------------
# Continuous seasonal g(u), for adding the shape back onto model forecasts.
# The cleaner regressed on delivery-AVERAGED harmonics, so the fitted betas are
# the coefficients of the underlying continuous g.
# ---------------------------------------------------------------------------
def seasonal_g(qc_json, window, n_harm=None):
    import json
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    b = json.load(open(qc_json))["windows"][window]["seasonal_coef"]
    if n_harm is None:                       # <-- ADD
        n_harm = sum(1 for k in b if k.startswith("cos"))   # <-- ADD
    beta = np.array([b[f"{f}{j}"] for j in range(1, n_harm + 1)
                     for f in ("cos", "sin")])

    def _hours(day):
        a = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo("Europe/Berlin"))
        d = (a + timedelta(days=1)).astimezone(ZoneInfo("UTC")) - \
            a.astimezone(ZoneInfo("UTC"))
        return round(d.total_seconds() / 3600)

    rows = []                                    # 12-month reference-year offset
    for m in range(1, 13):
        st = pd.Timestamp(2019, m, 1); en = st + pd.offsets.MonthEnd(0)
        dd = pd.date_range(st, en, freq="D")
        w = np.array([_hours(x) for x in dd], float); w /= w.sum()
        phi = ((dd - pd.Timestamp(2019, 1, 1)).days.to_numpy() + 0.5) / 365.25
        rows.append([f(2 * np.pi * j * phi) @ w
                     for j in range(1, n_harm + 1) for f in (np.cos, np.sin)])
    off = float(np.mean(np.asarray(rows) @ beta))

    def g(dates):
        d = pd.DatetimeIndex(np.atleast_1d(dates))
        phi = (d.dayofyear.to_numpy() - 0.5) / 365.25
        cols = [f(2 * np.pi * j * phi)
                for j in range(1, n_harm + 1) for f in (np.cos, np.sin)]
        return np.column_stack(cols) @ beta - off

    g.n_harm = n_harm
    return g, beta, off