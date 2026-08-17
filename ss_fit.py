"""
Reusable Schwartz-Smith estimation engine.

Same flow as the single-window notebook cell (precompute -> Nelder-Mead ->
L-BFGS-B -> smoother -> numerical Hessian), packaged so it can be looped over
windows and re-run inside a Monte Carlo.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from ss_flow import load_window

PNAMES = ['kappa', 'sigma_chi', 'lambda_chi', 'mu_xi',
          'sigma_xi', 'mu_xi_star', 'rho']
LOGIDX = [0, 1, 4]
S_MIN  = 1e-4


# --------------------------------------------------------------------------
# unconstrained <-> natural parameters
# --------------------------------------------------------------------------
def untransform(th, Params):
    z = np.asarray(th, float); v = z[:7].copy()
    v[LOGIDX] = np.exp(v[LOGIDX]); v[6] = np.tanh(v[6])
    return Params(*v), S_MIN + np.exp(z[7:])


def transform(p, s):
    v = np.array([p.kappa, p.sigma_chi, p.lambda_chi, p.mu_xi,
                  p.sigma_xi, p.mu_xi_star, p.rho], float)
    v[LOGIDX] = np.log(v[LOGIDX]); v[6] = np.arctanh(v[6])
    return np.concatenate([v, np.log(np.maximum(s - S_MIN, 1e-12))])


def jacobian(th):
    """d(natural)/d(unconstrained), elementwise -- for delta-method SEs."""
    j = np.ones(th.size)
    j[LOGIDX] = np.exp(th[LOGIDX])
    j[6]      = 1 - np.tanh(th[6])**2
    j[7:]     = np.exp(th[7:])
    return j


# --------------------------------------------------------------------------
# PRECOMPUTE
# --------------------------------------------------------------------------
def make_ctx(D, A, Params, n_diffuse=1, c0_xi=1.0):
    y = D['y']; n_T, n = y.shape
    ctx = dict(y=y, U=D['U'], W=D['W'], dtv=D['dt'], dates=D['dates'],
               tau=D['tau_mid'], tick=D['tickers'].to_numpy(),
               n_T=n_T, n=n, A=A, Params=Params,
               n_diffuse=n_diffuse, c0_xi=c0_xi)
    ctx['DT_KEY'] = np.concatenate([[0.0], D['dt']])
    ctx['DT_UNQ'] = np.unique(ctx['DT_KEY'][n_diffuse:])
    ctx['Upos']   = np.where(D['W'] > 0, D['U'], 1.0)
    return ctx


# --------------------------------------------------------------------------
# method-of-moments seed from the futures volatility term structure
# --------------------------------------------------------------------------
def mom_seed(ctx):
    y, tau, tick, n = ctx['y'], ctx['tau'], ctx['tick'], ctx['n']
    A, Params = ctx['A'], ctx['Params']
    same = tick[1:] == tick[:-1]
    r    = np.diff(y, axis=0) / np.sqrt(ctx['dtv'])[:, None]
    v    = np.array([np.nanvar(np.where(same[:, i], r[:, i], np.nan))
                     for i in range(n)])
    tb   = tau.mean(0)

    def obj(q):
        k, sx, sc, rho = np.exp(q[0]), np.exp(q[1]), np.exp(q[2]), np.tanh(q[3])
        m = np.exp(-2*k*tb)*sx**2 + sc**2 + 2*np.exp(-k*tb)*rho*sx*sc
        return np.sum((m - v)**2)

    q = minimize(obj, [np.log(1.0), np.log(.3), np.log(.15), 0.0],
                 method='Nelder-Mead',
                 options=dict(maxiter=8000, xatol=1e-10, fatol=1e-12)).x
    k, sx, sc, rho = np.exp(q[0]), np.exp(q[1]), np.exp(q[2]), np.tanh(q[3])

    G  = np.column_stack([np.ones(n), tb])
    mu = np.linalg.lstsq(G, y.mean(0), rcond=None)[0][1]
    p0 = Params(k, sx, 0.0, mu, sc, mu, rho)

    X   = np.stack([np.exp(-k*tau), np.ones_like(tau)], -1)
    res = np.array([y[t] - X[t] @ np.linalg.lstsq(X[t], y[t] - A(tau[t], p0),
                                                  rcond=None)[0] - A(tau[t], p0)
                    for t in range(ctx['n_T'])])
    return p0, np.maximum(res.std(0), 1e-3)


# --------------------------------------------------------------------------
# INNER LOOP:  theta -> scalar l(theta)   (store=True also returns the filter)
# --------------------------------------------------------------------------
def kalman(th, ctx, store=False):
    p, s = untransform(th, ctx['Params'])
    if not (np.isfinite(th).all() and p.kappa > 1e-8 and abs(p.rho) < .999):
        return np.inf
    y, n, n_T, A = ctx['y'], ctx['n'], ctx['n_T'], ctx['A']
    k, sx, sc, rho = p.kappa, p.sigma_chi, p.sigma_xi, p.rho

    TQ = {}
    for dt in ctx['DT_UNQ']:
        e = np.exp(-k*dt)
        TQ[dt] = (np.array([0.0, p.mu_xi*dt]), np.diag([e, 1.0]),
                  np.array([[(1-e**2)*sx**2/(2*k), (1-e)*rho*sx*sc/k],
                            [(1-e)*rho*sx*sc/k,    sc**2*dt]]))

    Zc = (ctx['W']*np.exp(-k*ctx['Upos'])).sum(-1)
    d  = (ctx['W']*A(ctx['Upos'], p)).sum(-1)
    Z  = np.stack([Zc, np.ones_like(Zc)], -1)
    H  = np.diag(s**2)

    m = np.array([0.0, float(np.mean(y[0] - d[0]))])
    C = np.diag([sx**2/(2*k), ctx['c0_xi']])
    ll, I2 = 0.0, np.eye(2)
    if store:
        aS, RS = np.zeros((n_T, 2)), np.zeros((n_T, 2, 2))
        mS, CS = np.zeros((n_T, 2)), np.zeros((n_T, 2, 2))
        VV     = np.zeros((n_T, n))

    for t in range(n_T):
        if t == 0:
            a, R = m, C
        else:
            c, Tm, Q = TQ[ctx['DT_KEY'][t]]
            a = c + Tm @ m
            R = Tm @ C @ Tm.T + Q
        Zt = Z[t]; ZR = Zt @ R
        v  = y[t] - (d[t] + Zt @ a)
        F  = ZR @ Zt.T + H; F = .5*(F + F.T)
        try:
            cf = cho_factor(F, lower=True)
        except np.linalg.LinAlgError:
            return np.inf
        if t >= ctx['n_diffuse']:
            ll += (n*np.log(2*np.pi) + 2*np.log(np.diag(cf[0])).sum()
                   + v @ cho_solve(cf, v))
        K = cho_solve(cf, ZR).T
        m = a + K @ v
        J = I2 - K @ Zt
        C = J @ R @ J.T + K @ H @ K.T; C = .5*(C + C.T)
        if store:
            aS[t], RS[t], mS[t], CS[t], VV[t] = a, R, m, C, v

    if not np.isfinite(ll):
        return np.inf
    return (aS, RS, mS, CS, Z, d, VV, .5*ll) if store else .5*ll


def smoother(p, aS, RS, mS, CS, DT_KEY):
    ms, Cs = mS.copy(), CS.copy()
    for t in range(len(mS) - 2, -1, -1):
        Tm = np.diag([np.exp(-p.kappa*DT_KEY[t+1]), 1.0])
        Jt = CS[t] @ Tm.T @ np.linalg.inv(RS[t+1])
        ms[t] = mS[t] + Jt @ (ms[t+1] - aS[t+1])
        Cs[t] = CS[t] + Jt @ (Cs[t+1] - RS[t+1]) @ Jt.T
    return ms, Cs


def hessian(fun, x, h=1e-4):
    k = x.size; Hm = np.zeros((k, k)); f0 = fun(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            if i == j:
                Hm[i, i] = (fun(x+ei) - 2*f0 + fun(x-ei))/h**2
            else:
                Hm[i, j] = Hm[j, i] = (fun(x+ei+ej) - fun(x+ei-ej)
                                       - fun(x-ei+ej) + fun(x-ei-ej))/(4*h*h)
    return Hm


# --------------------------------------------------------------------------
# OUTER LOOP + post-estimation
# --------------------------------------------------------------------------
def estimate(ctx, n_starts=1, seed=0, th0=None, se=True, verbose=True,
             nm_maxiter=20000):
    if th0 is None:
        p0, s0 = mom_seed(ctx)
        th0 = transform(p0, s0)
    rng, best = np.random.default_rng(seed), None
    for j in range(n_starts):
        st = th0 if j == 0 else th0 + rng.normal(0, .25, th0.size)
        if nm_maxiter:                       # nm_maxiter=0 -> gradient polish only
            r1 = minimize(kalman, st, args=(ctx,), method='Nelder-Mead',
                          options=dict(maxiter=nm_maxiter, maxfev=nm_maxiter,
                                       xatol=1e-8, fatol=1e-8, adaptive=True))
        else:
            class r1: pass
            r1.x, r1.fun = st, kalman(st, ctx)
        r2 = minimize(kalman, r1.x, args=(ctx,), method='L-BFGS-B',
                      options=dict(maxiter=5000, maxfun=50000,
                                   ftol=1e-12, gtol=1e-10))
        r = r2 if r2.fun <= r1.fun else r1
        if verbose:
            print(f'   start {j}: NM {r1.fun:12.4f} -> BFGS {r2.fun:12.4f}')
        if best is None or r.fun < best.fun:
            best = r

    th = best.x
    p, s = untransform(th, ctx['Params'])
    out = dict(theta=th, p=p, s=s, nll=best.fun, message=best.message)

    aS, RS, mS, CS, Z, d, VV, _ = kalman(th, ctx, store=True)
    ms, Cs = smoother(p, aS, RS, mS, CS, ctx['DT_KEY'])
    out.update(aS=aS, RS=RS, mS=mS, CS=CS, Z=Z, d=d, V=VV,
               chi_s=ms[:, 0], xi_s=ms[:, 1], Cs=Cs)

    if se:
        Hm = hessian(lambda x: kalman(x, ctx), th)
        out['hessian'] = Hm
        out['se'] = np.sqrt(np.clip(np.diag(np.linalg.pinv(.5*(Hm+Hm.T))), 0, None)) \
                    * np.abs(jacobian(th))
        out['eig'] = np.linalg.eigvalsh(.5*(Hm+Hm.T))
    return out


def table(out, n):
    est = np.r_[[getattr(out['p'], f) for f in PNAMES], out['s']]
    idx = PNAMES + [f's_{i+1}' for i in range(n)]
    t = pd.DataFrame({'estimate': est}, index=idx)
    if 'se' in out:
        t['std_error'] = out['se']
        t['t'] = t.estimate / t.std_error.replace(0, np.nan)
    return t


# --------------------------------------------------------------------------
# synthetic panel generator -- for simulate-and-recover
# --------------------------------------------------------------------------
def simulate_panel(p, s, ctx, rng):
    """Draw states under P on the real time grid, then price the real
    contracts through the geometric measurement equation and add noise."""
    A, n_T, n = ctx['A'], ctx['n_T'], ctx['n']
    k, sx, sc, rho = p.kappa, p.sigma_chi, p.sigma_xi, p.rho
    DT = ctx['DT_KEY']

    x = np.zeros((n_T, 2))
    x[0] = [0.0, float(np.mean(ctx['y'][0]))]          # anchor the level
    for t in range(1, n_T):
        dt = DT[t]; e = np.exp(-k*dt)
        Q = np.array([[(1-e**2)*sx**2/(2*k), (1-e)*rho*sx*sc/k],
                      [(1-e)*rho*sx*sc/k,    sc**2*dt]])
        x[t] = ([0.0, p.mu_xi*dt] + np.diag([e, 1.0]) @ x[t-1]
                + np.linalg.cholesky(Q) @ rng.standard_normal(2))

    Zc = (ctx['W']*np.exp(-k*ctx['Upos'])).sum(-1)
    d  = (ctx['W']*A(ctx['Upos'], p)).sum(-1)
    y  = d + Zc*x[:, [0]] + x[:, [1]] + rng.standard_normal((n_T, n))*s
    return y, x