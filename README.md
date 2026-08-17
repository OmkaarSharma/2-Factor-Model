# Schwartz–Smith Two-Factor Model on EEX Phelix Base Power Futures

An end-to-end implementation of the Schwartz & Smith (2000) short-term/long-term
commodity model, estimated by Kalman filtering and maximum likelihood on German
power futures (EEX Phelix Base, monthly delivery) from Bloomberg, 2018–2026.

The study is scoped to **short-term dynamics**: the panel's longest observable
maturity is 0.71 years. Extension to quarterly and calendar-year contracts is
discussed under [Limitations](#limitations).

---

## Contents

- [What this is](#what-this-is)
- [The model](#the-model)
- [Data](#data)
- [Cleaning pipeline](#cleaning-pipeline)
- [Three methodological decisions](#three-methodological-decisions)
- [Estimation](#estimation)
- [Results](#results)
- [Diagnostics](#diagnostics)
- [Monte Carlo validation](#monte-carlo-validation)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Reproducing](#reproducing)
- [References](#references)

---

## What this is

Schwartz & Smith decompose the log spot price of a commodity into two latent
factors — a mean-reverting short-term deviation and a random-walk equilibrium
level. Neither is observable; both are recovered from the futures curve by
Kalman filtering, with the model parameters estimated by maximising the
prediction-error decomposition of the likelihood.

The original paper applies this to crude oil. Power is a harder case: it is
non-storable, its futures are *delivery-period* (flow) contracts rather than
point-maturity claims, and it carries a large deterministic seasonal component
that oil does not. This repository works through those three complications
explicitly rather than approximating them away, and reports the diagnostics that
show where the two-factor structure still falls short.

Everything here is reproducible from a single raw Bloomberg export.

---

## The model

Log spot price decomposes as `ln S_t = X_t + CHI_t`, where `X` is the short-term
deviation and `CHI` the equilibrium level:

```
dX   = -kappa * X dt + sigma_X dz_X            (short-term, Ornstein–Uhlenbeck)
dCHI = mu_CHI dt + sigma_CHI dz_CHI            (equilibrium, Brownian motion)
dz_X dz_CHI = rho dt
```

Under the risk-neutral measure the drifts are reduced by constant risk premia
`lambda_X` and `lambda_CHI`, so `X` reverts to `-lambda_X/kappa` rather than
zero and the equilibrium drift becomes `mu_CHI* = mu_CHI - lambda_CHI`.
Futures prices follow (Eq. 9 of the paper):

```
ln F(T,0) = e^{-kappa*T} X_0 + CHI_0 + A(T)
```

Seven structural parameters
(`kappa, sigma_X, lambda_X, mu_CHI, sigma_CHI, mu_CHI*, rho`)
plus one measurement-error standard deviation per maturity slot.

---

## Data

Bloomberg `BDH` export of EEX Phelix Base **month** futures, ticker root `DET`.

| | |
|---|---|
| Contracts | 120 (`DETF18` … `DETZ27`) |
| Fields | `PX_SETTLE`, `PX_VOLUME`, `OPEN_INT` |
| Raw shape | 2,253 × 367 |
| Trading days | 2,242, 2018-01-01 → 2026-08-04 |
| Units | EUR/MWh |
| `#N/A N/A` cells | 401,054 (≈50% of the data region) |

The raw file also carries a contract reference table (delivery start, delivery
end, `LAST_TRADEABLE_DT`, delivery hours) that the pipeline uses for expiry
masking and delivery-period geometry.

---

## Cleaning pipeline

The export is not usable as delivered. `clean_phelix.py` performs the following,
in order, with hard assertions rather than silent coercion.

**Structural parse.** Two unrelated blocks share the sheet — a contract
reference table and the price panel. Column triples are located from the ticker
banner and the field header is asserted to be `PX_SETTLE / PX_VOLUME /
OPEN_INT`, not assumed.

**Date parsing.** Excel wrote day ≤ 12 as `D/M/YYYY` and the rest as
`DD-MM-YYYY`. Both are day-first; verified against the trading calendar.

**Validation gates.** Ticker month codes agree with delivery months; delivery
end is the true month end; `LAST_TRADEABLE_DT ≤ delivery end`; dates strictly
increasing with no weekends; all prices strictly positive; and the delivery-hour
count recomputed from a `zoneinfo` Europe/Berlin calendar matches Bloomberg's
column for all 120 contracts — including 743 hours every March and 745 every
October, which is the daylight-saving transition.

**Post-expiry mask.** Bloomberg carries the final settlement forward
indefinitely; `DETF18` sits at 29.46 for 2,200 days after it died. For all 120
contracts the last date on which the price changes equals `LAST_TRADEABLE_DT`
exactly, so the mask `date > LAST_TRADEABLE_DT` is exact rather than heuristic.

**In-delivery mask.** EEX months trade *through* their delivery month, where the
price is part realised spot average and part forward. Eq. (9) does not describe
that, so observations with `trade date ≥ delivery start` are dropped.

**Phantom run mask.** `DETZ27` reads a flat 30.30 with non-zero open interest
for its first 2,066 rows. Caught by a generic rule (leading constant run > 5
trading days); no other contract has a leading run longer than 2 days.

**Holiday rows.** 57 dates on which no live contract moved — Jan 1, Good Friday,
Easter Monday, May 1, Dec 24/25/26, Dec 31.

**Outliers: none removed.** The only surviving extreme cluster is
2022-02-24/25, log-returns of 0.35–0.40 across the whole strip. That is the
invasion of Ukraine, and it is real.

```
raw price cells                141,001
  post-expiry / in-delivery    -116,711
  DETZ27 phantom lead run        -2,066
holiday dates dropped                57
surviving observations           21,611
```

Zero interior gaps: once a contract begins quoting it quotes every trading day
until expiry, so no interpolation occurs anywhere in the pipeline.

**Output.** Three date windows, each a self-contained folder:

| window | span | weekly obs | character |
|---|---|---|---|
| `calm` | 2018-01 → 2021-06 | 183 | pre-crisis, max settle 89.8 |
| `post` | 2023-01 → 2026-08 | 188 | post-crisis, max settle 200.7 |
| `full` | 2018-01 → 2026-08 | 449 | includes the crisis, max settle 871.8 |

`full` is *not* `calm ∪ post`: the 78 weeks from 2021-07 to 2022-12 belong to
`full` alone, and they carry prices an order of magnitude above anything in
either sub-window.

---

## Three methodological decisions

These are the places where a naive port of the oil implementation would be
wrong for power.

### 1. Constant-maturity slots

Every EEX month future has a finite life, so the set of live tickers changes
monthly, while the Kalman filter needs a fixed-width observation vector. Each
trade date's surviving contracts are ranked by time to delivery and slot *k* is
defined as the *k*-th nearest month. Roughly 10 monthlies quote at any time
(EEX cascades quarters into months about nine months ahead), and **n = 8** gives
a completely dense panel — zero missing values across all 2,185 trading days;
n = 9 fails on seven dates.

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| median τ (yr) | 0.08 | 0.17 | 0.25 | 0.33 | 0.42 | 0.50 | 0.58 | 0.67 |

A slot is a *position*, not a contract: 42 distinct tickers pass through slot 1
over the `calm` window. τ within a slot sawtooths, which is harmless because
`Z` and `d` are evaluated at each observation's actual maturity.

### 2. Geometric delivery averaging

A Phelix month future settles against the *average* hourly spot over its
delivery month, so strictly `F = (1/H) Σ_u F(t,u)`. The log of a sum of
exponentials of the state is **not affine in the state**, which would break the
exactness of the linear Kalman filter.

The geometric average is affine and therefore drops straight into the existing
filter:

```
ln F_geo(t,i) = [ Σ_k w_ik e^{-kappa u_ik} ] X_t + CHI_t + Σ_k w_ik A(u_ik)
```

with `w` the DST-correct hour weights. Simulation at plausible power parameters
puts the geometric approximation error at 3e-6 to 1.6e-5 log points against
5e-5 to 1e-4 for the common delivery-midpoint shortcut — and under crisis-like
parameters the midpoint error on the front contract reaches 1e-2, comparable to
the estimated measurement noise itself and systematic in sign. Geometric
averaging costs nothing and removes that bias; the fitted mean errors per slot
come out at 1e-4 to 3e-3, with no maturity-dependent pattern.

### 3. Seasonality indexed on delivery time

Power carries a large deterministic annual shape that the two-factor structure
cannot produce: `e^{-kappa*tau}` is monotone in maturity, so a seasonal sawtooth
across the strip would be forced entirely into the measurement error, violating
both the diagonal-`H` and the iid assumptions.

The seasonal is a **3-harmonic, mean-zero function of delivery time**, hour-
averaged over each contract's delivery period, estimated by a two-way
fixed-effects regression with date effects absorbed by within-date demeaning
(Frisch–Waugh):

```
gbar(T_i) = Σ_{j=1..3} [ a_j * cbar_ji + b_j * sbar_ji ]
```

Three points that differ from a standard spot-price seasonal specification:

- **Indexed on delivery time, not trade time.** The seasonal varies across the
  cross-section at a fixed date; a function of trade date would be constant
  across all eight slots and remove nothing.
- **No intercept and no linear trend.** A constant is perfectly collinear with
  `CHI_t`, and a linear trend in delivery time splits into pieces that compete
  with `mu_CHI` and `mu_CHI*`. `CHI` carries all level and trend.
- **Annual frequencies only.** A month contract integrates over ~30 days × 24
  hours, so weekly and intraday cycles are annihilated by the delivery
  averaging and are not identifiable from this panel.

Fitted amplitude and explanatory power:

| window | partial R² | peak-to-trough (log pts) |
|---|---|---|
| `calm` | 0.625 | 0.239 |
| `post` | 0.704 | 0.374 |
| `full` | 0.621 | 0.331 |

Three harmonics explain 62–70% of the within-date cross-sectional variance in
log futures prices. The amplitude grows by 56% between the pre- and post-crisis
regimes — a 37% winter/spring spread after 2023 against 24% before — which is
why the seasonal is fitted separately per window.

---

## Estimation

Flow, implemented as specified:

```
PRECOMPUTE (once)
  delivery-time grid, hour weights, deseasonalised y matrix
theta0  <-  method-of-moments seed  ->  unconstrained space
OUTER LOOP  [Nelder-Mead, then L-BFGS-B]
  for each candidate theta:
    untransform; build (c, T, Q) per distinct step length
    evaluate Z, d on the full delivery grid (vectorised)
    set m0, C0  (stationary X, diffuse CHI)
    INNER LOOP over t:      predict -> forecast -> accumulate -> Joseph update
    return scalar l(theta)
POST-ESTIMATION (once)
  RTS smoother  ->  X_{t|T}, CHI_{t|T}
  numerical Hessian  ->  standard errors (delta method)
```

Implementation notes:

- **Method-of-moments seed from the volatility term structure.** The empirical
  variance of `dlnF` per slot equals
  `e^{-2*kappa*tau} sigma_X² + sigma_CHI² + 2 e^{-kappa*tau} rho sigma_X sigma_CHI`.
  Fitting that one curve recovers `kappa, sigma_X, sigma_CHI, rho` simultaneously.
  Roll transitions are masked so slot differences compare like with like.
- **`(c, T, Q)` per distinct step length.** Holiday-shifted weekly sampling gives
  irregular Δt (5–10 days), so the transition blocks are built once per candidate
  θ for each distinct Δt — still outside the recursion.
- **Joseph-form covariance update**, and Cholesky factorisation of the innovation
  covariance for the log-determinant and quadratic form.
- **Measurement-error floor** `s = 1e-4 + exp(theta)`, preventing the optimiser
  from chasing `s -> 0` as it does in the original paper's Table 2.
- **Delta-method standard errors**, since the Hessian is taken in the
  unconstrained parameterisation.

Optima were confirmed from multiple independent starts: `post` from 2, `full`
from 4, all reaching identical likelihood values.

---

## Results

Maximum-likelihood estimates, standard errors in parentheses.

| | `calm` | `post` | `full` |
|---|---|---|---|
| κ | **3.4172** (0.3285) | **1.6455** (0.2984) | **1.1349** (0.1980) |
| σ_X | 0.3568 (0.0337) | 0.4813 (0.0577) | 0.8027 (0.0909) |
| λ_X | −0.3630 (0.1971) | +0.1472 (0.2637) | −0.6154 (0.2922) |
| μ_CHI | +0.2156 (0.1136) | −0.2749 (0.1961) | +0.0927 (0.1694) |
| σ_CHI | 0.2120 (0.0166) | 0.3664 (0.0327) | 0.4962 (0.0516) |
| μ_CHI* | −0.0280 (0.0211) | +0.0196 (0.0453) | −0.4871 (0.0940) |
| ρ | **+0.5223** (0.1143) | **−0.2227** (0.1640) | **−0.4594** (0.1529) |
| s₁…s₈ | 0.028 – 0.045 | 0.030 – 0.049 | 0.038 – 0.077 |

| | `calm` | `post` | `full` |
|---|---|---|---|
| observations | 183 | 188 | 449 |
| −log L | −2526.6 | −2372.8 | −4792.4 |
| half-life of X | 2.43 mo | 5.05 mo | 7.33 mo |
| λ_CHI | +0.2436 | −0.2945 | +0.5798 |
| T=0 spot volatility | 50.1% | 53.6% | 72.4% |
| mean abs. error (log) | 0.0435 | 0.0550 | 0.0645 |
| min Hessian eigenvalue | 2.32e+01 | 1.33e+01 | 9.12e+00 |

All three Hessians are positive definite with no near-singular direction, so
every parameter — including `lambda_X` and `mu_CHI`, which are the weakly
identified pair in the original paper — is estimable here. The time-series span
of the panel is what identifies the P-measure drift; the Schwartz–Smith
indeterminacy argument bites on the cross-section.

![Parameter comparison across the three windows](figures/comparison_three_windows.png)

*Left: estimated measurement-error standard deviation by maturity slot. Centre:
model volatility term structure against realised per-slot volatilities (crosses),
roll transitions excluded. Right: smoothed equilibrium price on a log scale —
`calm` and `full` are separately estimated yet nearly coincide over 2018–2021.*

**Findings.**

*ρ changes sign across the crisis.* +0.52 before, −0.22 after. Pre-crisis, a
shock lifting the front of the curve lifted the whole curve; post-crisis the two
factors move against each other. This is the most striking result in the table.

*Mean reversion slows.* Half-life 2.4 → 5.0 months between the two clean
regimes. The `full` value of 7.3 months should not be read structurally — a
single κ fitted across a regime break is biased downward, because the 2021–22
level shift resembles one very slow-reverting deviation.

*λ_X is negative pre-crisis.* Under Q the short-term deviation reverts to
`-lambda_X/kappa = +0.091` rather than zero, so the front of the futures curve
sits above where the P-measure would place it. This is the opposite sign to the
oil result in the original paper and is consistent with hedging pressure from
load-serving entities that are structurally long forward demand.

*The long-run drift is not stable and should not be extrapolated.* `mu_CHI`
comes out +0.216 on `calm` — almost exactly the realised drift of the smoothed
equilibrium over that sample (+0.219) — and −0.275 on `post`. The estimator is
reading a level shift as a permanent growth rate. Schwartz & Smith hit the same
problem in reverse on both oil datasets (their footnote 9) and fixed `mu_CHI`
exogenously. Here it is left free and the instability reported as a result. Any
forecast beyond about one year is dominated by this parameter and should be
treated as parametric extrapolation, not a projection.

---

### Valuation output

![Futures curve and volatility term structure](figures/fig2_fig3_futures_and_vol.png)

*Left: the fitted futures curve against expected spot prices, with observed EEX
quotes (×) and the model's geometric delivery-average fits (•). Right: the model
volatility term structure, collapsing from the T=0 spot volatility toward
`sigma_CHI` at the long end.*

![Probabilistic forecasts](figures/fig1_probabilistic_forecasts.png)

*Left: model space, deseasonalised. Right: the same forecast with the seasonal
shape added back, in traded EUR/MWh. Both are anchored at the final observation
of the estimation window; the x-axis is a forecast horizon, not a data range.*

---

## Diagnostics

Standardised one-step-ahead innovations, `full` window:

- **Mean errors 1e-4 to 3e-3 across all eight slots** — no maturity-dependent
  bias, confirming the geometric delivery averaging.
- **Standardised innovation sd 0.92–1.10** — the innovation covariance is
  correctly scaled.
- **Residual seasonal R² = 0.001** — the harmonic decomposition is complete.
- **Ljung–Box(10) of 32–196 against a 95% critical value of 18.31**, with
  first-order autocorrelation up to 0.66 and cross-slot innovation correlation
  up to 0.39. The innovations are *not* independent.
  ![Smoothed states and standardised innovations](figures/states_and_innovations.png)

The last point is the honest limitation of the model, and it is reported rather
than patched. Near-zero mean errors combined with heavily autocorrelated
residuals of comparable size at every maturity is the signature of a systematic
curve shape that two factors cannot span. The volatility term structure shows
the same thing from another angle: realised per-slot volatilities lie *above*
the model curve at every maturity in all three windows, with the gap widening
toward the long end.

Pricing error by year, `full` window (per-observation averages):

| year | log | % of price | EUR/MWh |
|---|---|---|---|
| 2018 | 0.041 | 4.2 | 1.99 |
| 2019 | 0.044 | 4.4 | 2.02 |
| 2020 | 0.057 | 5.6 | 2.01 |
| 2021 | 0.088 | 9.2 | 14.69 |
| **2022** | **0.121** | **12.2** | **42.72** |
| 2023 | 0.074 | 7.2 | 9.06 |
| 2024 | 0.057 | 5.7 | 4.63 |
| 2025 | 0.044 | 4.4 | 3.83 |
| overall | 0.065 | 6.5 | 9.76 |

The aggregate figure of 6.5% conceals a 3× spread between the quiet years and
2022. Note also that these are *one-week-ahead* errors conditional on a state
vector re-estimated every week: across the full window the cross-date standard
deviation of the log price level is 0.657 against a residual standard deviation
of 0.093, so the equilibrium factor absorbs the level and the residual measures
curve shape only. The model tracks the crisis; it does not predict it.

---

## Monte Carlo validation

Simulate-and-recover on the `full` window: 100 synthetic panels drawn from the
fitted parameters on the real maturity grid, priced through the geometric
measurement equation, refitted, and compared against the Hessian standard
errors. Run in parallel, 404 s total (4.0 s per replication).

| | true | MC mean | bias | MC sd | Hessian se | sd ratio |
|---|---|---|---|---|---|---|
| κ | 1.1349 | 1.1702 | +0.0353 | 0.1663 | 0.1980 | 0.84 |
| σ_X | 0.8027 | 0.7977 | −0.0050 | 0.0927 | 0.0909 | 1.02 |
| λ_X | −0.6154 | −0.5653 | +0.0501 | 0.2370 | 0.2922 | 0.81 |
| μ_CHI | 0.0927 | 0.0837 | −0.0091 | 0.1533 | 0.1694 | 0.90 |
| σ_CHI | 0.4962 | 0.4980 | +0.0018 | 0.0497 | 0.0516 | 0.96 |
| μ_CHI* | −0.4871 | −0.4827 | +0.0044 | 0.0715 | 0.0940 | 0.76 |
| ρ | −0.4594 | −0.4361 | +0.0233 | 0.1385 | 0.1529 | 0.91 |

![Simulate-and-recover parameter distributions](figures/monte_carlo_recovery.png)

*R = 100 replications. Black line: fitted value used to generate the panels.
Red dashed: Monte Carlo mean.*

100/100 replications converged. Biases are small — the largest are κ at +3.1%
and λ_X at +8.1%, both marginally significant at R=100 and consistent with the
known finite-sample bias of mean-reversion parameters. Every `sd ratio` falls in
0.76–1.02, so the Hessian standard errors are *not* optimistic; if anything they
are mildly conservative.

**What this does and does not establish.** The synthetic panels are generated
from the model itself, so their innovations are iid by construction. This
confirms that the estimator and its Hessian are internally consistent — the code
recovers what it is given, and the reported standard errors are honest *under
the model's own assumptions*. It does not test whether those standard errors
survive the misspecification visible in the Ljung–Box statistics. Those are
distinct questions.

---

## Limitations

**Maximum maturity 0.71 years.** Only monthly contracts are used, and EEX
cascades quarters into months about nine months out. `sigma_CHI` and `mu_CHI*`
are therefore identified over a short lever arm. The natural extension is to add
quarterly and calendar-year contracts — but their delivery periods *nest*
(a calendar year is the hour-weighted average of its quarters, which are the
average of their months), so stacking all three tiers makes the innovation
covariance near-singular and the diagonal-`H` assumption indefensible. A
non-overlapping strip is required. This is left as future work.

**Residual autocorrelation.** Documented above. Two factors do not span the
observed curve shape; a third factor, or a stochastic short-term volatility,
is the natural next step.

**`mu_CHI` should not be extrapolated.** See Results. Fixing it exogenously (as
Schwartz & Smith do) is a reasonable robustness variant and is a one-line change.

**One seasonal per window.** The seasonal amplitude demonstrably shifted between
regimes, and the `full` window fits a single compromise shape across both. A
time-varying seasonal inside a constant-parameter model would be internally
inconsistent, so this is accepted rather than fixed.

**Data licensing.** The raw Bloomberg export is not redistributable and is not
included in this repository.

---

## Repository layout

```
clean_phelix.py        raw Bloomberg export -> cleaned, deseasonalised panels
ss_flow.py             load_window, geometric measurement equation, seasonal_g
ss_fit.py              estimation engine: MoM seed, Kalman, smoother, Hessian
mc_worker.py           worker module for the parallel Monte Carlo

notebook cells
  cell_kf_mle.py       single-window Kalman + MLE
  cell_diagnostics.py  innovation diagnostics and state plots
  cell_processes.py    P/Q state processes, spot moments, risk premium, Figure 1
  cell_futures.py      futures valuation (Eq. 9), Figures 2 and 3
  cell_compare.py      fit all three windows, side-by-side comparison
  cell_mc.py           parallel simulate-and-recover

prepped/
  contracts.csv        contract reference table
  qc_report.json       masks applied, seasonal coefficients, per-window summary
  calm/ post/ full/    y.npy, y_raw.npy, gbar.npy, tau_*.npy, U.npy, W.npy,
                       dt.npy, dates.npy, settle/volume/open_int, tickers.csv,
                       panel_long.csv
```

`U.npy` is `(n_T, n, 31)` — the delivery-day times of every contract in every
slot, in years from the trade date. `W.npy` holds the matching DST-correct hour
weights, padded with zeros for months shorter than 31 days.

---

## Reproducing

```bash
python clean_phelix.py GERMAN_POWER_EXCHANGE.csv prepped
```

Then run the notebook cells in order. Approximate timings on a modern laptop:

| step | time |
|---|---|
| cleaning, all three windows | < 1 min |
| `calm` fit (1 start) | ~2 min |
| `post` fit (2 starts) | ~4 min |
| `full` fit (4 starts) | ~21 min |
| Monte Carlo, R=100, parallel | ~7 min |

Requires `numpy`, `pandas`, `scipy`, `matplotlib`. Python 3.9+ (`zoneinfo`).

The Monte Carlo worker must remain a `.py` module rather than a notebook cell —
on Windows and macOS the multiprocessing start method is `spawn`, so each child
process re-imports the module defining the worker, and a function defined in a
notebook cell is not importable.

---

## References

Schwartz, E. and J. E. Smith (2000). *Short-Term Variations and Long-Term
Dynamics in Commodity Prices.* Management Science 46(7), 893–911.

Gibson, R. and E. S. Schwartz (1990). *Stochastic Convenience Yield and the
Pricing of Oil Contingent Claims.* Journal of Finance 45, 959–976.

Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman
Filter.* Cambridge University Press.
