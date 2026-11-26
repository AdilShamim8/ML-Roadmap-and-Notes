# Time Series Cheat Sheet

One page. The decomposition vocabulary, the stationarity machinery, the models, and the one rule that decides the round.

## The one rule
**Never shuffle. Random splits leak the future into the past.** Validate with:
- **Rolling-origin (walk-forward):** train on [0, t) → forecast [t, t+h) → roll t forward → average across origins. The gold standard.
- **Single temporal cut:** cheap take-home version.
- Report error **at the horizons the business consumes** (1-day ≠ 1-month error).

## Decomposition
```
Y = trend + seasonality + residual            (additive)
Y = trend × seasonality × residual            (multiplicative — amplitude grows with level → log first)
seasonal (fixed period: hour/day/week/quarter) ≠ cyclical (irregular period: business cycles)
STL: robust decomposition, tunable seasonal smoothness
```

## Stationarity
```
Stationary = mean/variance/autocorrelation constant over time (ARMA's requirement)
ADF:  H0 = unit root (non-stationary). p < 0.05 → stationary
KPSS: H0 = stationary.            p > 0.05 → stationary    (run both — the 2x2)
Fix, in order: log/Box-Cox (variance) → differencing d (trend) → seasonal D (seasonal unit root)
d: difference until ADF rejects, as few times as possible (over-differencing injects negative autocorrelation)
```

## ARIMA family
```
AR(p): yₜ = c + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ          (own lags — momentum)
MA(q): yₜ = c + εₜ + θ₁εₜ₋₁ + ... + θ_q εₜ₋q        (lagged errors — shock correction)
ARIMA(p, d, q): difference d times, then ARMA(p, q)
SARIMA(p,d,q)(P,D,Q,s): + seasonal components at period s
Order selection: d by ADF → ACF cutoff after q → MA(q); PACF cutoff after p → AR(p)
                 → refine by AIC/BIC → validate out-of-sample (rolling origin)
Exponential smoothing: SES (flat) → Holt (+trend) → Holt-Winters (+seasonality)
```

## The modern stack (2026)
| Approach | When |
| --- | --- |
| Naive / seasonal-naive | **always the baseline** — MASE < 1 or go home |
| ETS / ARIMA | clean single series, interpretability |
| GBM on lags+calendar (global model) | many related series, exogenous features |
| Deep (N-BEATS / TFT / DeepAR) | massive series counts, complex interactions |
| Ensemble of the above | the honest production answer |

**The supervised conversion:** lags (1, 2, 3, s), rolling stats **shifted** (`shift(1).rolling(k)`), calendar (dow/month/holiday), events → ordinary regression. `shift` before `rolling` or you leak the present.

## Metrics
```
MAE / RMSE — target units; RMSE punishes big misses
MAPE — undefined at 0, explodes near 0, asymmetric — positive series only
sMAPE — bounded version
MASE = MAE / in-sample naive MAE — MASE < 1 beats the naive forecast
```

## Diagnostics vocabulary
| Term | One-liner |
| --- | --- |
| ACF | autocorrelation at each lag; slow decay → non-stationary |
| PACF | lag-k correlation controlling for 1..k−1 |
| White noise | iid, zero ACF; residuals should be this |
| Random walk | differences are white noise; forecast = last value |
| Structural break | regime change (CUSUM/Chow); roll windows to survive |
| Granger causality | X's lags improve Y's forecast — prediction, NOT causation |
| Conformal prediction | distribution-free prediction intervals for any model |
| Recursive vs direct multi-step | feed predictions back (compounding) vs model per horizon |
| Backtesting | walk-forward evaluation, retro-terminology from finance |
| DST/holidays | calendar features, not afterthoughts — holidays move level |

## Patterns worth naming in interviews
- **"Lost in the middle" applies to lags too:** the most recent lags dominate; distant seasonal lags (t−52) carry the cycle.
- **Label delay ≠ forecast horizon:** predicting next-month churn with labels arriving 45 days late means your "real" horizon is longer than the business thinks — surface it.
- **Two error families:** forecasting the *level* vs forecasting the *change* — agree with the stakeholder which one the decision needs before modeling.
- **Residual checks:** ACF of residuals ≈ 0 everywhere (Ljung-Box), else there's structure left to model.
