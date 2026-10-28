# Forecast heads

[Ashlar](../README.md) forecasts one series at a time. `TimeSeriesForecaster` reads a window of `lookback` steps and emits the next `horizon` values. The default lookback is 48 and the default horizon is 12. The default head is `mse`.

`fit(series)` holds out the last 10% of that series, on the clock, and uses it only to choose the epoch. It then starts again from the same initialization and trains that many epochs on every window in the series you passed. `predict` reads the last `lookback` observations and returns one point per horizon step, on the original scale.

```python
from ashlar import TimeSeriesForecaster

model = TimeSeriesForecaster(lookback=48, horizon=1, head="quantile")
model.fit(series)
model.predict(history)
```

`history` has shape `(lookback,)` or `(batch, lookback)`. A batch in, a batch out.

The forecast figures in the README are one-step models on synthetic series, fit for a few epochs on CPU. [`examples/figures.py`](../examples/figures.py) writes them.

## Quantile

`head="quantile"` trains a pinball loss at 0.1, 0.5, and 0.9. Softplus gaps keep the three levels in that order. `predict` is the 0.5 level. `quantiles(history)` returns all three, shape `(horizon, 3)` for one window.

The [quantile figure](images/quantile-forecast.png) rolls a one-step model across a held-out cycle. Each window is the observed history ending at that step. That cycle was held out of the fit. The band is the 0.1 and 0.9 levels.

## Gaussian

`head="gaussian"` minimizes the normal negative log-density. The scale is `softplus(raw) + 1e-4`, the same map as a tabular `task="normal"`. `predict` is the mean. `gaussian(history)` returns the location and the scale on the original scale. `log_prob` is the density.

## Mixture

`head="mixture"` trains five normals. Component scales use that same positive map, and the weights are a softmax. The loss is the mixture negative log-density. `predict` is the weighted mean of the component locations.

`log_prob(history, value)` evaluates the density at `value`, which has the same shape `predict` returns. The [mixture figure](images/mixture-density.png) is that density on a grid, for a series of independent draws from two normals centered at −2 and 2. The bars are the series itself.

## Flow

`head="flow"` is one rational-quadratic spline per horizon step, with linear tails, pushed forward from a standard normal. The loss is the negative of the exact change-of-variables log-density. `predict` is the median, the image of zero. `sample(history, key, n)` is one forward draw per sample. `log_prob` is the density.

## Histogram

`head="histogram"` is a softmax over 16 bins fixed on the standardized interval [−4, 4], plus a small squared error of the probability-weighted bin center. `predict` is that center, mapped back to the original scale. `bin_probabilities(history)` returns the probabilities and the bin centers on the original scale.

The [histogram figure](images/histogram-bins.png) is one held-out step of a synthetic cycle. The line is the observed next value. The tall bar is the bin that contains it.

## A longer prefix

`FullHistoryForecaster` uses the same six heads. The last `lookback` observations are one token each. Every older observation sits in one non-overlapping patch of that length. `predict(series)` forecasts the horizon after the last observation. Pass `origins` to forecast at earlier cut points. `quantiles`, `log_prob`, `gaussian`, `sample`, and `bin_probabilities` take the same `origins` argument.
