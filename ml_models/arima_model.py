"""
ARIMA Directional Forecasting Model
===================================

This module implements an autoregressive integrated moving average
(ARIMA) approach to forecasting next‑day returns for each ticker in
a given dataset. The model automatically selects the best (p, d, q)
configuration from a small grid by minimising the Akaike Information
Criterion (AIC) on the in‑sample data. Although ARIMA is typically used
for point forecasts, we convert the point forecast into an expected
return estimate and a crude directional probability using a sigmoid
transformation. This probability is not meant to be as well‑calibrated
as the one produced by the LSTM model but provides an additional
perspective in the ensemble.

Usage
-----
```
from ml_models.arima_model import ARIMAModel
import pandas as pd

prices = pd.read_csv('adj_close_20y_wide.csv', parse_dates=['date'], index_col='date')

arima = ARIMAModel({'order_grid': [(1, 0, 0), (1, 0, 1), (2, 0, 0)]})
preds = arima.train_and_predict(prices)
print(preds['MSFT'])
```

Notes
-----
* This implementation relies on `statsmodels`. If the package is not
  available in the runtime environment, the model will fall back to a
  naive mean reversion strategy using the average of recent returns.
* The return series is assumed to be stationary; thus the differencing
  parameter ``d`` in the order tuple is typically set to 0.
* The directional probability is derived from the point forecast via
  a sigmoid function. It should not be interpreted as a calibrated
  probability but may still aid ensemble blending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None  # type: ignore


def _sigmoid(x: float) -> float:
    """Sigmoid function for mapping real numbers to (0, 1).

    A temperature parameter can be introduced here to control the
    steepness, but for simplicity we use 1.0.
    """
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ARIMAModel:
    """Autoregressive integrated moving average model for next‑day returns.

    The model fits a separate ARIMA process per ticker. It selects the
    best order from a user‑specified grid using the AIC and produces
    a one‑step ahead forecast. A rudimentary directional probability
    is computed from the forecast using a sigmoid transformation.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    order_grid: List[Tuple[int, int, int]] = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config or {}
        default_grid: List[Tuple[int, int, int]] = [(1, 0, 0), (1, 0, 1), (2, 0, 0)]
        user_grid = cfg.get('order_grid', default_grid)
        # Ensure a list of tuples
        if isinstance(user_grid, tuple):
            user_grid = [user_grid]
        self.order_grid = list({tuple(order) for order in (user_grid + default_grid)})

    def _fit_arima(self, series: pd.Series) -> Tuple[float, float]:
        """Fit ARIMA models and forecast the next value.

        Parameters
        ----------
        series : pd.Series
            Series of returns (must be stationary).

        Returns
        -------
        Tuple[float, float]
            A tuple containing the one‑step ahead forecast and the
            directional probability derived from it.
        """
        if ARIMA is None:
            # Fall back to naive prediction: mean of the last 20 returns
            mu = series.tail(20).mean()
            prob_up = _sigmoid(mu / (series.tail(20).std() + 1e-8))
            return float(mu), float(prob_up)
        best_aic = np.inf
        best_order = None
        best_model = None
        # Fit each candidate order and track the lowest AIC
        for order in self.order_grid:
            try:
                model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_model = res
            except Exception:
                # Skip invalid configurations
                continue
        if best_model is None:
            # Fallback to naive mean return
            mu = series.tail(20).mean()
            prob_up = _sigmoid(mu / (series.tail(20).std() + 1e-8))
            return float(mu), float(prob_up)
        # Forecast the next value
        forecast = best_model.forecast(steps=1)[0]
        # Convert forecast into probability via sigmoid scaled by recent volatility
        recent_vol = series.tail(30).std() + 1e-8
        prob_up = _sigmoid(float(forecast) / recent_vol)
        return float(forecast), float(prob_up)

    def train_and_predict(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Fit ARIMA models per ticker and generate forecasts.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted closing prices with datetime index.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping from ticker to forecast results. Each dictionary
            contains:

            - ``predicted_return``: the one‑step ahead forecast of the
              return.
            - ``prob_up``: a logistic mapping of the forecast to a
              directional probability.
        """
        # Ensure proper ordering and clean NaNs
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().ffill().bfill()
        returns = prices.pct_change().dropna(how='all')
        preds: Dict[str, Dict[str, float]] = {}
        for ticker in prices.columns:
            r = returns[ticker].dropna()
            # Ensure there is sufficient history
            if len(r) < 60:
                preds[ticker] = {'predicted_return': 0.0, 'prob_up': 0.5}
                continue
            forecast, prob = self._fit_arima(r)
            preds[ticker] = {'predicted_return': forecast, 'prob_up': prob}
        return preds