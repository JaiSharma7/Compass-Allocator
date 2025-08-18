"""
GARCH/EWMA Volatility Forecasting Model
======================================

This module estimates next‑day volatility for each ticker using either
a Generalised Autoregressive Conditional Heteroskedasticity (GARCH)
model or an exponentially weighted moving average (EWMA) model if the
`arch` library is unavailable. Volatility forecasts are essential for
risk management and for scaling return forecasts into directional
probabilities.

Usage
-----
```
from ml_models.garch_model import GARCHModel
import pandas as pd

prices = pd.read_csv('adj_close_20y_wide.csv', parse_dates=['date'], index_col='date')

garch = GARCHModel({'use_arch_package': True, 'ewma_lambda': 0.94, 'distribution': 't'})
vol_preds = garch.train_and_predict(prices)
print(vol_preds['AAPL'])
```

Notes
-----
* The model fits a separate volatility process for each ticker. When
  using the ``arch`` package, a GARCH(1,1) model is fit with the
  specified distribution (normal or t). If ``arch`` is not installed,
  an EWMA estimator is used, which is fast and widely accepted.
* The forecasted volatility is the square root of the one‑step ahead
  conditional variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any

try:
    from arch import arch_model
except ImportError:
    arch_model = None  # type: ignore


@dataclass
class GARCHModel:
    """Volatility forecasting using GARCH(1,1) or EWMA.

    Parameters
    ----------
    config : dict, optional
        Configuration for the model. Keys include:

        - ``use_arch_package`` (bool): Whether to attempt using
          ``arch`` for GARCH modelling. Defaults to True.
        - ``ewma_lambda`` (float): Decay factor for EWMA when GARCH
          cannot be used. Defaults to 0.94.
        - ``distribution`` (str): Distribution for the GARCH model
          ('normal' or 't'). Defaults to 'normal'.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    use_arch_package: bool = field(init=False)
    ewma_lambda: float = field(init=False)
    distribution: str = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config or {}
        self.use_arch_package = bool(cfg.get('use_arch_package', True))
        self.ewma_lambda = float(cfg.get('ewma_lambda', 0.94))
        self.distribution = cfg.get('distribution', 'normal')

    def _ewma_volatility(self, returns: pd.Series) -> float:
        """Compute one‑step ahead volatility forecast via EWMA.

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            The EWMA volatility forecast for the next period.
        """
        lam = self.ewma_lambda
        squared = returns ** 2
        # Initialise with unconditional variance (mean of squared returns)
        sigma2 = squared.mean()
        for r2 in squared:
            sigma2 = lam * sigma2 + (1 - lam) * r2
        return float(np.sqrt(sigma2))

    def _fit_garch(self, returns: pd.Series) -> float:
        """Fit a GARCH(1,1) model and forecast volatility.

        Parameters
        ----------
        returns : pd.Series
            Return series.

        Returns
        -------
        float
            One‑step ahead volatility forecast.
        """
        # Convert to per cent returns (as many GARCH libraries expect this scale)
        r = returns * 100.0
        # Remove any NaNs
        r = r.dropna()
        # Fit GARCH(1,1)
        model = arch_model(r, p=1, q=1, mean='Zero', vol='GARCH', dist=self.distribution)
        res = model.fit(disp='off')
        # Forecast next period variance
        forecast = res.forecast(horizon=1)
        variance = forecast.variance.values[-1, 0]
        return float(np.sqrt(variance)) / 100.0  # scale back to raw returns

    def train_and_predict(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Estimate volatilities for each ticker.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted closing prices.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary keyed by ticker with the forecasted volatility
            under the key ``predicted_volatility``.
        """
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().ffill().bfill()
        returns = prices.pct_change().dropna(how='all')
        preds: Dict[str, Dict[str, float]] = {}
        for ticker in prices.columns:
            r = returns[ticker].dropna()
            # If insufficient history, fallback to simple std
            if len(r) < 60:
                vol = float(r.std())
            else:
                use_garch = self.use_arch_package and arch_model is not None
                try:
                    if use_garch:
                        vol = self._fit_garch(r)
                    else:
                        raise ImportError
                except Exception:
                    vol = self._ewma_volatility(r)
            preds[ticker] = {'predicted_volatility': vol}
        return preds