"""
Hidden Markov Regime Detection Model
====================================

This module implements a simple market regime detection model using a
Gaussian mixture model (GMM) as a proxy for a hidden Markov model. It
analyses cross‑sectional return characteristics to identify
latent states such as bullish, bearish and neutral regimes. Each state
is characterised by a mean vector and covariance matrix over a set of
features derived from the market. The state with the highest mean
return is designated as the bullish state. The model outputs the
probability of being in this bullish regime on the next trading day.

Usage
-----
```
from ml_models.hmm_model import HMMModel
import pandas as pd

prices = pd.read_csv('adj_close_20y_wide.csv', parse_dates=['date'], index_col='date')
hmm = HMMModel({'n_states': 3})
states = hmm.train_and_predict(prices)
print(states['AAPL'])  # bullish probability
```

Notes
-----
* This model does not enforce Markov transitions explicitly; instead, it
  fits a mixture model to the cross‑sectional features. For shorter
  histories this approximation is robust and computationally efficient.
* The same bullish probability is applied to all tickers since it
  reflects a market‑wide regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class HMMModel:
    """Gaussian mixture model for market regime detection.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. Supported keys:

        - ``n_states`` (int): Number of latent regimes (clusters).
          Defaults to 3.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    n_states: int = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config or {}
        self.n_states = int(cfg.get('n_states', 3))

    def _compute_market_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Construct cross‑sectional market features for each date.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of returns with datetime index and tickers as columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with features such as mean return, volatility,
            skewness, kurtosis, and breadth.
        """
        # Mean return across tickers
        mean_return = returns.mean(axis=1)
        # Cross‑sectional volatility (standard deviation)
        volatility = returns.std(axis=1)
        # Skewness and kurtosis; handle potential NaNs
        skewness = returns.apply(lambda row: row.skew(), axis=1)
        kurtosis = returns.apply(lambda row: row.kurt(), axis=1)
        # Breadth: proportion of positive returns
        breadth = (returns > 0).sum(axis=1) / returns.shape[1]
        # Aggregate into DataFrame
        features = pd.DataFrame({
            'mean_return': mean_return,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'breadth': breadth
        })
        # Replace any infinities or NaNs with zeros
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.fillna(0.0)
        return features

    def train_and_predict(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Fit the mixture model and compute bullish regime probabilities.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted closing prices with datetime index.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary keyed by ticker with the probability of being
            in the bullish regime on the next day. The probability is
            identical across tickers.
        """
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().ffill().bfill()
        returns = prices.pct_change().dropna(how='all')
        # Compute features
        market_features = self._compute_market_features(returns)
        # Standardise features
        scaler = StandardScaler()
        X = scaler.fit_transform(market_features)
        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=self.n_states, covariance_type='full', random_state=42)
        gmm.fit(X)
        # Identify bullish state as the component with the highest mean return
        # Compute means in original feature space by inverse transforming
        means = scaler.inverse_transform(gmm.means_)
        mean_returns = means[:, 0]  # first column corresponds to mean_return
        bullish_state = int(np.argmax(mean_returns))
        # Compute posterior probabilities for each state on the last day
        last_features = X[-1].reshape(1, -1)
        posteriors = gmm.predict_proba(last_features)[0]
        bullish_prob = float(posteriors[bullish_state])
        # Prepare output: same probability for all tickers
        result: Dict[str, Dict[str, float]] = {}
        for ticker in prices.columns:
            result[ticker] = {'bullish_probability': bullish_prob}
        return result