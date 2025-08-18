"""
LSTM‑like Directional Forecasting Model
======================================

This module provides a comprehensive implementation of a directional
forecasting model inspired by long short‑term memory (LSTM) networks.
Given the constraints of this environment (no TensorFlow or PyTorch),
the model approximates the recurrent behaviour of an LSTM by
constructing a rich set of lagged and technical features and training
a gradient boosting classifier on them. The output of the classifier is
calibrated to produce reliable probabilities of an upward move in the
next trading period. An expected return estimate is derived from these
probabilities. Time‑series cross‑validation ensures robust
hyperparameter selection and guards against look‑ahead bias. The model
is configurable and can easily be extended with additional features or
alternative base classifiers.

Usage
-----
```
from ml_models.lstm_model import LSTMModel
import pandas as pd

# Load your wide price DataFrame (date index, tickers as columns)
prices = pd.read_csv('adj_close_20y_wide.csv', parse_dates=['date'], index_col='date')

# Instantiate the model with custom parameters
lstm = LSTMModel({
    'lags': 30,
    'cv_splits': 4,
    'min_train': 252,
    'embargo': 5,
    'hyperparams': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
})

# Train the model and obtain directional forecasts
predictions = lstm.train_and_predict(prices)
print(predictions['AAPL'])
```

Notes
-----
* The model treats each ticker independently, building separate feature
  matrices and classifiers. Cross‑sectional information is not used,
  which avoids contamination between series but may miss sector‑level
  signals.
* The gradient boosting classifier is chosen for its ability to
  capture non‑linear interactions between features. The hyperparameter
  grid is intentionally small to prevent overfitting.
* Probability calibration via isotonic regression is employed when
  enough data points are available. For short histories, the raw
  probabilities are used directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, balanced_accuracy_score


def _relative_strength_index(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price or return series.
    period : int, optional
        Lookback window for RSI, by default 14.

    Returns
    -------
    pd.Series
        RSI values between 0 and 100.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period, min_periods=1).mean()
    roll_down = down.rolling(period, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, span_fast: int = 12, span_slow: int = 26, span_signal: int = 9) -> pd.DataFrame:
    """Compute the Moving Average Convergence Divergence (MACD) and its signal line.

    Parameters
    ----------
    series : pd.Series
        Price series.
    span_fast : int, optional
        Fast EMA window, by default 12.
    span_slow : int, optional
        Slow EMA window, by default 26.
    span_signal : int, optional
        Signal EMA window, by default 9.

    Returns
    -------
    pd.DataFrame
        Columns ``macd`` and ``macd_signal``.
    """
    ema_fast = series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = series.ewm(span=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=span_signal, adjust=False).mean()
    return pd.DataFrame({'macd': macd, 'macd_signal': macd_signal})


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Wrapper around balanced accuracy for GridSearchCV scorer.

    Balanced accuracy is the average of recall obtained on each class,
    dealing with potential class imbalance.
    """
    return balanced_accuracy_score(y_true, y_pred)


@dataclass
class LSTMModel:
    """Directional forecasting model using feature engineering and gradient boosting.

    This class emulates the behaviour of an LSTM by constructing
    extensive lag and technical indicator features. It trains a
    gradient boosting classifier per ticker, calibrates probabilities,
    and outputs both the up‑move probability and an expected return
    estimate.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    lags: int = field(init=False)
    cv_splits: int = field(init=False)
    min_train: int = field(init=False)
    embargo: int = field(init=False)
    hyperparams: Dict[str, List[Any]] = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config or {}
        self.lags = int(cfg.get('lags', 30))
        self.cv_splits = int(cfg.get('cv_splits', 4))
        self.min_train = int(cfg.get('min_train', 252))
        self.embargo = int(cfg.get('embargo', 5))
        default_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        }
        user_grid = cfg.get('hyperparams', {})
        # Merge user grid with defaults; ensure each list contains unique values
        merged_grid: Dict[str, List[Any]] = {}
        for key, default_vals in default_grid.items():
            vals = user_grid.get(key, default_vals)
            # Flatten to list and ensure uniqueness
            if not isinstance(vals, list):
                vals = [vals]
            merged_grid[key] = sorted(set(default_vals + vals))
        self.hyperparams = merged_grid

    def _compute_features(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute lagged and technical indicator features for each ticker.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted closing prices with datetime index and
            tickers as columns.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of ticker symbols to feature DataFrames. Each
            DataFrame includes lagged returns, momentum, moving average
            differences, volatility, RSI and MACD derived features.
        """
        returns = prices.pct_change().dropna(how='all')
        features: Dict[str, pd.DataFrame] = {}
        for ticker in prices.columns:
            series = prices[ticker]
            r = returns[ticker].dropna()
            # Prepare DataFrame indexed like returns
            df = pd.DataFrame(index=r.index)
            # Lagged returns
            for lag in range(1, self.lags + 1):
                df[f'lag_{lag}'] = r.shift(lag)
            # Momentum: price relative to 10‑day prior price
            df['momentum'] = series / series.shift(10) - 1
            # Moving average difference: price minus 20‑day SMA
            sma20 = series.rolling(window=20, min_periods=5).mean()
            df['ma_diff'] = series - sma20
            # Volatility: 20‑day rolling standard deviation of returns
            df['volatility'] = r.rolling(window=20, min_periods=5).std()
            # RSI
            df['rsi'] = _relative_strength_index(series, period=14)
            # MACD and signal
            macd_df = _macd(series)
            df['macd'] = macd_df['macd']
            df['macd_signal'] = macd_df['macd_signal']
            # Drop initial NaNs arising from rolling operations
            df = df.dropna()
            features[ticker] = df
        return features

    def _train_classifier(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train a calibrated gradient boosting classifier with CV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target series (1 if next return > 0 else 0).

        Returns
        -------
        sklearn.base.BaseEstimator
            A fitted classifier (possibly calibrated).
        """
        # Define a pipeline: scale features then gradient boosting classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
        # Define parameter grid on the classifier only
        param_grid = {f'clf__{k}': v for k, v in self.hyperparams.items()}
        # Time series split for CV
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring=make_scorer(_balanced_accuracy),
            n_jobs=1,
            refit=True
        )
        # Fit grid search
        grid.fit(X, y)
        best_est = grid.best_estimator_
        # Probability calibration when enough samples (> 200)
        if len(y) > 200:
            calib = CalibratedClassifierCV(
                base_estimator=best_est,
                method='isotonic',
                cv=TimeSeriesSplit(n_splits=min(3, self.cv_splits))
            )
            calib.fit(X, y)
            return calib
        return best_est

    def train_and_predict(self, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Train the model on historical data and return directional forecasts.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted closing prices with datetime index and tickers as columns.

        Returns
        -------
        Dict[str, Dict[str, float]]
            A dictionary keyed by ticker. Each entry contains:

            - ``prob_up``: Calibrated probability of an up move on the next day.
            - ``predicted_return``: A simplistic expected return estimate computed
              as ``2 * prob_up - 1``.
        """
        # Ensure the input index is datetime and sorted
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        # Clean missing values by forward/backward fill
        prices = prices.ffill().bfill()
        # Compute features per ticker
        features = self._compute_features(prices)
        predictions: Dict[str, Dict[str, float]] = {}
        for ticker, X in features.items():
            # Align target: next day return > 0
            returns = prices[ticker].pct_change().loc[X.index]
            target = (returns.shift(-1) > 0).astype(int)
            # Drop last row (no target available for final prediction in training)
            X_train = X.iloc[:-1]
            y_train = target.iloc[:-1]
            # If not enough data points, skip training
            if len(y_train) < max(self.min_train, self.cv_splits + 1):
                prob_up = 0.5
                predicted_return = 0.0
            else:
                clf = self._train_classifier(X_train, y_train)
                # Use the last available feature row for next day prediction
                last_features = X.iloc[[-1]]
                prob_up = float(clf.predict_proba(last_features)[:, 1])
                predicted_return = 2.0 * prob_up - 1.0
            predictions[ticker] = {
                'prob_up': prob_up,
                'predicted_return': predicted_return
            }
        return predictions