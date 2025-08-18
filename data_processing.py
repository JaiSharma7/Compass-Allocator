# data_processing/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
import os
from tqdm import tqdm
import joblib

logger = logging.getLogger('investment_model.data_processing')


class DataPreprocessor:
    """Class for preprocessing financial data."""

    def __init__(self, fill_method='ffill', scaling_method='robust'):
        """Initialize the data preprocessor.

        Args:
            fill_method (str): Method to fill missing values ('ffill', 'bfill', 'interpolate').
            scaling_method (str): Method to scale features ('minmax', 'standard', 'robust', None).
        """
        self.fill_method = fill_method
        self.scaling_method = scaling_method
        self.scalers = {}

    def preprocess(self, stock_data_dict):
        """Preprocess stock data for all tickers.

        Args:
            stock_data_dict (dict): Dictionary of stock DataFrames by ticker.

        Returns:
            dict: Dictionary of preprocessed stock DataFrames by ticker.
        """
        logger.info("Preprocessing stock data")

        processed_data = {}

        for ticker, df in tqdm(stock_data_dict.items(),
                               desc="Preprocessing stock data"):
            try:
                # Make a copy to avoid modifying the original data
                processed_df = df.copy()

                # Calculate additional price features
                processed_df = self._calculate_price_features(processed_df)

                # Calculate returns
                processed_df = self._calculate_returns(processed_df)

                # Handle missing values
                processed_df = self._handle_missing_values(processed_df)

                # Scale features if specified
                if self.scaling_method:
                    processed_df = self._scale_features(processed_df, ticker)

                # Store processed data
                processed_data[ticker] = processed_df

            except Exception as e:
                logger.error(
                    f"Error preprocessing data for {ticker}: {str(e)}")

        # Save processed data
        self._save_processed_data(processed_data)

        logger.info(
            f"Successfully preprocessed data for {len(processed_data)} tickers")
        return processed_data

    def _calculate_price_features(self, df):
        """Calculate additional price-based features.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with additional price features.
        """
        # Calculate typical price
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3

        # Calculate price ranges
        df['daily_range'] = df['High'] - df['Low']
        df['daily_range_pct'] = df['daily_range'] / df['Open'] * 100

        # Calculate price position within the range
        df['range_position'] = (df['Close'] - df['Low']) / df['daily_range']

        # Calculate log prices for statistical stability
        for col in ['Open', 'High', 'Low', 'Close', 'typical_price']:
            df[f'log_{col}'] = np.log(df[col])

        return df

    def _calculate_returns(self, df):
        """Calculate various return metrics.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with return metrics.
        """
        # Calculate daily returns
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close']).diff()

        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

        # Calculate returns over different periods
        for period in [5, 10, 20, 60, 120, 252]:
            if len(df) > period:
                df[f'return_{period}d'] = df['Close'].pct_change(
                    periods=period)

        return df

    def _handle_missing_values(self, df):
        """Handle missing values in the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with handled missing values.
        """
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logger.debug(f"Found {missing_count} missing values")

            # Apply specified fill method
            if self.fill_method == 'ffill':
                df = df.fillna(method='ffill')
                # Forward fill might leave NaNs at the beginning
                df = df.fillna(method='bfill')
            elif self.fill_method == 'bfill':
                df = df.fillna(method='bfill')
                # Backward fill might leave NaNs at the end
                df = df.fillna(method='ffill')
            elif self.fill_method == 'interpolate':
                df = df.interpolate(method='linear')
                # Interpolation might leave NaNs at the beginning and end
                df = df.fillna(method='ffill').fillna(method='bfill')
            else:
                logger.warning(
                    f"Unknown fill method: {self.fill_method}. Using forward fill.")
                df = df.fillna(method='ffill').fillna(method='bfill')

        # Check if there are still missing values
        remaining_missing = df.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(
                f"There are still {remaining_missing} missing values after filling")

        return df

    def _scale_features(self, df, ticker):
        """Scale numerical features using the specified method.

        Args:
            df (pd.DataFrame): Stock price DataFrame.
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with scaled features.
        """
        # Identify columns to scale (numerical columns excluding date and categorical)
        cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()

        # Skip scaling if no columns to scale
        if not cols_to_scale:
            return df

        # Create a copy to avoid modifying the original
        scaled_df = df.copy()

        # Select the appropriate scaler
        if self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}")
            return df

        # Fit and transform the data
        scaled_values = scaler.fit_transform(df[cols_to_scale])

        # Update the DataFrame with scaled values
        scaled_df[cols_to_scale] = scaled_values

        # Store the scaler for future use
        self.scalers[ticker] = {
            'scaler': scaler,
            'columns': cols_to_scale
        }

        return scaled_df

    def _save_processed_data(self, processed_data):
        """Save processed data to files.

        Args:
            processed_data (dict): Dictionary of processed DataFrames by ticker.
        """
        # Create directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)

        # Save each processed DataFrame
        for ticker, df in processed_data.items():
            df.to_csv(f'data/processed/{ticker}_processed.csv')

        # Save scalers for future use
        if self.scalers:
            os.makedirs('models/scalers', exist_ok=True)
            joblib.dump(self.scalers, 'models/scalers/feature_scalers.joblib')

        logger.info("Processed data saved to files")


# data_processing/feature_engineering.py

logger = logging.getLogger('investment_model.data_processing')

class FeatureEngineer:
    """Class for engineering financial features for analysis and modeling."""

    def __init__(self):
        """Initialize the feature engineer."""
        pass

    def engineer_features(self, processed_data):
        """Engineer features for all stocks.

        Args:
            processed_data (dict): Dictionary of preprocessed stock DataFrames by ticker.

        Returns:
            dict: Dictionary of DataFrames with engineered features by ticker.
        """
        logger.info("Engineering features for stock data")

        engineered_data = {}

        for ticker, df in tqdm(processed_data.items(), desc="Engineering features"):
            try:
                # Make a copy to avoid modifying the original data
                engineered_df = df.copy()

                # Add technical indicators
                engineered_df = self._add_technical_indicators(engineered_df)

                # Add volatility features
                engineered_df = self._add_volatility_features(engineered_df)

                # Add momentum features
                engineered_df = self._add_momentum_features(engineered_df)

                # Add volume features
                engineered_df = self._add_volume_features(engineered_df)

                # Add cycle features
                engineered_df = self._add_cycle_features(engineered_df)

                # Add statistical features
                engineered_df = self._add_statistical_features(engineered_df)

                # Store engineered data
                engineered_data[ticker] = engineered_df

            except Exception as e:
                logger.error(f"Error engineering features for {ticker}: {str(e)}")

        logger.info(f"Successfully engineered features for {len(engineered_data)} tickers")
        return engineered_data

    def _add_technical_indicators(self, df):
        """Add technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with technical indicators.
        """
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['Close'].values, timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['Close'].values, timeperiod=period)

        # Price relative to moving averages
        for period in [20, 50, 200]:
            df[f'Close_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1
            df[f'Close_to_EMA_{period}'] = df['Close'] / df[f'EMA_{period}'] - 1

        # Bollinger Bands
        for period in [20, 50]:
            for std_dev in [2.0, 2.5]:
                upper, middle, lower = talib.BBANDS(
                    df['Close'].values,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev,
                    matype=0
                )
                df[f'BBUpper_{period}_{std_dev}'] = upper
                df[f'BBMiddle_{period}_{std_dev}'] = middle
                df[f'BBLower_{period}_{std_dev}'] = lower

                # Calculate Bollinger Band width and %B
                df[f'BBWidth_{period}_{std_dev}'] = (upper - lower) / middle
                df[f'BBPct_{period}_{std_dev}'] = (df['Close'] - lower) / (upper - lower)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['Close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist

        # RSI
        for period in [9, 14, 21]:
            df[f'RSI_{period}'] = talib.RSI(df['Close'].values, timeperiod=period)

        # Stochastic Oscillator
        for period_k, period_d in [(5, 3), (14, 3)]:
            slowk, slowd = talib.STOCH(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                fastk_period=period_k,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=period_d,
                slowd_matype=0
            )
            df[f'StochK_{period_k}'] = slowk
            df[f'StochD_{period_k}_{period_d}'] = slowd

        # Commodity Channel Index (CCI)
        for period in [20, 40]:
            df[f'CCI_{period}'] = talib.CCI(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )

        # Average Directional Index (ADX)
        for period in [14, 28]:
            df[f'ADX_{period}'] = talib.ADX(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )

        # Parabolic SAR
        df['SAR'] = talib.SAR(
            df['High'].values,
            df['Low'].values,
            acceleration=0.02,
            maximum=0.2
        )

        # Ichimoku Cloud
        high9 = df['High'].rolling(window=9).max()
        low9 = df['Low'].rolling(window=9).min()
        high26 = df['High'].rolling(window=26).max()
        low26 = df['Low'].rolling(window=26).min()
        high52 = df['High'].rolling(window=52).max()
        low52 = df['Low'].rolling(window=52).min()

        df['Ichimoku_Tenkan'] = (high9 + low9) / 2
        df['Ichimoku_Kijun'] = (high26 + low26) / 2
        df['Ichimoku_SenkouA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        df['Ichimoku_SenkouB'] = ((high52 + low52) / 2).shift(26)
        df['Ichimoku_Chikou'] = df['Close'].shift(-26)

        # Fibonacci Retracement Levels (based on 52-week high and low)
        if len(df) >= 252:  # Approximately 1 year of trading days
            high52w = df['High'].rolling(window=252).max()
            low52w = df['Low'].rolling(window=252).min()
            range52w = high52w - low52w

            for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                df[f'Fib_{int(level*1000)}'] = high52w - (range52w * level)

        return df

    def _add_volatility_features(self, df):
        """Add volatility-related features to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with volatility features.
        """
        # Historical Volatility
        for period in [5, 10, 20, 60]:
            df[f'Volatility_{period}d'] = df['log_return'].rolling(window=period).std() * np.sqrt(252)

        # Average True Range (ATR)
        for period in [14, 28]:
            df[f'ATR_{period}'] = talib.ATR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )
            # Normalized ATR (ATR as % of price)
            df[f'NATR_{period}'] = talib.NATR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )

        # Volatility Ratio (short-term vs long-term)
        if len(df) >= 60:
            df['Volatility_Ratio'] = df['Volatility_20d'] / df['Volatility_60d']

        # High-Low Range as percentage
        df['HL_Range_Pct'] = (df['High'] - df['Low']) / df['Open'] * 100

        # Bollinger Band Width (already added in technical indicators)

        # Keltner Channels
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        for period in [20]:
            df[f'KC_Middle_{period}'] = talib.EMA(typical_price.values, timeperiod=period)
            atr = talib.ATR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )
            df[f'KC_Upper_{period}'] = df[f'KC_Middle_{period}'] + (atr * 2)
            df[f'KC_Lower_{period}'] = df[f'KC_Middle_{period}'] - (atr * 2)
            df[f'KC_Width_{period}'] = (df[f'KC_Upper_{period}'] - df[f'KC_Lower_{period}']) / df[f'KC_Middle_{period}']

        # Ulcer Index (UI)
        if len(df) >= 14:
            r_max = df['Close'].rolling(window=14).max()
            pct_drawdown = 100 * ((df['Close'] - r_max) / r_max)
            df['Ulcer_Index'] = np.sqrt((pct_drawdown**2).rolling(window=14).mean())

        return df

    def _add_momentum_features(self, df):
        """Add momentum-related features to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFramepd.DataFrame: DataFrame with momentum features.
        """
        # Rate of Change (ROC)
        for period in [5, 10, 21, 63, 126, 252]:
            if len(df) > period:
                df[f'ROC_{period}'] = talib.ROC(df['Close'].values, timeperiod=period)

        # Momentum
        for period in [10, 21]:
            df[f'Momentum_{period}'] = talib.MOM(df['Close'].values, timeperiod=period)

        # Relative Strength Index (RSI already calculated in technical indicators)

        # Stochastic Oscillator (already calculated in technical indicators)

        # Williams %R
        for period in [14, 28]:
            df[f'WillR_{period}'] = talib.WILLR(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )

        # Percentage Price Oscillator (PPO)
        df['PPO'] = talib.PPO(
            df['Close'].values,
            fastperiod=12,
            slowperiod=26,
            matype=0
        )

        # Money Flow Index (MFI)
        if 'Volume' in df.columns:
            for period in [14, 28]:
                df[f'MFI_{period}'] = talib.MFI(
                    df['High'].values,
                    df['Low'].values,
                    df['Close'].values,
                    df['Volume'].values,
                    timeperiod=period
                )

        # Aroon Indicator
        for period in [14, 25]:
            aroon_down, aroon_up = talib.AROON(
                df['High'].values,
                df['Low'].values,
                timeperiod=period
            )
            df[f'AroonUp_{period}'] = aroon_up
            df[f'AroonDown_{period}'] = aroon_down
            df[f'AroonOsc_{period}'] = aroon_up - aroon_down

        # Chande Momentum Oscillator (CMO)
        for period in [14, 21]:
            df[f'CMO_{period}'] = talib.CMO(df['Close'].values, timeperiod=period)

        # True Strength Index (TSI)
        # Custom implementation since not available in talib
        if len(df) >= 40:  # Need at least 40 days of data
            # First smoothing
            momentum = df['Close'].diff()
            abs_momentum = momentum.abs()
            smooth_momentum = momentum.ewm(span=25, adjust=False).mean()
            smooth_abs_momentum = abs_momentum.ewm(span=25, adjust=False).mean()

            # Second smoothing
            double_smooth_momentum = smooth_momentum.ewm(span=13, adjust=False).mean()
            double_smooth_abs_momentum = smooth_abs_momentum.ewm(span=13, adjust=False).mean()

            # TSI calculation
            df['TSI'] = 100 * (double_smooth_momentum / double_smooth_abs_momentum)

        # Directional Movement Index (DMI) components
        for period in [14]:
            plus_di = talib.PLUS_DI(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )
            minus_di = talib.MINUS_DI(
                df['High'].values,
                df['Low'].values,
                df['Close'].values,
                timeperiod=period
            )
            df[f'PlusDI_{period}'] = plus_di
            df[f'MinusDI_{period}'] = minus_di
            df[f'DI_Spread_{period}'] = plus_di - minus_di

        return df

    def _add_volume_features(self, df):
        """Add volume-related features to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with volume features.
        """
        if 'Volume' not in df.columns:
            logger.warning("Volume data not available, skipping volume features")
            return df

        # Volume Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'Volume_SMA_{period}'] = talib.SMA(df['Volume'].values, timeperiod=period)

        # Volume relative to moving average
        for period in [20, 50]:
            df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']

        # On-Balance Volume (OBV)
        df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values)

        # Chaikin Oscillator
        df['Chaikin_Oscillator'] = talib.ADOSC(
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            df['Volume'].values,
            fastperiod=3,
            slowperiod=10
        )

        # Ease of Movement (EOM)
        distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = (df['Volume'] / 1000000) / ((df['High'] - df['Low']))
        df['EOM'] = distance_moved / box_ratio
        df['EOM_14'] = df['EOM'].rolling(window=14).mean()

        # Money Flow Volume (MFV)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        # Money Flow Ratio
        positive_money_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=14).sum()
        negative_money_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=14).sum()

        # Avoid division by zero
        negative_money_flow = negative_money_flow.replace(0, 1e-10)
        df['Money_Flow_Ratio'] = positive_money_flow / negative_money_flow

        # Volume Price Trend (VPT)
        df['VPT'] = (df['Volume'] * df['daily_return']).cumsum()

        # Price-Volume Trend (PVT)
        df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
        df['PVT'] = df['PVT'].cumsum()

        # Volume Weighted Average Price (VWAP)
        df['VWAP'] = (df['Volume'] * df['typical_price']).cumsum() / df['Volume'].cumsum()

        # Volume Oscillator
        if len(df) >= 26:
            df['Vol_Osc'] = 100 * ((df['Volume_SMA_5'] - df['Volume_SMA_20']) / df['Volume_SMA_20'])

        return df

    def _add_cycle_features(self, df):
        """Add cyclical features to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with cycle features.
        """
        # Hilbert Transform - Dominant Cycle Period
        if len(df) >= 40:  # Need enough data points
            df['HT_DCPeriod'] = talib.HT_DCPERIOD(df['Close'].values)

            # Hilbert Transform - Dominant Cycle Phase
            df['HT_DCPhase'] = talib.HT_DCPHASE(df['Close'].values)

            # Hilbert Transform - Phasor Components
            inphase, quadrature = talib.HT_PHASOR(df['Close'].values)
            df['HT_InPhase'] = inphase
            df['HT_Quadrature'] = quadrature

            # Hilbert Transform - SineWave
            sine, leadsine = talib.HT_SINE(df['Close'].values)
            df['HT_Sine'] = sine
            df['HT_LeadSine'] = leadsine

            # Hilbert Transform - Trend vs. Cycle Mode
            df['HT_Trendmode'] = talib.HT_TRENDMODE(df['Close'].values)

        # Day of week (numerical)
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek

            # Month
            df['month'] = df.index.month

            # Quarter
            df['quarter'] = df.index.quarter

            # Day of month
            df['day_of_month'] = df.index.day

            # Is month end
            df['is_month_end'] = df.index.is_month_end.astype(int)

            # Is quarter end
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

        return df

    def _add_statistical_features(self, df):
        """Add statistical features to the DataFrame.

        Args:
            df (pd.DataFrame): Stock price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with statistical features.
        """
        # Z-score (normalized deviation from moving average)
        for period in [20, 50]:
            rolling_mean = df['Close'].rolling(window=period).mean()
            rolling_std = df['Close'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['Close'] - rolling_mean) / rolling_std

        # Distance from 52-week high/low
        if len(df) >= 252:  # Approximately 1 year of trading days
            high52w = df['High'].rolling(window=252).max()
            low52w = df['Low'].rolling(window=252).min()

            df['pct_from_52w_high'] = (df['Close'] / high52w - 1) * 100
            df['pct_from_52w_low'] = (df['Close'] / low52w - 1) * 100

        # Skewness and Kurtosis of returns
        for period in [20, 60]:
            if len(df) >= period:
                df[f'return_skew_{period}d'] = df['daily_return'].rolling(window=period).skew()
                df[f'return_kurt_{period}d'] = df['daily_return'].rolling(window=period).kurt()

        # Autocorrelation of returns
        for lag in [1, 5]:
            for period in [20, 60]:
                if len(df) >= period + lag:
                    # Ensure the series is aligned properly for correlation calculation
                    returns = df['daily_return']
                    returns_lagged = returns.shift(lag)
                    rolling_corr = returns.rolling(window=period).corr(returns_lagged)
                    df[f'return_autocorr_{lag}_{period}d'] = rolling_corr

        # Rolling statistics
        for period in [10, 20, 50]:
            # Standard deviation
            df[f'close_std_{period}d'] = df['Close'].rolling(window=period).std()

            # Mean absolute deviation
            df[f'close_mad_{period}d'] = df['Close'].rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )

            # Range
            df[f'close_range_{period}d'] = df['Close'].rolling(window=period).apply(
                lambda x: x.max() - x.min()
            )

            # Coefficient of variation
            mean = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'close_cv_{period}d'] = std / mean

        # Hurst exponent (measure of long-term memory of time series)
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent for the time series."""
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]  # Slope of the log-log plot

        if len(df) >= 100:  # Need sufficient data for reliable calculation
            # Calculate Hurst exponent on rolling windows
            df['hurst_exponent_50d'] = df['Close'].rolling(window=50).apply(
                lambda x: hurst_exponent(x.values) if len(x.dropna()) >= 20 else np.nan
            )

        # Detrended price oscillator (DPO)
        for period in [20]:
            df[f'DPO_{period}'] = df['Close'] - df['Close'].rolling(window=period+1).mean().shift(int(period/2 + 1))

        return df