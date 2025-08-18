# analysis/technical_analysis.py

import pandas as pd
import numpy as np
import talib
import logging
from tqdm import tqdm
import os

logger = logging.getLogger('investment_model.analysis')


class TechnicalAnalyzer:
    """Class for performing technical analysis on stock price data."""

    def __init__(self, output_dir='data/technical'):
        """Initialize the technical analyzer.

        Args:
            output_dir (str): Directory to save technical analysis results.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_indicators(self, processed_data_dict):
        """Calculate technical indicators for all stocks.

        Args:
            processed_data_dict (dict): Dictionary of processed stock DataFrames by ticker.

        Returns:
            dict: Dictionary of technical indicators and signals by ticker.
        """
        logger.info("Calculating technical indicators")

        technical_indicators = {}

        for ticker, df in tqdm(processed_data_dict.items(),
                               desc="Technical analysis"):
            try:
                # Calculate technical indicators
                indicators = self._calculate_stock_indicators(df)

                # Calculate technical signals
                signals = self._calculate_technical_signals(df, indicators)

                # Combine indicators and signals
                technical_indicators[ticker] = {**indicators, **signals}

            except Exception as e:
                logger.error(
                    f"Error calculating technical indicators for {ticker}: {str(e)}")

        # Save technical indicators
        self._save_technical_indicators(technical_indicators)

        logger.info(
            f"Successfully calculated technical indicators for {len(technical_indicators)} tickers")
        return technical_indicators

    def _calculate_stock_indicators(self, df):
        """Calculate technical indicators for a single stock.

        Args:
            df (pd.DataFrame): Processed stock price DataFrame.

        Returns:
            dict: Dictionary of calculated technical indicators.
        """
        indicators = {}

        # Extract price and volume data
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_price = df['Open'].values
        volume = df['Volume'].values if 'Volume' in df.columns else None

        # Moving Averages
        if len(close) >= 200:
            indicators['SMA_50'] = talib.SMA(close, timeperiod=50)[-1]
            indicators['SMA_200'] = talib.SMA(close, timeperiod=200)[-1]
            indicators['EMA_20'] = talib.EMA(close, timeperiod=20)[-1]
            indicators['EMA_50'] = talib.EMA(close, timeperiod=50)[-1]
            indicators['EMA_200'] = talib.EMA(close, timeperiod=200)[-1]

            # Price relative to moving averages
            indicators['Price_to_SMA_50'] = close[-1] / indicators[
                'SMA_50'] - 1
            indicators['Price_to_SMA_200'] = close[-1] / indicators[
                'SMA_200'] - 1
            indicators['Price_to_EMA_20'] = close[-1] / indicators[
                'EMA_20'] - 1
            indicators['Price_to_EMA_50'] = close[-1] / indicators[
                'EMA_50'] - 1

            # Golden Cross / Death Cross
            indicators['Golden_Cross'] = indicators['SMA_50'] > indicators[
                'SMA_200']
            indicators['Death_Cross'] = indicators['SMA_50'] < indicators[
                'SMA_200']

            # Moving Average Convergence Divergence (MACD)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12,
                                                      slowperiod=26,
                                                      signalperiod=9)
            indicators['MACD'] = macd[-1]
            indicators['MACD_Signal'] = macd_signal[-1]
            indicators['MACD_Hist'] = macd_hist[-1]

            # Relative Strength Index (RSI)
            indicators['RSI_14'] = talib.RSI(close, timeperiod=14)[-1]

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20,
                                                nbdevup=2, nbdevdn=2, matype=0)
            indicators['BB_Upper'] = upper[-1]
            indicators['BB_Middle'] = middle[-1]
            indicators['BB_Lower'] = lower[-1]
            indicators['BB_Width'] = (upper[-1] - lower[-1]) / middle[-1]

            # Momentum Indicators
            indicators['ROC_10'] = talib.ROC(close, timeperiod=10)[
                -1]  # Rate of Change
            indicators['MOM_10'] = talib.MOM(close, timeperiod=10)[
                -1]  # Momentum

            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14,
                                       slowk_period=3, slowd_period=3)
            indicators['Stoch_K'] = slowk[-1]
            indicators['Stoch_D'] = slowd[-1]

            # Average Directional Index (ADX) - Trend strength
            indicators['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)[
                -1]

            # On-Balance Volume (OBV)
            if volume is not None:
                indicators['OBV'] = talib.OBV(close, volume)[-1]

            # Commodity Channel Index (CCI)
            indicators['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)[
                -1]

            # Aroon Oscillator
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            indicators['Aroon_Up'] = aroon_up[-1]
            indicators['Aroon_Down'] = aroon_down[-1]
            indicators['Aroon_Oscillator'] = aroon_up[-1] - aroon_down[-1]

            # Average True Range (ATR) - Volatility
            indicators['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)[
                -1]

            # Historical Volatility
            if len(close) > 20:
                returns = np.diff(np.log(close))
                indicators['Volatility_20'] = np.std(returns[-20:]) * np.sqrt(
                    252)  # Annualized

            # Parabolic SAR
            indicators['PSAR'] = \
            talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]

            # Ichimoku Cloud (simplified)
            nine_period_high = talib.MAX(high, timeperiod=9)
            nine_period_low = talib.MIN(low, timeperiod=9)
            indicators['Tenkan_sen'] = (nine_period_high[-1] + nine_period_low[
                -1]) / 2

            twenty_six_period_high = talib.MAX(high, timeperiod=26)
            twenty_six_period_low = talib.MIN(low, timeperiod=26)
            indicators['Kijun_sen'] = (twenty_six_period_high[-1] +
                                       twenty_six_period_low[-1]) / 2

        return indicators

    def _calculate_technical_signals(self, df, indicators):
        """Calculate technical signals based on indicators.

        Args:
            df (pd.DataFrame): Processed stock price DataFrame.
            indicators (dict): Dictionary of calculated technical indicators.

        Returns:
            dict: Dictionary of technical signals.
        """
        signals = {}
        close = df['Close'].values[-1]

        # Skip calculations if not enough data
        if len(df) < 200 or not indicators:
            return signals

        # Trend signals
        signals['Uptrend_MA'] = indicators['SMA_50'] > indicators['SMA_200']
        signals['Downtrend_MA'] = indicators['SMA_50'] < indicators['SMA_200']

        # MACD signals
        if 'MACD' in indicators and 'MACD_Signal' in indicators:
            signals['MACD_Bullish'] = indicators['MACD'] > indicators[
                'MACD_Signal']
            signals['MACD_Bearish'] = indicators['MACD'] < indicators[
                'MACD_Signal']
            signals['MACD_Bullish_Crossover'] = False
            signals['MACD_Bearish_Crossover'] = False

            # Calculate crossovers if we have enough history
            if len(df) > 2:
                macd_prev, macd_signal_prev = talib.MACD(
                    df['Close'].values[:-1],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9)
                signals['MACD_Bullish_Crossover'] = (
                            indicators['MACD'] > indicators['MACD_Signal'] and
                            macd_prev[-1] <= macd_signal_prev[-1])
                signals['MACD_Bearish_Crossover'] = (
                            indicators['MACD'] < indicators['MACD_Signal'] and
                            macd_prev[-1] >= macd_signal_prev[-1])

        # RSI signals
        if 'RSI_14' in indicators:
            signals['RSI_Overbought'] = indicators['RSI_14'] > 70
            signals['RSI_Oversold'] = indicators['RSI_14'] < 30
            signals['RSI_Bullish'] = 40 < indicators['RSI_14'] < 60 and \
                                     indicators['RSI_14'] > indicators.get(
                'RSI_14_prev', 0)
            signals['RSI_Bearish'] = 40 < indicators['RSI_14'] < 60 and \
                                     indicators['RSI_14'] < indicators.get(
                'RSI_14_prev', 100)

        # Bollinger Band signals
        if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
            signals['Price_Above_Upper_BB'] = close > indicators['BB_Upper']
            signals['Price_Below_Lower_BB'] = close < indicators['BB_Lower']
            signals['BB_Squeeze'] = indicators['BB_Width'] < indicators.get(
                'BB_Width_prev', float('inf'))  # Volatility contraction

        # Stochastic signals
        if 'Stoch_K' in indicators and 'Stoch_D' in indicators:
            signals['Stoch_Overbought'] = indicators['Stoch_K'] > 80
            signals['Stoch_Oversold'] = indicators['Stoch_K'] < 20
            signals['Stoch_Bullish_Crossover'] = indicators['Stoch_K'] > \
                                                 indicators['Stoch_D'] and \
                                                 indicators['Stoch_K'] < 80
            signals['Stoch_Bearish_Crossover'] = indicators['Stoch_K'] < \
                                                 indicators['Stoch_D'] and \
                                                 indicators['Stoch_K'] > 20

        # ADX Trend Strength
        if 'ADX_14' in indicators:
            signals['Strong_Trend'] = indicators['ADX_14'] > 25
            signals['Weak_Trend'] = indicators['ADX_14'] < 20

        # Support and Resistance
        signals['Near_Support'] = False
        signals['Near_Resistance'] = False
        if len(df) > 50:
            recent_lows = df['Low'].rolling(window=10).min().iloc[-50:].values
            recent_highs = df['High'].rolling(window=10).max().iloc[
                           -50:].values

            # Find support levels (local minimums)
            potential_support = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < \
                        recent_lows[i + 1]:
                    potential_support.append(recent_lows[i])

            # Find resistance levels (local maximums)
            potential_resistance = []
            for i in range(1, len(recent_highs) - 1):
                if recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > \
                        recent_highs[i + 1]:
                    potential_resistance.append(recent_highs[i])

            # Check if current price is near support or resistance
            threshold = 0.03  # 3% threshold

            for level in potential_support:
                if abs(close - level) / close < threshold:
                    signals['Near_Support'] = True
                    break

            for level in potential_resistance:
                if abs(close - level) / close < threshold:
                    signals['Near_Resistance'] = True
                    break

        # Long-term Investment Signals (for buy and forget strategy)
        signals['Long_Term_Bullish'] = (
                signals.get('Uptrend_MA', False) and
                indicators.get('Price_to_SMA_200',
                               -1) > -0.05 and  # Price is near or above 200-day MA
                indicators.get('RSI_14', 50) > 40 and  # RSI is not too low
                indicators.get('ADX_14', 0) > 20  # Trend has some strength
        )

        # Volatility signals for long-term investors
        if 'Volatility_20' in indicators:
            signals['Low_Volatility'] = indicators[
                                            'Volatility_20'] < 0.25  # Less than 25% annualized volatility
            signals['High_Volatility'] = indicators[
                                             'Volatility_20'] > 0.40  # More than 40% annualized volatility

        # Combined score for long-term investing
        signals['Long_Term_Score'] = self._calculate_long_term_score(
            indicators, signals)

        return signals

    def _calculate_long_term_score(self, indicators, signals):
        """Calculate a score for long-term investment potential.

        Args:
            indicators (dict): Dictionary of calculated technical indicators.
            signals (dict): Dictionary of technical signals.

        Returns:
            float: Score from 0-100 where higher is better for long-term investing.
        """
        score = 50  # Neutral starting point

        # Trend strength
        if signals.get('Uptrend_MA', False):
            score += 15
        elif signals.get('Downtrend_MA', False):
            score -= 15

        # Distance from moving averages
        if 'Price_to_SMA_200' in indicators:
            # Favor stocks near but above their 200-day MA
            if 0 < indicators['Price_to_SMA_200'] < 0.1:
                score += 10
            elif indicators['Price_to_SMA_200'] > 0.2:
                score -= 5  # Potential overextension
            elif indicators['Price_to_SMA_200'] < -0.1:
                score -= 10  # Significantly below long-term average

        # RSI - favor neutral to slightly bullish
        if 'RSI_14' in indicators:
            rsi = indicators['RSI_14']
            if 40 <= rsi <= 60:
                score += 10  # Healthy range for long-term
            elif rsi > 70:
                score -= 10  # Overbought
            elif rsi < 30:
                score -= 5  # Oversold might present opportunity but also risk

        # Volatility
        if 'Volatility_20' in indicators:
            vol = indicators['Volatility_20']
            if vol < 0.25:
                score += 10  # Low volatility favored for long-term investing
            elif vol > 0.4:
                score -= 10  # High volatility represents risk

        # MACD
        if signals.get('MACD_Bullish', False):
            score += 5

        # ADX - trend strength
        if 'ADX_14' in indicators:
            adx = indicators['ADX_14']
            if adx > 25:
                score += 5  # Strong trend
            elif adx < 15:
                score -= 5  # Very weak trend

        # Support/Resistance
        if signals.get('Near_Support', False):
            score += 5
        if signals.get('Near_Resistance', False):
            score -= 5

        # Ensure score is between 0 and 100
        return max(0, min(100, score))

    def _save_technical_indicators(self, technical_indicators):
        """Save technical indicators to disk.

        Args:
            technical_indicators (dict): Dictionary of technical indicators by ticker.
        """
        # Convert to DataFrame for easier analysis
        results = []
        for ticker, indicators in technical_indicators.items():
            row = {'ticker': ticker}
            row.update(indicators)
            results.append(row)

        if results:
            df = pd.DataFrame(results)
            output_path = os.path.join(self.output_dir,
                                       'technical_indicators.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Saved technical indicators to {output_path}")

    def get_technical_ratings(self, technical_indicators, min_score=60):
        """Get technical ratings for stocks based on long-term score.

        Args:
            technical_indicators (dict): Dictionary of technical indicators by ticker.
            min_score (int): Minimum score to consider a stock technically sound.

        Returns:
            dict: Dictionary of technical ratings by ticker.
        """
        ratings = {}

        for ticker, indicators in technical_indicators.items():
            long_term_score = indicators.get('Long_Term_Score', 0)

            if long_term_score >= min_score:
                rating = 'strong_buy' if long_term_score >= 80 else 'buy'
            elif long_term_score <= 30:
                rating = 'sell'
            else:
                rating = 'hold'

            ratings[ticker] = {
                'technical_rating': rating,
                'technical_score': long_term_score,
                'is_uptrend': indicators.get('Uptrend_MA', False),
                'is_volatile': indicators.get('High_Volatility', False),
                'rsi_level': indicators.get('RSI_14', 50)
            }

        return ratings