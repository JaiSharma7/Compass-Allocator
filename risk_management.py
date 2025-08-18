# analysis/risk_management.py

import pandas as pd
import numpy as np
import logging
import os
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logger = logging.getLogger('investment_model.risk_management')


class RiskManager:
    """Class for analyzing and managing investment risks."""

    def __init__(self, market_index='SPY'):
        """Initialize the risk manager.

        Args:
            market_index (str): Ticker symbol for the market index to use as benchmark
        """
        self.market_index = market_index
        self.risk_thresholds = {
            'low_risk': {
                'volatility': 0.15,  # Annualized volatility
                'max_drawdown': 0.15,  # Maximum drawdown
                'var_95': 0.02,  # 95% Value at Risk (daily)
                'beta': 0.8,  # Market beta
                'concentration': 0.05,  # Max single position size
                'sector_exposure': 0.25,  # Max single sector exposure
                'liquidity_score': 8,  # Min liquidity score (1-10)
                'correlation': 0.7,  # Max correlation with portfolio
                'tail_risk': 0.03,  # Expected shortfall
                'credit_risk_score': 8  # Min credit risk score (1-10)
            },
            'medium_risk': {
                'volatility': 0.25,
                'max_drawdown': 0.25,
                'var_95': 0.03,
                'beta': 1.2,
                'concentration': 0.08,
                'sector_exposure': 0.35,
                'liquidity_score': 6,
                'correlation': 0.8,
                'tail_risk': 0.05,
                'credit_risk_score': 6
            },
            'high_risk': {
                'volatility': 0.35,
                'max_drawdown': 0.35,
                'var_95': 0.05,
                'beta': 1.5,
                'concentration': 0.12,
                'sector_exposure': 0.45,
                'liquidity_score': 4,
                'correlation': 0.9,
                'tail_risk': 0.08,
                'credit_risk_score': 4
            }
        }

    def analyze_portfolio_risk(self, portfolio, price_data,
                               fundamental_data=None):
        """Analyze risk metrics for a portfolio.

        Args:
            portfolio (dict): Dictionary with tickers as keys and weights as values
            price_data (dict): Dictionary with tickers as keys and price history DataFrames as values
            fundamental_data (dict, optional): Dictionary with fundamental data by ticker

        Returns:
            dict: Dictionary of portfolio risk metrics
        """
        logger.info("Analyzing portfolio risk metrics")

        # Ensure market index is in price data
        if self.market_index not in price_data:
            logger.warning(
                f"Market index {self.market_index} not found in price data")
            market_returns = None
        else:
            market_prices = price_data[self.market_index]['Close']
            market_returns = market_prices.pct_change().dropna()

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio,
                                                              price_data)
        if portfolio_returns.empty:
            logger.error(
                "Could not calculate portfolio returns, insufficient data")
            return {}

        # Calculate individual stock risk metrics
        stock_risk_metrics = {}
        for ticker, weight in tqdm(portfolio.items(),
                                   desc="Analyzing individual stock risks"):
            if ticker not in price_data:
                logger.warning(f"Price data for {ticker} not found, skipping")
                continue

            stock_risk_metrics[ticker] = self._calculate_stock_risk_metrics(
                ticker,
                price_data[ticker],
                market_returns,
                weight,
                fundamental_data.get(ticker) if fundamental_data else None
            )

        # Calculate portfolio-level risk metrics
        portfolio_risk = self._calculate_portfolio_risk_metrics(
            portfolio,
            portfolio_returns,
            market_returns,
            stock_risk_metrics
        )

        # Add concentration risk metrics
        portfolio_risk.update(
            self._calculate_concentration_risk(portfolio, stock_risk_metrics))

        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(portfolio_risk)
        portfolio_risk['risk_assessment'] = risk_assessment

        # Save risk report
        self._save_risk_report(portfolio_risk, stock_risk_metrics)

        return portfolio_risk

    def _calculate_portfolio_returns(self, portfolio, price_data):
        """Calculate historical returns for the portfolio.

        Args:
            portfolio (dict): Dictionary with tickers as keys and weights as values
            price_data (dict): Dictionary with tickers as keys and price history DataFrames as values

        Returns:
            pd.Series: Series of portfolio returns
        """
        # Get returns for each stock
        returns_dict = {}
        for ticker in portfolio:
            if ticker in price_data:
                stock_prices = price_data[ticker]['Close']
                returns_dict[ticker] = stock_prices.pct_change().dropna()

        if not returns_dict:
            return pd.Series()

        # Combine into a DataFrame
        returns_df = pd.DataFrame(returns_dict)

        # Handle missing data
        returns_df = returns_df.dropna(how='all')
        returns_df = returns_df.fillna(0)  # Replace remaining NaN with 0

        # Calculate portfolio returns
        portfolio_weights = {}
        total_weight = sum(weight for ticker, weight in portfolio.items() if
                           ticker in returns_df.columns)

        # Normalize weights to sum to 1 for available tickers
        for ticker, weight in portfolio.items():
            if ticker in returns_df.columns:
                portfolio_weights[
                    ticker] = weight / total_weight if total_weight > 0 else 0

        # Calculate weighted returns
        weighted_returns = pd.Series(0.0, index=returns_df.index)
        for ticker, weight in portfolio_weights.items():
            weighted_returns += returns_df[ticker] * weight

        return weighted_returns

    def _calculate_stock_risk_metrics(self, ticker, price_data, market_returns,
                                      weight, fundamental_data=None):
        """Calculate risk metrics for an individual stock.

        Args:
            ticker (str): Stock ticker symbol
            price_data (pd.DataFrame): Price history DataFrame
            market_returns (pd.Series): Market index returns
            weight (float): Stock weight in portfolio
            fundamental_data (dict, optional): Fundamental data for this stock

        Returns:
            dict: Dictionary of risk metrics
        """
        # Extract closing prices and calculate returns
        stock_prices = price_data['Close']
        stock_returns = stock_prices.pct_change().dropna()

        # Return empty dict if not enough data
        if len(stock_returns) < 20:  # Need at least 20 data points for meaningful analysis
            logger.warning(f"Insufficient return data for {ticker}")
            return {}

        # Calculate basic statistical measures
        daily_volatility = stock_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Calculate drawdown
        drawdown = self._calculate_drawdowns(stock_prices)
        max_drawdown = abs(drawdown.min() if not drawdown.empty else 0)

        # Calculate Value at Risk (VaR)
        var_95 = self._calculate_var(stock_returns, confidence=0.95)
        var_99 = self._calculate_var(stock_returns, confidence=0.99)

        # Calculate Expected Shortfall (Conditional VaR)
        es_95 = self._calculate_expected_shortfall(stock_returns,
                                                   confidence=0.95)

        # Calculate beta and alpha if market data available
        beta = np.nan
        alpha = np.nan
        r_squared = np.nan

        if market_returns is not None:
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns],
                                     axis=1).dropna()
            if len(aligned_data) > 20:  # Need sufficient data for regression
                X = aligned_data.iloc[:, 1].values.reshape(-1,
                                                           1)  # Market returns
                y = aligned_data.iloc[:, 0].values  # Stock returns

                # Add constant for intercept
                X = np.hstack([np.ones((X.shape[0], 1)), X])

                # OLS regression
                try:
                    # Use statsmodels-like approach with numpy
                    betas, residuals, rank, s = np.linalg.lstsq(X, y,
                                                                rcond=None)
                    alpha = betas[0] * 252  # Annualize alpha
                    beta = betas[1]

                    # Calculate R-squared
                    y_mean = np.mean(y)
                    ss_total = np.sum((y - y_mean) ** 2)
                    ss_residual = np.sum(residuals ** 2) if len(
                        residuals) > 0 else np.sum((y - np.dot(X, betas)) ** 2)
                    r_squared = 1 - (
                                ss_residual / ss_total) if ss_total != 0 else 0
                except:
                    logger.warning(f"Error calculating beta for {ticker}")

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(price_data)

        # Calculate tail risk indicators
        skewness = stock_returns.skew()
        kurtosis = stock_returns.kurtosis()

        # Calculate credit risk score if fundamental data available
        credit_risk_score = 5  # Default neutral score
        if fundamental_data:
            credit_risk_score = self._calculate_credit_risk_score(
                fundamental_data)

        return {
            'volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'weight': weight,
            'liquidity_score': liquidity_score,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'credit_risk_score': credit_risk_score
        }

    def _calculate_portfolio_risk_metrics(self, portfolio, portfolio_returns,
                                          market_returns, stock_risk_metrics):
        """Calculate portfolio-level risk metrics.

        Args:
            portfolio (dict): Dictionary with tickers as keys and weights as values
            portfolio_returns (pd.Series): Portfolio historical returns
            market_returns (pd.Series): Market index returns
            stock_risk_metrics (dict): Dictionary of risk metrics by ticker

        Returns:
            dict: Dictionary of portfolio risk metrics
        """
        # Calculate basic statistical measures
        daily_volatility = portfolio_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Calculate portfolio Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = portfolio_returns - risk_free_rate
        sharpe_ratio = (
                                   excess_returns.mean() / excess_returns.std()) * np.sqrt(
            252)

        # Calculate drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        drawdown = self._calculate_drawdowns(cumulative_returns)
        max_drawdown = abs(drawdown.min() if not drawdown.empty else 0)

        # Calculate VaR and Expected Shortfall
        var_95 = self._calculate_var(portfolio_returns, confidence=0.95)
        var_99 = self._calculate_var(portfolio_returns, confidence=0.99)
        es_95 = self._calculate_expected_shortfall(portfolio_returns,
                                                   confidence=0.95)

        # Calculate portfolio beta
        portfolio_beta = np.nan
        if market_returns is not None:
            # Align data
            aligned_data = pd.concat([portfolio_returns, market_returns],
                                     axis=1).dropna()
            if len(aligned_data) > 20:
                X = aligned_data.iloc[:, 1].values.reshape(-1,
                                                           1)  # Market returns
                y = aligned_data.iloc[:, 0].values  # Portfolio returns

                # Add constant for intercept
                X = np.hstack([np.ones((X.shape[0], 1)), X])

                # OLS regression
                try:
                    betas, residuals, rank, s = np.linalg.lstsq(X, y,
                                                                rcond=None)
                    portfolio_beta = betas[1]
                except:
                    logger.warning("Error calculating portfolio beta")

        # Calculate downside deviation (semi-deviation)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)

        # Calculate Sortino ratio
        sortino_ratio = np.nan
        if downside_deviation > 0:
            sortino_ratio = (
                                        portfolio_returns.mean() - risk_free_rate) * 252 / downside_deviation

        # Calculate Calmar ratio (annualized return / max drawdown)
        annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
        calmar_ratio = np.nan
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown

        # Calculate skewness and kurtosis
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()

        # Calculate weighted average of stock-specific metrics
        weighted_metrics = {
            'beta': 0,
            'credit_risk_score': 0,
            'liquidity_score': 0
        }

        total_weight = 0
        for ticker, metrics in stock_risk_metrics.items():
            weight = portfolio.get(ticker, 0)
            total_weight += weight

            for key in weighted_metrics.keys():
                if key in metrics and not np.isnan(metrics[key]):
                    weighted_metrics[key] += metrics[key] * weight

        # Normalize by total weight
        if total_weight > 0:
            for key in weighted_metrics:
                weighted_metrics[key] /= total_weight

        return {
            'volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'beta': weighted_metrics['beta'],
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'downside_deviation': downside_deviation,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'liquidity_score': weighted_metrics['liquidity_score'],
            'credit_risk_score': weighted_metrics['credit_risk_score']
        }

    def _calculate_concentration_risk(self, portfolio, stock_risk_metrics):
        """Calculate concentration risk metrics.

        Args:
            portfolio (dict): Dictionary with tickers as keys and weights as values
            stock_risk_metrics (dict): Dictionary of risk metrics by ticker

        Returns:
            dict: Dictionary of concentration risk metrics
        """
        # Calculate standard concentration metrics
        weights = np.array(list(portfolio.values()))

        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(weights ** 2)

        # Get top 5 position weights
        top_positions = sorted(portfolio.items(), key=lambda x: x[1],
                               reverse=True)[:5]
        top5_concentration = sum(weight for _, weight in top_positions)

        # Create dummy sector allocation (in a real system, this would come from actual sector data)
        # For demonstration, we'll assign random sectors
        sectors = {
            'Technology': 0,
            'Healthcare': 0,
            'Financials': 0,
            'Consumer': 0,
            'Energy': 0,
            'Industrials': 0,
            'Other': 0
        }

        # Assign a random sector to each stock (in a real implementation, use actual sector data)
        np.random.seed(42)  # For reproducibility
        stock_sectors = {}
        for ticker in portfolio:
            sector_list = list(sectors.keys())
            stock_sectors[ticker] = sector_list[
                np.random.randint(0, len(sector_list))]

        # Calculate sector exposures
        for ticker, weight in portfolio.items():
            sector = stock_sectors.get(ticker, 'Other')
            sectors[sector] += weight

        # Find highest sector concentration
        max_sector = max(sectors.items(), key=lambda x: x[1])
        max_sector_name, max_sector_exposure = max_sector

        return {
            'hhi': hhi,
            'top5_concentration': top5_concentration,
            'max_position': max(weights) if len(weights) > 0 else 0,
            'max_sector': max_sector_name,
            'max_sector_exposure': max_sector_exposure,
            'sector_allocation': sectors
        }

    def _generate_risk_assessment(self, portfolio_risk):
        """Generate overall risk assessment based on calculated metrics.

        Args:
            portfolio_risk (dict): Dictionary of portfolio risk metrics

        Returns:
            dict: Risk assessment with categorizations and recommendations
        """
        # Categorize risk level for each metric
        risk_levels = {}

        for metric, value in portfolio_risk.items():
            if metric in ['volatility', 'max_drawdown', 'var_95', 'beta']:
                if value <= self.risk_thresholds['low_risk'][metric]:
                    risk_levels[metric] = 'low'
                elif value <= self.risk_thresholds['medium_risk'][metric]:
                    risk_levels[metric] = 'medium'
                else:
                    risk_levels[metric] = 'high'

        # Check concentration risks
        if portfolio_risk.get('max_position', 0) <= \
                self.risk_thresholds['low_risk']['concentration']:
            risk_levels['concentration'] = 'low'
        elif portfolio_risk.get('max_position', 0) <= \
                self.risk_thresholds['medium_risk']['concentration']:
            risk_levels['concentration'] = 'medium'
        else:
            risk_levels['concentration'] = 'high'

        if portfolio_risk.get('max_sector_exposure', 0) <= \
                self.risk_thresholds['low_risk']['sector_exposure']:
            risk_levels['sector_exposure'] = 'low'
        elif portfolio_risk.get('max_sector_exposure', 0) <= \
                self.risk_thresholds['medium_risk']['sector_exposure']:
            risk_levels['sector_exposure'] = 'medium'
        else:
            risk_levels['sector_exposure'] = 'high'

        # Calculate weighted average risk score
        risk_values = {'low': 1, 'medium': 2, 'high': 3}
        risk_weights = {
            'volatility': 0.2,
            'max_drawdown': 0.2,
            'var_95': 0.15,
            'beta': 0.15,
            'concentration': 0.15,
            'sector_exposure': 0.15
        }

        weighted_risk_score = 0
        total_weight = 0

        for metric, level in risk_levels.items():
            if metric in risk_weights:
                weighted_risk_score += risk_values[level] * risk_weights[
                    metric]
                total_weight += risk_weights[metric]

        if total_weight > 0:
            weighted_risk_score /= total_weight

        # Determine overall risk level
        if weighted_risk_score < 1.67:
            overall_risk = 'low'
        elif weighted_risk_score < 2.33:
            overall_risk = 'medium'
        else:
            overall_risk = 'high'

        # Generate recommendations
        recommendations = []

        if risk_levels.get('concentration', 'low') == 'high':
            recommendations.append(
                "Consider reducing position sizes for top holdings to decrease concentration risk.")

        if risk_levels.get('sector_exposure', 'low') == 'high':
            recommendations.append(
                f"Reduce exposure to {portfolio_risk.get('max_sector', 'dominant sector')} to increase diversification.")

        if risk_levels.get('volatility', 'low') == 'high':
            recommendations.append(
                "Portfolio shows high volatility. Consider adding uncorrelated assets or defensive positions.")

        if risk_levels.get('max_drawdown', 'low') == 'high':
            recommendations.append(
                "High maximum drawdown detected. Consider implementing stop-loss strategies or hedging positions.")

        if risk_levels.get('beta', 'low') == 'high' and portfolio_risk.get(
                'beta', 1) > 1.3:
            recommendations.append(
                "Portfolio has high market sensitivity. Consider reducing beta by adding defensive stocks or fixed income.")

        if portfolio_risk.get('sharpe_ratio', 0) < 0.5:
            recommendations.append(
                "Poor risk-adjusted returns. Review holdings for potential replacements with better return/risk profiles.")

        return {
            'risk_levels': risk_levels,
            'overall_risk': overall_risk,
            'risk_score': weighted_risk_score,
            'recommendations': recommendations
        }

    def _calculate_drawdowns(self, prices):
        """Calculate drawdowns for a price series.

        Args:
            prices (pd.Series): Series of prices

        Returns:
            pd.Series: Series of drawdowns
        """
        # Calculate running maximum
        running_max = prices.cummax()

        # Calculate drawdowns
        drawdowns = (prices / running_max) - 1

        return drawdowns

    def _calculate_var(self, returns, confidence=0.95):
        """Calculate Value at Risk.

        Args:
            returns (pd.Series): Series of returns
            confidence (float): Confidence level

        Returns:
            float: Value at Risk
        """
        # Sort returns
        sorted_returns = sorted(returns)

        # Find index at specified confidence level
        index = int((1 - confidence) * len(sorted_returns))

        # Return VaR (as positive number for easier interpretation)
        return abs(sorted_returns[index])

    def _calculate_expected_shortfall(self, returns, confidence=0.95):
        """Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns (pd.Series): Series of returns
            confidence (float): Confidence level

        Returns:
            float: Expected Shortfall
        """
        # Sort returns
        sorted_returns = sorted(returns)

        # Find index at specified confidence level
        index = int((1 - confidence) * len(sorted_returns))

        # Calculate ES as mean of losses beyond VaR
        tail_losses = sorted_returns[:index]

        # Return ES (as positive number for easier interpretation)
        return abs(np.mean(tail_losses))

    def _calculate_liquidity_score(self, price_data):
        """Calculate liquidity score based on volume and price data.

        Args:
            price_data (pd.DataFrame): Price history DataFrame with Volume

        Returns:
            float: Liquidity score (1-10, higher is more liquid)
        """
        # Check if volume data is available
        if 'Volume' not in price_data.columns or price_data[
            'Volume'].isnull().all():
            return 5  # Neutral score if no volume data

        # Use recent data (last 30 trading days)
        recent_data = price_data.tail(30)

        # Calculate average daily volume
        avg_volume = recent_data['Volume'].mean()

        # Calculate average daily dollar volume
        avg_dollar_volume = (
                    recent_data['Volume'] * recent_data['Close']).mean()

        # Basic score based on dollar volume (simplified)
        if avg_dollar_volume > 50000000:  # $50M+ daily volume
            return 10
        elif avg_dollar_volume > 10000000:  # $10M+
            return 9
        elif avg_dollar_volume > 5000000:  # $5M+
            return 8
        elif avg_dollar_volume > 1000000:  # $1M+
            return 7
        elif avg_dollar_volume > 500000:  # $500K+
            return 6
        elif avg_dollar_volume > 100000:  # $100K+
            return 5
        elif avg_dollar_volume > 50000:  # $50K+
            return 4
        elif avg_dollar_volume > 10000:  # $10K+
            return 3
        elif avg_dollar_volume > 5000:  # $5K+
            return 2
        else:
            return 1

    def _calculate_credit_risk_score(self, fundamental_data):
        """Calculate credit risk score based on fundamental data.

        Args:
            fundamental_data (dict): Fundamental data for a stock

        Returns:
            float: Credit risk score (1-10, higher is better credit quality)
        """
        # Extract relevant metrics if available
        metrics = fundamental_data.get('metrics', {})

        interest_coverage = metrics.get('Interest_Coverage', np.nan)
        debt_to_equity = metrics.get('Debt_to_Equity', np.nan)
        current_ratio = metrics.get('Current_Ratio', np.nan)
        debt_to_ebitda = metrics.get('Debt_to_EBITDA', np.nan)
        altman_z = metrics.get('Altman_Z_Score', np.nan)

        # Calculate score components
        scores = []

        if not np.isnan(interest_coverage):
            if interest_coverage > 15:
                scores.append(10)
            elif interest_coverage > 10:
                scores.append(9)
            elif interest_coverage > 6:
                scores.append(8)
            elif interest_coverage > 4:
                scores.append(7)
            elif interest_coverage > 3:
                scores.append(6)
            elif interest_coverage > 2:
                scores.append(5)
            elif interest_coverage > 1.5:
                scores.append(4)
            elif interest_coverage > 1:
                scores.append(3)
            else:
                scores.append(1)

        if not np.isnan(debt_to_equity):
            if debt_to_equity < 0.1:
                scores.append(10)
            elif debt_to_equity < 0.3:
                scores.append(9)
            elif debt_to_equity < 0.5:
                scores.append(8)
            elif debt_to_equity < 0.75:
                scores.append(7)
            elif debt_to_equity < 1:
                scores.append(6)
            elif debt_to_equity < 1.5:
                scores.append(5)
            elif debt_to_equity < 2:
                scores.append(4)
            elif debt_to_equity < 3:
                scores.append(3)
            elif debt_to_equity < 4:
                scores.append(2)
            else:
                scores.append(1)

        if not np.isnan(current_ratio):
            if current_ratio > 3:
                scores.append(10)
            elif current_ratio > 2.5:
                scores.append(9)
            elif current_ratio > 2:
                scores.append(8)
            elif current_ratio > 1.5:
                scores.append(7)
            elif current_ratio > 1.2:
                scores.append(6)
            elif current_ratio > 1:
                scores.append(5)
            elif current_ratio > 0.8:
                scores.append(4)
            elif current_ratio > 0.6:
                scores.append(3)
            elif current_ratio > 0.4:
                scores.append(2)
            else:
                scores.append(1)

        if not np.isnan(debt_to_ebitda):
            if debt_to_ebitda < 1:
                scores.append(10)
            elif debt_to_ebitda < 2:
                scores.append(9)
            elif debt_to_ebitda < 3:
                scores.append(8)
            elif debt_to_ebitda < 4:
                scores.append(7)
            elif debt_to_ebitda < 5:
                scores.append(6)
            elif debt_to_ebitda < 6:
                scores.append(5)
            elif debt_to_ebitda < 7:
                scores.append(4)
            elif debt_to_ebitda < 8:
                scores.append(3)
            elif debt_to_ebitda < 10:
                scores.append(2)
            else:
                scores.append(1)

        if not np.isnan(altman_z):
            if altman_z > 3:
                scores.append(10)
            elif altman_z > 2.7:
                scores.append(9)
            elif altman_z > 2.4:
                scores.append(8)
            elif altman_z > 2.1:
                scores.append(7)
            elif altman_z > 1.8:
                scores.append(6)
            elif altman_z > 1.5:
                scores.append(5)
            elif altman_z > 1.2:
                scores.append(4)
            elif altman_z > 0.9:
                scores.append(3)
            elif altman_z > 0.6:
                scores.append(2)
            else:
                scores.append(1)

        # Return average score
        if len(scores) > 0:
            return np.mean(scores)
        else:
            return 5  # Neutral score if no data

    def _save_risk_report(self, portfolio_risk, stock_risk_metrics):
        """Save risk analysis results to files.

        Args:
            portfolio_risk (dict): Dictionary of portfolio risk metrics
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        # Create directory if it doesn't exist
        os.makedirs('results/risk', exist_ok=True)

        # Save portfolio risk metrics
        portfolio_df = pd.DataFrame([portfolio_risk])
        portfolio_df.to_csv('results/risk/portfolio_risk.csv', index=False)

        # Save stock risk metrics
        stock_df = pd.DataFrame()
        for ticker, metrics in stock_risk_metrics.items():
            ticker_metrics = pd.Series(metrics, name=ticker)
            stock_df = pd.concat([stock_df, ticker_metrics.to_frame().T])

        stock_df.to_csv('results/risk/stock_risk.csv')

        # Generate risk report
        self._generate_risk_report(portfolio_risk, stock_risk_metrics)

        logger.info("Risk analysis results saved to files")

    def _generate_risk_report(self, portfolio_risk, stock_risk_metrics):
        """Generate a comprehensive risk report with visualizations.

        Args:
            portfolio_risk (dict): Dictionary of portfolio risk metrics
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        logger.info("Generating comprehensive risk report")

        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f'results/risk/report_{timestamp}'
        os.makedirs(report_dir, exist_ok=True)

        # Create report summary text file
        with open(f'{report_dir}/risk_summary.txt', 'w') as f:
            f.write("PORTFOLIO RISK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall risk assessment
            f.write("RISK ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            assessment = portfolio_risk.get('risk_assessment', {})

            f.write(
                f"Overall Risk Level: {assessment.get('overall_risk', 'N/A').upper()}\n")
            f.write(
                f"Risk Score: {assessment.get('risk_score', 0):.2f}/3.00\n\n")

            # Risk level breakdown
            f.write("Risk Metrics:\n")
            for metric, level in assessment.get('risk_levels', {}).items():
                f.write(
                    f"- {metric.replace('_', ' ').title()}: {level.upper()}\n")
            f.write("\n")

            # Key portfolio metrics
            f.write("KEY PORTFOLIO METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Volatility (Annualized): {portfolio_risk.get('volatility', 0):.2%}\n")
            f.write(
                f"Maximum Drawdown: {portfolio_risk.get('max_drawdown', 0):.2%}\n")
            f.write(
                f"Value at Risk (95%): {portfolio_risk.get('var_95', 0):.2%}\n")
            f.write(
                f"Expected Shortfall (95%): {portfolio_risk.get('expected_shortfall_95', 0):.2%}\n")
            f.write(f"Portfolio Beta: {portfolio_risk.get('beta', 0):.2f}\n")
            f.write(
                f"Sharpe Ratio: {portfolio_risk.get('sharpe_ratio', 0):.2f}\n")
            f.write(
                f"Sortino Ratio: {portfolio_risk.get('sortino_ratio', 0):.2f}\n")
            f.write(
                f"Calmar Ratio: {portfolio_risk.get('calmar_ratio', 0):.2f}\n\n")

            # Concentration metrics
            f.write("CONCENTRATION METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Herfindahl-Hirschman Index: {portfolio_risk.get('hhi', 0):.4f}\n")
            f.write(
                f"Top 5 Positions Concentration: {portfolio_risk.get('top5_concentration', 0):.2%}\n")
            f.write(
                f"Maximum Position Size: {portfolio_risk.get('max_position', 0):.2%}\n")
            f.write(
                f"Largest Sector: {portfolio_risk.get('max_sector', 'N/A')}\n")
            f.write(
                f"Largest Sector Exposure: {portfolio_risk.get('max_sector_exposure', 0):.2%}\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for rec in assessment.get('recommendations', []):
                f.write(f"- {rec}\n")

            if not assessment.get('recommendations', []):
                f.write("- No specific recommendations at this time.\n")
            f.write("\n")

            # Individual stock risks
            f.write("INDIVIDUAL STOCK RISK METRICS\n")
            f.write("-" * 30 + "\n")

            # Sort stocks by risk (volatility)
            sorted_stocks = sorted(stock_risk_metrics.items(),
                                   key=lambda x: x[1].get('volatility', 0),
                                   reverse=True)

            for ticker, metrics in sorted_stocks:
                if not metrics:  # Skip if empty
                    continue

                weight = metrics.get('weight', 0)
                volatility = metrics.get('volatility', 0)
                beta = metrics.get('beta', 0)
                var = metrics.get('var_95', 0)

                f.write(f"{ticker} (Weight: {weight:.2%}):\n")
                f.write(f"  Volatility: {volatility:.2%}\n")
                f.write(f"  Beta: {beta:.2f}\n")
                f.write(f"  VaR (95%): {var:.2%}\n")
                f.write(
                    f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
                f.write("\n")

        # Generate visualizations
        self._generate_risk_visualizations(report_dir, portfolio_risk,
                                           stock_risk_metrics)

        logger.info(f"Risk report generated in directory: {report_dir}")

    def _generate_risk_visualizations(self, report_dir, portfolio_risk,
                                      stock_risk_metrics):
        """Generate visualizations for risk report.

        Args:
            report_dir (str): Directory to save visualizations
            portfolio_risk (dict): Dictionary of portfolio risk metrics
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        # Create figures directory
        fig_dir = f'{report_dir}/figures'
        os.makedirs(fig_dir, exist_ok=True)

        # 1. Risk metrics radar chart
        self._plot_risk_radar(fig_dir, portfolio_risk)

        # 2. Sector allocation pie chart
        self._plot_sector_allocation(fig_dir, portfolio_risk)

        # 3. Stock volatility vs. beta scatter plot
        self._plot_volatility_beta_scatter(fig_dir, stock_risk_metrics)

        # 4. VaR comparison chart
        self._plot_var_comparison(fig_dir, stock_risk_metrics)

        # 5. Position size bar chart
        self._plot_position_sizes(fig_dir, stock_risk_metrics)

        logger.info(f"Risk visualizations saved to {fig_dir}")

    def _plot_risk_radar(self, fig_dir, portfolio_risk):
        """Create radar chart of key risk metrics.

        Args:
            fig_dir (str): Directory to save figure
            portfolio_risk (dict): Dictionary of portfolio risk metrics
        """
        try:
            # Define risk categories and values
            categories = ['Volatility', 'Max Drawdown', 'VaR (95%)',
                          'Beta', 'Concentration', 'Sector Exposure']

            # Risk thresholds (convert to list keeping the same order as categories)
            low_vals = [
                self.risk_thresholds['low_risk']['volatility'],
                self.risk_thresholds['low_risk']['max_drawdown'],
                self.risk_thresholds['low_risk']['var_95'],
                self.risk_thresholds['low_risk']['beta'],
                self.risk_thresholds['low_risk']['concentration'],
                self.risk_thresholds['low_risk']['sector_exposure']
            ]

            # Actual values
            values = [
                portfolio_risk.get('volatility', 0),
                portfolio_risk.get('max_drawdown', 0),
                portfolio_risk.get('var_95', 0),
                portfolio_risk.get('beta', 0),
                portfolio_risk.get('max_position', 0),  # Concentration
                portfolio_risk.get('max_sector_exposure', 0)  # Sector exposure
            ]

            # Normalize values relative to low risk threshold
            norm_values = [min(v / t, 2) for v, t in zip(values, low_vals)]

            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)

            # Number of variables
            N = len(categories)

            # Angle for each variable
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Add the normalized values
            norm_values += norm_values[:1]  # Close the loop

            # Plot
            ax.plot(angles, norm_values, linewidth=2, linestyle='solid')
            ax.fill(angles, norm_values, alpha=0.4)

            # Add reference circles (1.0 = low risk threshold)
            plt.yticks([0.5, 1.0, 1.5, 2.0], ['0.5x', '1.0x', '1.5x', '2.0x'],
                       color="grey", size=8)
            plt.ylim(0, 2)

            # Set category labels
            plt.xticks(angles[:-1], categories, size=10)

            # Add title
            plt.title(
                'Portfolio Risk Metrics (Relative to Low Risk Threshold)',
                size=12, y=1.1)

            # Save figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/risk_radar.png', dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generating risk radar chart: {e}")

    def _plot_sector_allocation(self, fig_dir, portfolio_risk):
        """Create pie chart of sector allocation.

        Args:
            fig_dir (str): Directory to save figure
            portfolio_risk (dict): Dictionary of portfolio risk metrics
        """
        try:
            # Get sector allocation
            sector_allocation = portfolio_risk.get('sector_allocation', {})

            if not sector_allocation:
                logger.warning("No sector allocation data available")
                return

            # Create data for pie chart
            labels = []
            sizes = []

            for sector, weight in sector_allocation.items():
                if weight > 0.01:  # Only include sectors with >1% allocation
                    labels.append(sector)
                    sizes.append(weight)

            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
            plt.title('Portfolio Sector Allocation', size=14)

            # Save figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/sector_allocation.png', dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generating sector allocation chart: {e}")

    def _plot_volatility_beta_scatter(self, fig_dir, stock_risk_metrics):
        """Create scatter plot of volatility vs. beta for all stocks.

        Args:
            fig_dir (str): Directory to save figure
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        try:
            tickers = []
            volatilities = []
            betas = []
            weights = []

            for ticker, metrics in stock_risk_metrics.items():
                if not metrics:
                    continue

                vol = metrics.get('volatility')
                beta = metrics.get('beta')
                weight = metrics.get('weight', 0)

                if vol is not None and not np.isnan(
                        vol) and beta is not None and not np.isnan(beta):
                    tickers.append(ticker)
                    volatilities.append(vol)
                    betas.append(beta)
                    weights.append(weight)

            if not tickers:
                logger.warning(
                    "No valid volatility/beta data for scatter plot")
                return

            # Create scatter plot
            plt.figure(figsize=(12, 8))

            # Size points by weight
            sizes = [w * 1000 for w in
                     weights]  # Scale weights for better visualization

            # Create scatter plot
            sc = plt.scatter(betas, volatilities, s=sizes, alpha=0.6,
                             c=volatilities, cmap='YlOrRd')

            # Add colorbar
            plt.colorbar(sc, label='Volatility')

            # Add ticker labels
            for i, ticker in enumerate(tickers):
                plt.annotate(ticker, (betas[i], volatilities[i]),
                             xytext=(5, 5), textcoords='offset points')

            # Add grid, title and labels
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title('Stock Risk Profile: Volatility vs. Beta', size=14)
            plt.xlabel('Beta', size=12)
            plt.ylabel('Volatility (Annualized)', size=12)

            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(1.0))

            # Reference lines
            plt.axhline(y=self.risk_thresholds['low_risk']['volatility'],
                        color='g', linestyle='--', alpha=0.5,
                        label=f"Low Risk Threshold: {self.risk_thresholds['low_risk']['volatility']:.0%}")
            plt.axhline(y=self.risk_thresholds['medium_risk']['volatility'],
                        color='orange', linestyle='--', alpha=0.5,
                        label=f"Medium Risk Threshold: {self.risk_thresholds['medium_risk']['volatility']:.0%}")
            plt.axvline(x=self.risk_thresholds['low_risk']['beta'],
                        color='g', linestyle=':', alpha=0.5,
                        label=f"Low Beta Threshold: {self.risk_thresholds['low_risk']['beta']:.1f}")
            plt.axvline(x=self.risk_thresholds['medium_risk']['beta'],
                        color='orange', linestyle=':', alpha=0.5,
                        label=f"Medium Beta Threshold: {self.risk_thresholds['medium_risk']['beta']:.1f}")

            plt.legend(loc='upper left')

            # Save figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/volatility_beta_scatter.png', dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generating volatility-beta scatter plot: {e}")

    def _plot_var_comparison(self, fig_dir, stock_risk_metrics):
        """Create bar chart comparing VaR across stocks.

        Args:
            fig_dir (str): Directory to save figure
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        try:
            # Extract VaR data
            tickers = []
            var_values = []

            for ticker, metrics in sorted(stock_risk_metrics.items(),
                                          key=lambda x: x[1].get('var_95',
                                                                 0) if x[
                                              1] else 0,
                                          reverse=True):
                if not metrics:
                    continue

                var = metrics.get('var_95')

                if var is not None and not np.isnan(var):
                    tickers.append(ticker)
                    var_values.append(var)

            # Limit to top 15 stocks by VaR
            if len(tickers) > 15:
                tickers = tickers[:15]
                var_values = var_values[:15]

            if not tickers:
                logger.warning("No valid VaR data for comparison chart")
                return

            # Create horizontal bar chart
            plt.figure(figsize=(10, 8))

            # Create colormap based on VaR values
            colors = plt.cm.YlOrRd(np.array(var_values) / max(var_values))

            # Create horizontal bar chart
            y_pos = range(len(tickers))
            bars = plt.barh(y_pos, var_values, align='center', color=colors,
                            alpha=0.8)

            # Add values to end of bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{var_values[i]:.2%}',
                         ha='left', va='center')

            # Customize chart
            plt.yticks(y_pos, tickers)
            plt.xlabel('Value at Risk (95%)')
            plt.title('Value at Risk (95%) Comparison Across Top Stocks')

            # Add reference line for low risk threshold
            plt.axvline(x=self.risk_thresholds['low_risk']['var_95'],
                        color='g', linestyle='--',
                        label=f"Low Risk Threshold: {self.risk_thresholds['low_risk']['var_95']:.1%}")

            # Add reference line for medium risk threshold
            plt.axvline(x=self.risk_thresholds['medium_risk']['var_95'],
                        color='orange', linestyle='--',
                        label=f"Medium Risk Threshold: {self.risk_thresholds['medium_risk']['var_95']:.1%}")

            plt.legend()

            # Format x-axis as percentage
            plt.gca().xaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(1.0))

            # Save figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/var_comparison.png', dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generating VaR comparison chart: {e}")

    def _plot_position_sizes(self, fig_dir, stock_risk_metrics):
        """Create bar chart of position sizes.

        Args:
            fig_dir (str): Directory to save figure
            stock_risk_metrics (dict): Dictionary of stock risk metrics
        """
        try:
            # Extract position weights
            tickers = []
            weights = []

            for ticker, metrics in sorted(stock_risk_metrics.items(),
                                          key=lambda x: x[1].get('weight',
                                                                 0) if x[
                                              1] else 0,
                                          reverse=True):
                if not metrics:
                    continue

                weight = metrics.get('weight')

                if weight is not None and not np.isnan(weight):
                    tickers.append(ticker)
                    weights.append(weight)

            if not tickers:
                logger.warning("No valid position size data")
                return

            # Create horizontal bar chart
            plt.figure(figsize=(10, 8))

            # Create colormap based on weight values
            norm = plt.Normalize(0, max(weights))
            colors = plt.cm.Blues(norm(weights))

            # Create horizontal bar chart
            y_pos = range(len(tickers))
            bars = plt.barh(y_pos, weights, align='center', color=colors,
                            alpha=0.8)

            # Add percentage values to end of bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                         f'{weights[i]:.2%}',
                         ha='left', va='center')

            # Customize chart
            plt.yticks(y_pos, tickers)
            plt.xlabel('Position Weight')
            plt.title('Portfolio Position Sizes')

            # Add reference line for concentration risk thresholds
            plt.axvline(x=self.risk_thresholds['low_risk']['concentration'],
                        color='g', linestyle='--',
                        label=f"Low Risk Threshold: {self.risk_thresholds['low_risk']['concentration']:.1%}")

            plt.axvline(x=self.risk_thresholds['medium_risk']['concentration'],
                        color='orange', linestyle='--',
                        label=f"Medium Risk Threshold: {self.risk_thresholds['medium_risk']['concentration']:.1%}")

            plt.legend()

            # Format x-axis as percentage
            plt.gca().xaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(1.0))

            # Save figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/position_sizes.png', dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Error generating position sizes chart: {e}")

    def simulate_stress_test(self, portfolio, price_data,
                             scenario="market_crash"):
        """Simulate stress test scenarios on the portfolio.

        Args:
            portfolio (dict): Dictionary with tickers as keys and weights as values
            price_data (dict): Dictionary with tickers as keys and price history DataFrames as values
            scenario (str): Stress test scenario to simulate
                          ("market_crash", "rate_hike", "sector_decline", "liquidity_crisis")

        Returns:
            dict: Stress test results
        """
        logger.info(f"Running stress test: {scenario}")

        # Calculate baseline portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio,
                                                              price_data)

        # Define scenario shocks
        scenarios = {
            "market_crash": {
                "market_return": -0.30,  # 30% market decline
                "sector_adjustments": {
                    "Technology": -0.40,
                    "Consumer": -0.25,
                    "Healthcare": -0.15,
                    "Utilities": -0.10,
                    "Energy": -0.35,
                    "Financials": -0.40,
                    "Industrials": -0.30,
                    "Other": -0.25
                },
                "volatility_multiplier": 3.0,
                "correlation_adjustment": 0.3  # Correlation increase
            },
            "rate_hike": {
                "market_return": -0.10,
                "sector_adjustments": {
                    "Technology": -0.15,
                    "Consumer": -0.10,
                    "Healthcare": -0.05,
                    "Utilities": -0.20,
                    "Energy": -0.05,
                    "Financials": -0.15,
                    "Industrials": -0.10,
                    "Other": -0.10
                },
                "volatility_multiplier": 1.5,
                "correlation_adjustment": 0.1
            },
            "sector_decline": {
                "market_return": -0.05,
                "sector_adjustments": {
                    "Technology": -0.25,
                    "Consumer": -0.05,
                    "Healthcare": -0.05,
                    "Utilities": 0.05,
                    "Energy": -0.05,
                    "Financials": -0.05,
                    "Industrials": -0.05,
                    "Other": -0.05
                },
                "volatility_multiplier": 1.2,
                "correlation_adjustment": 0.05
            },
            "liquidity_crisis": {
                "market_return": -0.15,
                "sector_adjustments": {
                    "Technology": -0.20,
                    "Consumer": -0.15,
                    "Healthcare": -0.10,
                    "Utilities": -0.05,
                    "Energy": -0.25,
                    "Financials": -0.30,
                    "Industrials": -0.15,
                    "Other": -0.15
                },
                "volatility_multiplier": 2.5,
                "correlation_adjustment": 0.2,
                "liquidity_factor": 0.5
                # Stocks with lower liquidity perform worse
            }
        }

        # Use default scenario if the requested one is not defined
        if scenario not in scenarios:
            logger.warning(
                f"Scenario {scenario} not defined, using market_crash instead")
            scenario = "market_crash"

        scenario_config = scenarios[scenario]

        # Create dummy sector allocation for demonstration
        # (In a real implementation, use actual sector data)
        np.random.seed(42)
        sector_list = list(scenario_config["sector_adjustments"].keys())
        stock_sectors = {}
        for ticker in portfolio:
            stock_sectors[ticker] = sector_list[
                np.random.randint(0, len(sector_list))]

        # Calculate stressed returns for each stock
        stressed_returns = {}
        for ticker, weight in portfolio.items():
            if ticker not in price_data:
                continue

            # Get historical returns and volatility
            stock_prices = price_data[ticker]['Close']
            stock_returns = stock_prices.pct_change().dropna()

            if len(stock_returns) < 20:
                continue

            # Get sector for this stock
            sector = stock_sectors.get(ticker, "Other")

            # Apply sector-specific shock
            sector_shock = scenario_config["sector_adjustments"].get(sector,
                                                                     scenario_config[
                                                                         "sector_adjustments"][
                                                                         "Other"])

            # Apply liquidity adjustment if defined
            liquidity_adjustment = 0
            if "liquidity_factor" in scenario_config and "Volume" in \
                    price_data[ticker].columns:
                liquidity_score = self._calculate_liquidity_score(
                    price_data[ticker])
                # Lower liquidity score means worse performance during crisis
                liquidity_adjustment = (10 - liquidity_score) / 10 * \
                                       scenario_config["liquidity_factor"]

            # Calculate stressed return
            stressed_return = scenario_config[
                                  "market_return"] + sector_shock - liquidity_adjustment

            # Store in dictionary
            stressed_returns[ticker] = stressed_return

        # Calculate portfolio stressed return
        portfolio_stressed_return = 0
        for ticker, weight in portfolio.items():
            if ticker in stressed_returns:
                portfolio_stressed_return += stressed_returns[ticker] * weight

        # Calculate impact on risk metrics
        var_stressed = portfolio_risk_metrics.get('var_95', 0) * \
                       scenario_config["volatility_multiplier"]

        # Prepare results
        results = {
            "scenario": scenario,
            "portfolio_return": portfolio_stressed_return,
            "ticker_returns": stressed_returns,
            "var_stressed": var_stressed,
            "max_loss_ticker": min(stressed_returns.items(), key=lambda x: x[
                1]) if stressed_returns else ("N/A", 0),
            "min_loss_ticker": max(stressed_returns.items(), key=lambda x: x[
                1]) if stressed_returns else ("N/A", 0)
        }

        logger.info(
            f"Stress test complete. Portfolio stressed return: {portfolio_stressed_return:.2%}")

        return results