# backtesting/backtester.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger('backtesting')


class Backtester:
    """
    Backtesting class for evaluating the performance of the investment model
    using historical data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtester with configuration parameters.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing backtesting parameters
        """
        self.config = config

        # Extract backtesting parameters from config
        self.start_date = config.get('start_date', '2010-01-01')
        self.end_date = config.get('end_date',
                                   datetime.now().strftime('%Y-%m-%d'))
        self.rebalance_frequency = config.get('rebalance_frequency',
                                              'quarterly')
        self.initial_capital = config.get('initial_capital', 1000000)
        self.transaction_cost = config.get('transaction_cost',
                                           0.001)  # 10 basis points
        self.benchmark = config.get('benchmark', 'SPY')
        self.cash_allocation = config.get('cash_allocation',
                                          0.05)  # 5% cash reserve
        self.max_position_size = config.get('max_position_size',
                                            0.1)  # 10% max per position
        self.output_dir = config.get('output_dir', 'results/backtesting')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize results containers
        self.portfolio_history = None
        self.trade_history = None
        self.performance_metrics = None
        self.drawdowns = None

    def get_rebalance_dates(self, stock_data: pd.DataFrame) -> List[datetime]:
        """
        Generate rebalance dates based on the specified frequency.

        Args:
            stock_data (pd.DataFrame): Historical stock data with DateTimeIndex

        Returns:
            List[datetime]: List of rebalance dates
        """
        # Get all dates from the stock data
        all_dates = pd.Series(stock_data.index).dt.date.unique()
        all_dates = sorted(all_dates)

        # Generate rebalance dates based on frequency
        if self.rebalance_frequency == 'monthly':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='MS'  # Month Start
            )
        elif self.rebalance_frequency == 'quarterly':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='QS'  # Quarter Start
            )
        elif self.rebalance_frequency == 'semi-annual':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='6MS'  # Every 6 Month Start
            )
        elif self.rebalance_frequency == 'annual':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='AS'  # Year Start
            )
        else:
            raise ValueError(
                f"Unsupported rebalance frequency: {self.rebalance_frequency}")

        # Find closest trading days (forward-looking for each rebalance date)
        result_dates = []
        for rebalance_date in rebalance_dates:
            rebalance_date = rebalance_date.date()
            # Find the next available trading day
            for date in all_dates:
                if date >= rebalance_date:
                    result_dates.append(date)
                    break

        return result_dates

    def run_backtest(self, investment_model, historical_data: pd.DataFrame) -> \
    Dict[str, Any]:
        """
        Run the backtesting simulation using the investment model.

        Args:
            investment_model: Instance of the InvestmentModel class
            historical_data (pd.DataFrame): Historical price data for all stocks

        Returns:
            Dict[str, Any]: Dictionary with backtesting results and metrics
        """
        logger.info("Starting backtesting simulation")

        # Initialize portfolio
        portfolio = self._initialize_portfolio()

        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates(historical_data)

        # Setup tracking variables
        portfolio_history = []
        trade_history = []
        daily_returns = []

        # Track the positions and their values over time
        positions = {}
        cash = self.initial_capital * self.cash_allocation
        invested_capital = self.initial_capital * (1 - self.cash_allocation)

        # Setup benchmark tracking
        benchmark_history = []

        # Convert dates to datetime for easy manipulation
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        # Get all trading days
        trading_days = historical_data.index
        trading_days = trading_days[
            (trading_days >= start_date) & (trading_days <= end_date)]

        # Get benchmark data
        if self.benchmark in historical_data.columns.get_level_values(0):
            benchmark_prices = historical_data[self.benchmark]['Adj Close']
        else:
            logger.warning(
                f"Benchmark {self.benchmark} not found in data, using equal-weighted index")
            benchmark_prices = historical_data.xs('Adj Close', level=1,
                                                  axis=1).mean(axis=1)

        # Initialize benchmark investment
        benchmark_units = self.initial_capital / benchmark_prices.loc[
            trading_days[0]]

        # Run simulation for each trading day
        logger.info(f"Running backtest from {start_date} to {end_date}")
        for current_date in tqdm(trading_days, desc="Backtesting Progress"):
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Check if current date is a rebalance date
            is_rebalance_day = current_date.date() in rebalance_dates

            # If rebalance day, update portfolio allocations
            if is_rebalance_day:
                logger.info(f"Rebalancing portfolio on {current_date_str}")

                # Slice historical data up to current date (don't peek into future)
                current_data = historical_data.loc[:current_date]

                # Run the investment model to get optimal portfolio for current date
                # This passes a time-sliced version of data to the model
                optimal_portfolio, _ = self._run_model_for_period(
                    investment_model, current_data, current_date
                )

                # Calculate current portfolio value
                portfolio_value = cash
                for ticker, shares in positions.items():
                    if ticker in historical_data.columns.get_level_values(0):
                        price = historical_data.loc[
                            current_date, (ticker, 'Adj Close')]
                        portfolio_value += shares * price

                # Execute trades based on new optimal portfolio
                trades = self._rebalance_portfolio(
                    positions, optimal_portfolio,
                    portfolio_value, historical_data,
                    current_date
                )

                # Update cash after trades
                cash = trades['ending_cash']

                # Update positions
                positions = trades['new_positions']

                # Record trades
                for trade in trades['trades']:
                    trade_history.append({
                        'date': current_date_str,
                        'ticker': trade['ticker'],
                        'action': trade['action'],
                        'shares': trade['shares'],
                        'price': trade['price'],
                        'value': trade['value'],
                        'transaction_cost': trade['transaction_cost']
                    })

            # Calculate daily portfolio value
            daily_portfolio_value = cash
            daily_positions = {}

            for ticker, shares in positions.items():
                if ticker in historical_data.columns.get_level_values(0):
                    try:
                        price = historical_data.loc[
                            current_date, (ticker, 'Adj Close')]
                        position_value = shares * price
                        daily_portfolio_value += position_value
                        daily_positions[ticker] = {
                            'shares': shares,
                            'price': price,
                            'value': position_value,
                            'weight': position_value / daily_portfolio_value
                        }
                    except KeyError:
                        logger.warning(
                            f"Price data missing for {ticker} on {current_date_str}")

            # Calculate benchmark value
            benchmark_value = benchmark_units * benchmark_prices.loc[
                current_date]

            # Record daily portfolio snapshot
            portfolio_history.append({
                'date': current_date_str,
                'portfolio_value': daily_portfolio_value,
                'cash': cash,
                'cash_percentage': cash / daily_portfolio_value,
                'invested_value': daily_portfolio_value - cash,
                'benchmark_value': benchmark_value
            })

            # Calculate daily return
            if len(portfolio_history) > 1:
                prev_value = portfolio_history[-2]['portfolio_value']
                daily_return = (daily_portfolio_value / prev_value) - 1
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0)

            # Record benchmark snapshot
            benchmark_history.append({
                'date': current_date_str,
                'value': benchmark_value
            })

        # Convert tracking data to DataFrames
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)

        trades_df = pd.DataFrame(trade_history)
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df.set_index('date', inplace=True)

        benchmark_df = pd.DataFrame(benchmark_history)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        benchmark_df.set_index('date', inplace=True)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_df, benchmark_df, daily_returns
        )

        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(portfolio_df)

        # Store results
        self.portfolio_history = portfolio_df
        self.trade_history = trades_df
        self.performance_metrics = performance_metrics
        self.drawdowns = drawdowns

        # Generate reports
        self._generate_reports()

        logger.info("Backtesting simulation completed")

        return {
            'portfolio_history': portfolio_df,
            'trade_history': trades_df,
            'performance_metrics': performance_metrics,
            'drawdowns': drawdowns
        }

    def _initialize_portfolio(self) -> Dict[str, Any]:
        """
        Initialize an empty portfolio with the specified initial capital.

        Returns:
            Dict[str, Any]: Initial portfolio state
        """
        return {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }

    def _run_model_for_period(self, investment_model,
                              historical_data: pd.DataFrame,
                              current_date: datetime) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        """
        Run the investment model for a specific time period.

        Args:
            investment_model: Instance of the InvestmentModel class
            historical_data (pd.DataFrame): Historical data up to current date
            current_date (datetime): Current simulation date

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Optimal portfolio and evaluation results
        """
        # Save original config dates
        original_start_date = investment_model.config['data_collection'][
            'start_date']
        original_end_date = investment_model.config['data_collection'][
            'end_date']

        try:
            # Update model config to use data only up to current date
            # This prevents look-ahead bias
            training_start = (current_date - timedelta(days=365 * 5)).strftime(
                '%Y-%m-%d')  # 5 years of training data
            investment_model.config['data_collection'][
                'start_date'] = training_start
            investment_model.config['data_collection'][
                'end_date'] = current_date.strftime('%Y-%m-%d')

            # Create a modified version of the model's data collector that uses pre-loaded data
            original_collect_data = investment_model.data_collector.collect_data

            def mock_collect_data():
                # Filter data to the specified time period
                filtered_data = historical_data.loc[training_start:current_date]
                # Return stock data and empty fundamental data (or pre-loaded fundamentals if available)
                return filtered_data, {}

            # Temporarily replace the collect_data method
            investment_model.data_collector.collect_data = mock_collect_data

            # Run the model to get portfolio allocation
            optimal_portfolio, evaluation = investment_model.run()

            return optimal_portfolio, evaluation

        finally:
            # Restore original config and methods
            investment_model.config['data_collection'][
                'start_date'] = original_start_date
            investment_model.config['data_collection'][
                'end_date'] = original_end_date
            investment_model.data_collector.collect_data = original_collect_data

    def _rebalance_portfolio(self, current_positions: Dict[str, float],
                             optimal_portfolio: pd.DataFrame,
                             portfolio_value: float,
                             historical_data: pd.DataFrame,
                             current_date: datetime) -> Dict[str, Any]:
        """
        Rebalance the portfolio based on the optimal weights from the model.

        Args:
            current_positions (Dict[str, float]): Current portfolio positions
            optimal_portfolio (pd.DataFrame): Model-generated optimal portfolio
            portfolio_value (float): Current total portfolio value
            historical_data (pd.DataFrame): Historical price data
            current_date (datetime): Current simulation date

        Returns:
            Dict[str, Any]: Rebalancing results including trades and new positions
        """
        # Calculate target allocations
        allocated_capital = portfolio_value * (1 - self.cash_allocation)

        # Calculate target positions
        target_positions = {}
        for idx, row in optimal_portfolio.iterrows():
            ticker = idx
            weight = min(row['weight'],
                         self.max_position_size)  # Cap position size

            # Ensure ticker is in our historical data
            if ticker in historical_data.columns.get_level_values(0):
                try:
                    current_price = historical_data.loc[
                        current_date, (ticker, 'Adj Close')]
                    target_value = allocated_capital * weight
                    target_shares = target_value / current_price
                    target_positions[ticker] = target_shares
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Could not calculate target position for {ticker}: {e}")

        # Calculate trades needed
        trades = []
        cash_flow = 0
        transaction_costs = 0

        # First, sell positions that are not in the target portfolio
        for ticker, current_shares in current_positions.items():
            if ticker not in target_positions:
                # Sell all shares
                try:
                    current_price = historical_data.loc[
                        current_date, (ticker, 'Adj Close')]
                    trade_value = current_shares * current_price
                    transaction_cost = trade_value * self.transaction_cost

                    trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': current_shares,
                        'price': current_price,
                        'value': trade_value,
                        'transaction_cost': transaction_cost
                    })

                    cash_flow += trade_value - transaction_cost
                    transaction_costs += transaction_cost
                except KeyError:
                    logger.warning(
                        f"Missing price data for {ticker} on {current_date}")

        # Then adjust existing positions and add new ones
        for ticker, target_shares in target_positions.items():
            current_shares = current_positions.get(ticker, 0)
            delta_shares = target_shares - current_shares

            if abs(delta_shares) > 0.001:  # Avoid tiny trades
                try:
                    current_price = historical_data.loc[
                        current_date, (ticker, 'Adj Close')]
                    trade_value = abs(delta_shares) * current_price
                    transaction_cost = trade_value * self.transaction_cost

                    if delta_shares > 0:
                        action = 'BUY'
                        cash_flow -= (trade_value + transaction_cost)
                    else:
                        action = 'SELL'
                        cash_flow += (trade_value - transaction_cost)

                    trades.append({
                        'ticker': ticker,
                        'action': action,
                        'shares': abs(delta_shares),
                        'price': current_price,
                        'value': trade_value,
                        'transaction_cost': transaction_cost
                    })

                    transaction_costs += transaction_cost
                except KeyError:
                    logger.warning(
                        f"Missing price data for {ticker} on {current_date}")

        # Calculate new positions
        new_positions = {}
        for ticker, target_shares in target_positions.items():
            # Only include positions with non-zero shares
            if target_shares > 0:
                new_positions[ticker] = target_shares

        # Calculate ending cash
        starting_cash = portfolio_value - sum(
            current_positions.get(ticker, 0) * historical_data.loc[
                current_date, (ticker, 'Adj Close')]
            for ticker in current_positions
            if ticker in historical_data.columns.get_level_values(0)
        )
        ending_cash = starting_cash + cash_flow

        return {
            'trades': trades,
            'cash_flow': cash_flow,
            'transaction_costs': transaction_costs,
            'new_positions': new_positions,
            'ending_cash': ending_cash
        }

    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame,
                                       benchmark_df: pd.DataFrame,
                                       daily_returns: List[float]) -> Dict[
        str, float]:
        """
        Calculate performance metrics for the backtested portfolio.

        Args:
            portfolio_df (pd.DataFrame): Portfolio value history
            benchmark_df (pd.DataFrame): Benchmark value history
            daily_returns (List[float]): Daily portfolio returns

        Returns:
            Dict[str, float]: Performance metrics
        """
        # Convert lists to numpy arrays for calculations
        daily_returns = np.array(daily_returns)

        # Calculate portfolio returns
        portfolio_returns = portfolio_df[
            'portfolio_value'].pct_change().dropna()
        benchmark_returns = benchmark_df['value'].pct_change().dropna()

        # Annualized returns
        days_in_simulation = len(portfolio_df)
        years_in_simulation = days_in_simulation / 252  # Trading days in a year

        total_portfolio_return = (portfolio_df['portfolio_value'].iloc[-1] /
                                  portfolio_df['portfolio_value'].iloc[0]) - 1
        total_benchmark_return = (benchmark_df['value'].iloc[-1] /
                                  benchmark_df['value'].iloc[0]) - 1

        annualized_portfolio_return = (1 + total_portfolio_return) ** (
                    1 / years_in_simulation) - 1
        annualized_benchmark_return = (1 + total_benchmark_return) ** (
                    1 / years_in_simulation) - 1

        # Volatility (annualized)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

        # Sharpe Ratio (assuming risk-free rate of 0.02 or 2%)
        risk_free_rate = 0.02
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_returns = portfolio_returns - daily_risk_free
        sharpe_ratio = (
                                   excess_returns.mean() / portfolio_returns.std()) * np.sqrt(
            252)

        # Sortino Ratio (downside risk only)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (
                                        annualized_portfolio_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

        # Alpha (Jensen's Alpha)
        alpha = annualized_portfolio_return - (risk_free_rate + beta * (
                    annualized_benchmark_return - risk_free_rate))

        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = annualized_portfolio_return / abs(
            max_drawdown) if max_drawdown != 0 else 0

        # Win Rate
        winning_days = sum(r > 0 for r in portfolio_returns)
        win_rate = winning_days / len(portfolio_returns) if len(
            portfolio_returns) > 0 else 0

        # Information Ratio
        tracking_error = (
                                     portfolio_returns - benchmark_returns).std() * np.sqrt(
            252)
        information_ratio = (
                                        annualized_portfolio_return - annualized_benchmark_return) / tracking_error if tracking_error != 0 else 0

        # Turnover Rate
        if not self.trade_history.empty:
            total_trade_value = self.trade_history['value'].sum()
            avg_portfolio_value = portfolio_df['portfolio_value'].mean()
            turnover_rate = total_trade_value / (
                        avg_portfolio_value * 2 * years_in_simulation)
        else:
            turnover_rate = 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)

        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Compile metrics
        metrics = {
            'total_return': total_portfolio_return,
            'annualized_return': annualized_portfolio_return,
            'benchmark_total_return': total_benchmark_return,
            'benchmark_annualized_return': annualized_benchmark_return,
            'volatility': portfolio_volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'maximum_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'turnover_rate': turnover_rate,
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95,
            'excess_return': annualized_portfolio_return - annualized_benchmark_return
        }

        return metrics

    def _calculate_drawdowns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and analyze drawdowns in the portfolio.

        Args:
            portfolio_df (pd.DataFrame): Portfolio value history

        Returns:
            pd.DataFrame: Drawdown analysis
        """
        # Calculate returns
        portfolio_returns = portfolio_df[
            'portfolio_value'].pct_change().dropna()

        # Calculate drawdowns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1

        # Find drawdown periods
        is_drawdown = drawdowns < 0

        # Initialize variables for tracking drawdown periods
        drawdown_periods = []
        current_drawdown_start = None
        current_drawdown = 0
        recovery_date = None

        # Analyze each day
        for date, value in drawdowns.items():
            if value < 0:
                # In drawdown
                if current_drawdown_start is None:
                    # Start of a new drawdown
                    current_drawdown_start = date
                    current_drawdown = value
                elif value < current_drawdown:
                    # Deepening of existing drawdown
                    current_drawdown = value
            elif current_drawdown_start is not None:
                # Recovery from drawdown
                recovery_date = date

                # Record completed drawdown period
                drawdown_periods.append({
                    'start_date': current_drawdown_start,
                    'end_date': recovery_date,
                    'max_drawdown': current_drawdown,
                    'duration_days': (
                                recovery_date - current_drawdown_start).days,
                    'recovery_duration': (
                                recovery_date - current_drawdown_start).days
                })

                # Reset tracking variables
                current_drawdown_start = None
                current_drawdown = 0
                recovery_date = None

        # Handle ongoing drawdown at the end of backtest
        if current_drawdown_start is not None:
            drawdown_periods.append({
                'start_date': current_drawdown_start,
                'end_date': drawdowns.index[-1],
                'max_drawdown': current_drawdown,
                'duration_days': (
                            drawdowns.index[-1] - current_drawdown_start).days,
                'recovery_duration': None  # No recovery yet
            })

        # Convert to DataFrame
        drawdowns_df = pd.DataFrame(drawdown_periods)

        # Sort by max drawdown (absolute value)
        if not drawdowns_df.empty:
            drawdowns_df = drawdowns_df.sort_values('max_drawdown')

        return drawdowns_df

    def _generate_reports(self) -> None:
        """
        Generate performance reports and visualizations.
        """
        if self.portfolio_history is None or self.performance_metrics is None:
            logger.warning(
                "Cannot generate reports: no backtest results available")
            return

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save portfolio history
        self.portfolio_history.to_csv(
            f'{self.output_dir}/portfolio_history_{timestamp}.csv')

        # Save trade history
        if self.trade_history is not None and not self.trade_history.empty:
            self.trade_history.to_csv(
                f'{self.output_dir}/trade_history_{timestamp}.csv')

        # Save performance metrics
        pd.Series(self.performance_metrics).to_csv(
            f'{self.output_dir}/performance_metrics_{timestamp}.csv')

        # Save drawdowns
        if self.drawdowns is not None and not self.drawdowns.empty:
            self.drawdowns.to_csv(
                f'{self.output_dir}/drawdowns_{timestamp}.csv')

        # Generate performance plots
        self._plot_performance_charts(timestamp)

        logger.info(f"Reports generated with timestamp {timestamp}")

    def _plot_performance_charts(self, timestamp: str) -> None:
        """
        Create performance visualization charts.

        Args:
            timestamp (str): Timestamp for file naming
        """
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Portfolio Growth vs Benchmark
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_history.index,
                 self.portfolio_history['portfolio_value'],
                 label='Portfolio', linewidth=2)
        plt.plot(self.portfolio_history.index,
                 self.portfolio_history['benchmark_value'],
                 label='Benchmark', linewidth=2, linestyle='--')
        plt.title('Portfolio Growth vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/growth_comparison_{timestamp}.png')

        # 2. Drawdown Chart
        plt.figure(figsize=(12, 6))
        returns = self.portfolio_history['portfolio_value'].pct_change()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        plt.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
        plt.fill_between(drawdown.index, drawdown * 100, 0, color='red',
                         alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/drawdown_{timestamp}.png')

        # 3. Monthly Returns Heatmap
        plt.figure(figsize=(14, 8))
        returns = self.portfolio_history['portfolio_value'].pct_change()
        monthly_returns = returns.groupby(
            [returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns = monthly_returns.unstack()

        if not monthly_returns.empty:
            # Rename columns to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_returns.columns = [month_names[int(col) - 1] for col in
                                       monthly_returns.columns]

            # Create heatmap
            sns.heatmap(monthly_returns * 100, annot=True, fmt=".2f",
                        cmap="RdYlGn",
                        center=0, cbar_kws={'label': 'Monthly Return (%)'})
            plt.title('Monthly Returns (%)')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/monthly_returns_{timestamp}.png')

        # 4. Rolling Performance Metrics
        plt.figure(figsize=(16, 12))