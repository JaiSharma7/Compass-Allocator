# main.py - Entry point for the investment model

import os
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
import yaml
import joblib
from tqdm import tqdm

# Import custom modules
from data_collection import YahooFinanceDataCollector
from data_processing import DataPreprocessor
from fundamental_analysis import FundamentalAnalyzer
from technical_analysis import TechnicalAnalyzer
from backtester import Backtester
from ml_models.lstm_model import LSTMModel
from ml_models.arima_model import ARIMAModel
from ml_models.hmm_model import HMMModel
from ml_models.garch_model import GARCHModel
from portfolio.optimizer import PortfolioOptimizer
from risk_management import RiskManager
from utils.visualization import Visualizer
from utils.evaluation import PerformanceEvaluator
from utils.config_loader import ConfigLoader

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('investment_model')


class InvestmentModel:
    """Main class for the long-term S&P 500 investment model."""

    def __init__(self, config_path='config/config.yaml'):
        """Initialize the investment model.

        Args:
            config_path (str): Path to the configuration file.
        """
        logger.info("Initializing Investment Model")

        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()

        # Create output directories if they don't exist
        self._create_directories()

        # Initialize components
        self._initialize_components()

    def run_backtest(self, config=None):
        """
        Run a backtest of the investment model.

        Args:
            config (dict, optional): Backtest configuration. If None, uses default settings.

        Returns:
            Dict[str, Any]: Backtesting results
        """
        logger.info("Starting backtesting")

        # Use default config if none provided
        if config is None:
            config = {
                'start_date': '2018-01-01',  # 5 years of backtest by default
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'rebalance_frequency': 'quarterly',
                'initial_capital': 1000000,
                'transaction_cost': 0.001,  # 10 basis points
                'benchmark': 'SPY',
                'cash_allocation': 0.05,
                'max_position_size': 0.1,
                'output_dir': 'results/backtesting'
            }

        # Collect historical data for backtesting
        # We'll use existing data collector to get a longer history
        original_start_date = self.config['data_collection']['start_date']
        self.config['data_collection']['start_date'] = config['start_date']

        try:
            stock_data, _ = self.data_collector.collect_data()

            # Initialize backtester with config
            backtester = Backtester(config)

            # Run backtest using this investment model
            results = backtester.run_backtest(self, stock_data)

            logger.info("Backtesting completed")
            return results

        finally:
            # Restore original config
            self.config['data_collection']['start_date'] = original_start_date
    def _create_directories(self):
        """Create necessary directories for outputs."""
        dirs = [
            'data/raw',
            'data/processed',
            'models/lstm',
            'models/arima',
            'models/hmm',
            'models/garch',
            'results/predictions',
            'results/portfolio',
            'results/visualizations',
            'logs'
        ]

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")

    def _initialize_components(self):
        """Initialize all model components."""
        # Data collection
        self.data_collector = YahooFinanceDataCollector(
            self.config['data_collection']['start_date'],
            self.config['data_collection']['end_date'],
            self.config['data_collection']['tickers_file']
        )

        # Data preprocessing
        self.preprocessor = DataPreprocessor(
            self.config['data_processing']['fill_method'],
            self.config['data_processing']['scaling_method']
        )

        # Analysis
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()

        # ML Models
        self.lstm_model = LSTMModel(
            self.config['ml_models']['lstm']
        )
        self.arima_model = ARIMAModel(
            self.config['ml_models']['arima']
        )
        self.hmm_model = HMMModel(
            self.config['ml_models']['hmm']
        )
        self.garch_model = GARCHModel(
            self.config['ml_models']['garch']
        )

        # Portfolio management
        self.portfolio_optimizer = PortfolioOptimizer(
            self.config['portfolio']['optimizer']
        )
        self.risk_manager = RiskManager(
            self.config['portfolio']['risk']
        )

        # Utilities
        self.visualizer = Visualizer()
        self.evaluator = PerformanceEvaluator()

    def run(self):
        """Execute the full investment model workflow."""
        logger.info("Starting investment model workflow")

        # 1. Collect data
        logger.info("Collecting S&P 500 data")
        stock_data, fundamental_data = self.data_collector.collect_data()

        # 2. Preprocess data
        logger.info("Preprocessing data")
        processed_data = self.preprocessor.preprocess(stock_data)

        # 3. Analyze fundamentals
        logger.info("Analyzing fundamentals")
        fundamental_scores = self.fundamental_analyzer.analyze(
            fundamental_data)

        # 4. Perform technical analysis
        logger.info("Performing technical analysis")
        technical_indicators = self.technical_analyzer.calculate_indicators(
            processed_data)

        # 5. Train and predict with ML models
        logger.info("Training and predicting with ML models")

        # LSTM predictions
        logger.info("Training LSTM model")
        lstm_predictions = self.lstm_model.train_and_predict(processed_data)

        # ARIMA predictions
        logger.info("Training ARIMA model")
        arima_predictions = self.arima_model.train_and_predict(processed_data)

        # HMM market state predictions
        logger.info("Training HMM model")
        market_states = self.hmm_model.train_and_predict(processed_data)

        # GARCH volatility predictions
        logger.info("Training GARCH model")
        volatility_predictions = self.garch_model.train_and_predict(
            processed_data)

        # 6. Combine predictions and scores
        logger.info("Combining predictions and scores")
        combined_scores = self._combine_scores(
            fundamental_scores,
            technical_indicators,
            lstm_predictions,
            arima_predictions,
            market_states,
            volatility_predictions
        )

        # 7. Optimize portfolio
        logger.info("Optimizing portfolio")
        optimal_portfolio = self.portfolio_optimizer.optimize(
            processed_data,
            combined_scores,
            volatility_predictions
        )

        # 8. Apply risk management
        logger.info("Applying risk management")
        final_portfolio = self.risk_manager.manage_risk(
            optimal_portfolio,
            volatility_predictions,
            market_states
        )

        # 9. Evaluate and visualize results
        logger.info("Evaluating results")
        evaluation_results = self.evaluator.evaluate(
            final_portfolio,
            processed_data
        )

        logger.info("Visualizing results")
        self.visualizer.create_visualizations(
            processed_data,
            final_portfolio,
            lstm_predictions,
            arima_predictions,
            market_states,
            volatility_predictions,
            evaluation_results
        )

        # 10. Save results
        logger.info("Saving results")
        self._save_results(final_portfolio, evaluation_results)

        logger.info("Investment model workflow completed")
        return final_portfolio, evaluation_results

    def _combine_scores(self, fundamental_scores, technical_indicators,
                        lstm_predictions, arima_predictions,
                        market_states, volatility_predictions):
        """Combine various scores and predictions to rank stocks.

        Args:
            fundamental_scores (dict): Fundamental analysis scores.
            technical_indicators (dict): Technical indicators.
            lstm_predictions (dict): LSTM model predictions.
            arima_predictions (dict): ARIMA model predictions.
            market_states (dict): HMM model market state predictions.
            volatility_predictions (dict): GARCH model volatility predictions.

        Returns:
            pd.DataFrame: Combined scores for each stock.
        """
        logger.info("Combining scores from different models")

        # Convert all inputs to DataFrames if they aren't already
        tickers = list(fundamental_scores.keys())
        combined_data = pd.DataFrame(index=tickers)

        # Add fundamental scores
        for metric in ['value_score', 'growth_score', 'quality_score',
                       'momentum_score']:
            combined_data[f'fundamental_{metric}'] = [
                fundamental_scores[ticker].get(metric, 0) for ticker in
                tickers]

        # Add technical indicators
        for indicator in ['trend_strength', 'rsi_signal', 'macd_signal']:
            combined_data[f'technical_{indicator}'] = [
                technical_indicators[ticker].get(indicator, 0) for ticker in
                tickers]

        # Add ML predictions (normalized to scores)
        combined_data['lstm_return_prediction'] = [
            lstm_predictions[ticker].get('predicted_return', 0) for ticker in
            tickers]
        combined_data['arima_return_prediction'] = [
            arima_predictions[ticker].get('predicted_return', 0) for ticker in
            tickers]
        combined_data['market_state_score'] = [
            market_states[ticker].get('bullish_probability', 0.5) for ticker in
            tickers]
        combined_data['volatility_score'] = [1 / (
                    1 + volatility_predictions[ticker].get(
                'predicted_volatility', 1)) for ticker in tickers]

        # Calculate weighted combined score
        weights = self.config['scoring']['weights']
        combined_data['combined_score'] = (
                weights['fundamental'] * combined_data[
            ['fundamental_value_score', 'fundamental_growth_score',
             'fundamental_quality_score', 'fundamental_momentum_score']].mean(
            axis=1) +
                weights['technical'] * combined_data[
                    ['technical_trend_strength', 'technical_rsi_signal',
                     'technical_macd_signal']].mean(axis=1) +
                weights['lstm'] * combined_data['lstm_return_prediction'] +
                weights['arima'] * combined_data['arima_return_prediction'] +
                weights['market_state'] * combined_data['market_state_score'] +
                weights['volatility'] * combined_data['volatility_score']
        )

        # Normalize the combined score
        combined_data['combined_score'] = (combined_data['combined_score'] -
                                           combined_data[
                                               'combined_score'].min()) / \
                                          (combined_data[
                                               'combined_score'].max() -
                                           combined_data[
                                               'combined_score'].min())

        # Rank stocks
        combined_data['rank'] = combined_data['combined_score'].rank(
            ascending=False)

        return combined_data

    def _save_results(self, final_portfolio, evaluation_results):
        """Save the final portfolio and evaluation results.

        Args:
            final_portfolio (pd.DataFrame): The optimized portfolio.
            evaluation_results (dict): Performance evaluation results.
        """
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save portfolio
        final_portfolio.to_csv(f'results/portfolio/portfolio_{timestamp}.csv')

        # Save evaluation results
        with open(f'results/portfolio/evaluation_{timestamp}.yaml', 'w') as f:
            yaml.dump(evaluation_results, f)

        logger.info(f"Results saved with timestamp {timestamp}")


if __name__ == "__main__":
    model = InvestmentModel()
    portfolio, evaluation = model.run()

    # Display top 10 stocks in the portfolio
    print("\nTop 10 stocks in the optimized portfolio:")
    print(portfolio.sort_values('weight', ascending=False).head(10))

    # Display key performance metrics
    print("\nExpected portfolio performance:")
    print(
        f"Expected annual return: {evaluation['expected_return'] * 100:.2f}%")
    print(
        f"Expected volatility: {evaluation['expected_volatility'] * 100:.2f}%")
    print(f"Sharpe ratio: {evaluation['sharpe_ratio']:.2f}")
    print(f"Sortino ratio: {evaluation['sortino_ratio']:.2f}")
    print(f"Maximum drawdown: {evaluation['max_drawdown'] * 100:.2f}%")