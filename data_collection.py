# data_collection/yahoo_finance_data.py

import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger('investment_model.data_collection')


class YahooFinanceDataCollector:
    """Class for collecting financial data from Yahoo Finance."""

    def __init__(self, start_date, end_date,
                 tickers_file='data/sp500_tickers.csv'):
        """Initialize the Yahoo Finance data collector.

        Args:
            start_date (str): Start date for data collection (YYYY-MM-DD).
            end_date (str): End date for data collection (YYYY-MM-DD).
            tickers_file (str): Path to file containing S&P 500 tickers.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers_file = tickers_file
        self._load_tickers()

    def _load_tickers(self):
        """Load S&P 500 tickers from file or download them if file doesn't exist."""
        if os.path.exists(self.tickers_file):
            self.tickers = pd.read_csv(self.tickers_file)['Symbol'].tolist()
            logger.info(
                f"Loaded {len(self.tickers)} tickers from {self.tickers_file}")
        else:
            logger.info(
                "Tickers file not found. Downloading S&P 500 tickers...")
            # Try to get S&P 500 tickers using pandas_datareader
            try:
                import pandas_datareader.data as web
                sp500 = web.DataReader('sp500', 'fred', self.start_date,
                                       self.end_date)
                self.tickers = sp500.columns.tolist()
            except:
                # Fallback to a hardcoded list of major S&P 500 companies if the above fails
                logger.warning(
                    "Failed to download S&P 500 tickers. Using default list of major companies.")
                self.tickers = [
                    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'NVDA',
                    'JPM', 'V', 'PG',
                    'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM', 'CMCSA',
                    'VZ', 'NFLX',
                    'KO', 'PEP', 'T', 'INTC', 'PFE', 'ABT', 'MRK', 'WMT',
                    'CSCO', 'CVX'
                ]

            # Save tickers to file
            os.makedirs(os.path.dirname(self.tickers_file), exist_ok=True)
            pd.DataFrame({'Symbol': self.tickers}).to_csv(self.tickers_file,
                                                          index=False)
            logger.info(
                f"Saved {len(self.tickers)} tickers to {self.tickers_file}")

    def _download_stock_data(self, ticker):
        """Download historical stock data for a single ticker.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            tuple: (ticker, stock_data) or (ticker, None) if download fails.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=self.start_date, end=self.end_date)

            # Check if data is empty
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return ticker, None

            # Add ticker column
            data['ticker'] = ticker

            return ticker, data
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return ticker, None

    def _download_fundamental_data(self, ticker):
        """Download fundamental data for a single ticker.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            tuple: (ticker, fundamental_data) or (ticker, None) if download fails.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Get key statistics
            info = stock.info

            # Create a dictionary with all fundamental data
            fundamental_data = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info
            }

            return ticker, fundamental_data
        except Exception as e:
            logger.error(
                f"Error downloading fundamental data for {ticker}: {str(e)}")
            return ticker, None

    def collect_data(self, max_workers=10):
        """Collect stock price and fundamental data for S&P 500 companies.

        Args:
            max_workers (int): Maximum number of threads for parallel downloading.

        Returns:
            tuple: (stock_data_dict, fundamental_data_dict)
        """
        logger.info(
            f"Collecting data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")

        # Download stock price data
        stock_data = {}
        fundamental_data = {}

        # Use ThreadPoolExecutor for parallel downloading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit stock data download tasks
            stock_futures = {
                executor.submit(self._download_stock_data, ticker): ticker for
                ticker in self.tickers}

            # Process stock data results
            for future in tqdm(as_completed(stock_futures),
                               total=len(stock_futures),
                               desc="Downloading stock data"):
                ticker, data = future.result()
                if data is not None:
                    stock_data[ticker] = data

            # Submit fundamental data download tasks
            fundamental_futures = {
                executor.submit(self._download_fundamental_data,
                                ticker): ticker for ticker in self.tickers}

            # Process fundamental data results
            for future in tqdm(as_completed(fundamental_futures),
                               total=len(fundamental_futures),
                               desc="Downloading fundamental data"):
                ticker, data = future.result()
                if data is not None:
                    fundamental_data[ticker] = data

        logger.info(
            f"Successfully collected stock data for {len(stock_data)} tickers")
        logger.info(
            f"Successfully collected fundamental data for {len(fundamental_data)} tickers")

        # Save raw data
        self._save_raw_data(stock_data, fundamental_data)

        return stock_data, fundamental_data

    def _save_raw_data(self, stock_data, fundamental_data):
        """Save raw data to files.

        Args:
            stock_data (dict): Dictionary of stock price data by ticker.
            fundamental_data (dict): Dictionary of fundamental data by ticker.
        """
        # Create directories if they don't exist
        os.makedirs('data/raw/stock_data', exist_ok=True)
        os.makedirs('data/raw/fundamental_data', exist_ok=True)

        # Save stock data
        for ticker, data in stock_data.items():
            data.to_csv(f'data/raw/stock_data/{ticker}.csv')

        # Save fundamental data (convert to DataFrames where possible)
        for ticker, data in fundamental_data.items():
            # Create directory for each ticker
            os.makedirs(f'data/raw/fundamental_data/{ticker}', exist_ok=True)

            # Save income statement
            if not data['income_statement'].empty:
                data['income_statement'].to_csv(
                    f'data/raw/fundamental_data/{ticker}/income_statement.csv')

            # Save balance sheet
            if not data['balance_sheet'].empty:
                data['balance_sheet'].to_csv(
                    f'data/raw/fundamental_data/{ticker}/balance_sheet.csv')

            # Save cash flow
            if not data['cash_flow'].empty:
                data['cash_flow'].to_csv(
                    f'data/raw/fundamental_data/{ticker}/cash_flow.csv')

            # Save info as JSON
            if data['info']:
                pd.Series(data['info']).to_json(
                    f'data/raw/fundamental_data/{ticker}/info.json')

        logger.info("Raw data saved to files")



logger = logging.getLogger('investment_model.data_collection')


class MarketDataCollector:
    """Class for collecting market-wide data like indices and economic indicators."""

    def __init__(self, start_date, end_date):
        """Initialize the market data collector.

        Args:
            start_date (str): Start date for data collection (YYYY-MM-DD).
            end_date (str): End date for data collection (YYYY-MM-DD).
        """
        self.start_date = start_date
        self.end_date = end_date
        self.market_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
        self.sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI', 'XLP',
                            'XLB', 'XLU', 'XLRE']

    def collect_index_data(self):
        """Collect market index data.

        Returns:
            dict: Dictionary of index data by index symbol.
        """
        logger.info(f"Collecting market index data")
        index_data = {}

        for index in self.market_indices:
            try:
                data = yf.download(index, start=self.start_date,
                                   end=self.end_date)
                if not data.empty:
                    index_data[index] = data
                    logger.info(f"Downloaded data for {index}")
                else:
                    logger.warning(f"No data found for {index}")
            except Exception as e:
                logger.error(f"Error downloading data for {index}: {str(e)}")

        logger.info(
            f"Successfully collected data for {len(index_data)} market indices")
        return index_data

    def collect_sector_data(self):
        """Collect sector ETF data.

        Returns:
            dict: Dictionary of sector ETF data by ETF symbol.
        """
        logger.info(f"Collecting sector ETF data")
        sector_data = {}

        for etf in self.sector_etfs:
            try:
                data = yf.download(etf, start=self.start_date,
                                   end=self.end_date)
                if not data.empty:
                    sector_data[etf] = data
                    logger.info(f"Downloaded data for {etf}")
                else:
                    logger.warning(f"No data found for {etf}")
            except Exception as e:
                logger.error(f"Error downloading data for {etf}: {str(e)}")

        logger.info(
            f"Successfully collected data for {len(sector_data)} sector ETFs")
        return sector_data

    def collect_economic_indicators(self):
        """Collect economic indicator data.

        Returns:
            dict: Dictionary of economic indicator data.
        """
        logger.info("Collecting economic indicator data")

        # This function would typically use APIs like FRED (Federal Reserve Economic Data)
        # Since we're using yfinance, we'll use some ETFs as proxies for economic indicators
        indicators = {
            'treasury_10Y': '^TNX',  # 10-Year Treasury Yield
            'treasury_2Y': '^TYX',  # 30-Year Treasury Yield
            'high_yield': 'HYG',  # High Yield Corporate Bonds
            'investment_grade': 'LQD',  # Investment Grade Corporate Bonds
            'gold': 'GLD',  # Gold
            'dollar_index': 'UUP'  # US Dollar Index
        }

        indicator_data = {}

        for name, symbol in indicators.items():
            try:
                data = yf.download(symbol, start=self.start_date,
                                   end=self.end_date)
                if not data.empty:
                    indicator_data[name] = data
                    logger.info(f"Downloaded data for {name} ({symbol})")
                else:
                    logger.warning(f"No data found for {name} ({symbol})")
            except Exception as e:
                logger.error(
                    f"Error downloading data for {name} ({symbol}): {str(e)}")

        logger.info(
            f"Successfully collected data for {len(indicator_data)} economic indicators")
        return indicator_data

    def collect_all_market_data(self):
        """Collect all market-related data.

        Returns:
            dict: Dictionary containing all market data.
        """
        market_data = {
            'indices': self.collect_index_data(),
            'sectors': self.collect_sector_data(),
            'economic_indicators': self.collect_economic_indicators()
        }

        # Save market data
        self._save_market_data(market_data)

        return market_data

    def _save_market_data(self, market_data):
        """Save market data to files.

        Args:
            market_data (dict): Dictionary containing all market data.
        """
        # Create directories if they don't exist
        for category in ['indices', 'sectors', 'economic_indicators']:
            os.makedirs(f'data/raw/market_data/{category}', exist_ok=True)

            # Save each data item
            for name, data in market_data[category].items():
                file_name = name.replace('^',
                                         '')  # Remove ^ from index names for file names
                data.to_csv(f'data/raw/market_data/{category}/{file_name}.csv')

        logger.info("Market data saved to files")