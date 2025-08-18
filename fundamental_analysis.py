# analysis/fundamental_analysis.py

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import os

logger = logging.getLogger('investment_model.analysis')


class FundamentalAnalyzer:
    """Class for analyzing fundamental financial data of companies."""

    def __init__(self):
        """Initialize the fundamental analyzer."""
        self.fundamental_metrics = {
            'valuation': [
                'P/E', 'Forward_P/E', 'PEG', 'P/S', 'P/B', 'P/FCF',
                'EV/EBITDA', 'EV/Revenue',
                'CAPE', 'Earnings_Yield', 'Dividend_Yield', 'FCF_Yield',
                'Buyback_Yield'
            ],
            'growth': [
                'Revenue_Growth_1Y', 'Revenue_Growth_3Y', 'Revenue_Growth_5Y',
                'EPS_Growth_1Y', 'EPS_Growth_3Y', 'EPS_Growth_5Y',
                'FCF_Growth_1Y', 'FCF_Growth_3Y', 'FCF_Growth_5Y',
                'Dividend_Growth_5Y', 'EBITDA_Growth_5Y',
                'Book_Value_Growth_5Y'
            ],
            'profitability': [
                'Gross_Margin', 'Operating_Margin', 'Net_Margin', 'FCF_Margin',
                'ROE', 'ROA', 'ROIC', 'ROCE', 'ROI', 'ROC',
                'Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover'
            ],
            'financial_health': [
                'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity',
                'Debt_to_Assets',
                'Debt_to_EBITDA', 'Interest_Coverage', 'EBITDA_to_Interest',
                'Cash_to_Debt', 'Cash_to_Assets', 'Altman_Z_Score',
                'Piotroski_F_Score'
            ],
            'efficiency': [
                'Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover',
                'Cash_Conversion_Cycle', 'Days_Sales_Outstanding',
                'Days_Inventory_Outstanding', 'Days_Payables_Outstanding',
                'Operating_Cycle', 'Fixed_Asset_Turnover',
                'Total_Asset_Turnover'
            ],
            'cash_flow': [
                'FCF_to_Sales', 'FCF_to_Net_Income', 'FCF_to_Operating_CF',
                'FCF_to_CapEx', 'FCF_Growth_Rate', 'CapEx_to_Sales',
                'CapEx_to_Depreciation', 'Cash_Flow_Coverage', 'EBITDA_to_OCF'
            ],
            'quality': [
                'Accrual_Ratio', 'Beneish_M_Score', 'Quality_of_Earnings',
                'Sloan_Ratio', 'Gross_Profits_to_Assets',
                'Earnings_Persistence',
                'Net_Operating_Assets'
            ],
            'dividend': [
                'Dividend_Yield', 'Dividend_Payout_Ratio', 'Dividend_Coverage',
                'Dividend_Growth_Rate', 'Consecutive_Dividend_Years',
                'Dividend_Stability'
            ],
            'management': [
                'CEO_Tenure', 'Insider_Ownership',
                'Executive_Compensation_Ratio',
                'Share_Buyback_Ratio', 'Inside_Buys_Sells_Ratio', 'ROE_Trend'
            ]
        }

    def analyze(self, fundamental_data_dict):
        """Analyze fundamental data for all companies.

        Args:
            fundamental_data_dict (dict): Dictionary of fundamental data by ticker.

        Returns:
            dict: Dictionary of fundamental analysis results by ticker.
        """
        logger.info("Analyzing fundamental data")

        analysis_results = {}

        for ticker, data in tqdm(fundamental_data_dict.items(),
                                 desc="Fundamental analysis"):
            try:
                # Extract data from different financial statements
                income_stmt = data.get('income_statement', pd.DataFrame())
                balance_sheet = data.get('balance_sheet', pd.DataFrame())
                cash_flow = data.get('cash_flow', pd.DataFrame())
                info = data.get('info', {})

                # Skip if essential data is missing
                if income_stmt.empty or balance_sheet.empty:
                    logger.warning(
                        f"Missing essential financial data for {ticker}, skipping fundamental analysis")
                    continue

                # Calculate fundamental metrics
                metrics = self._calculate_fundamental_metrics(income_stmt,
                                                              balance_sheet,
                                                              cash_flow, info)

                # Calculate fundamental scores
                scores = self._calculate_fundamental_scores(metrics)

                # Store analysis results
                analysis_results[ticker] = {
                    'metrics': metrics,
                    'scores': scores
                }

            except Exception as e:
                logger.error(
                    f"Error analyzing fundamental data for {ticker}: {str(e)}")

        # Save analysis results
        self._save_analysis_results(analysis_results)

        # Return scores for all tickers for easier integration with other components
        fundamental_scores = {ticker: results['scores'] for ticker, results in
                              analysis_results.items()}

        logger.info(
            f"Successfully analyzed fundamental data for {len(analysis_results)} tickers")
        return fundamental_scores

    def _calculate_fundamental_metrics(self, income_stmt, balance_sheet,
                                       cash_flow, info):
        """Calculate fundamental metrics from financial statements.

        Args:
            income_stmt (pd.DataFrame): Income statement data.
            balance_sheet (pd.DataFrame): Balance sheet data.
            cash_flow (pd.DataFrame): Cash flow statement data.
            info (dict): General company information.

        Returns:
            dict: Dictionary of calculated fundamental metrics.
        """
        metrics = {}

        # Extract latest values (most recent fiscal year)
        if not income_stmt.empty:
            latest_income = income_stmt.iloc[:, 0]  # Most recent column
        else:
            latest_income = pd.Series()

        if not balance_sheet.empty:
            latest_balance = balance_sheet.iloc[:, 0]  # Most recent column
        else:
            latest_balance = pd.Series()

        if not cash_flow.empty:
            latest_cash_flow = cash_flow.iloc[:, 0]  # Most recent column
        else:
            latest_cash_flow = pd.Series()

        # Helper function to safely extract a value
        def safe_get(df, key, default=np.nan):
            if key in df:
                return df[key]
            return default

        # Helper function to calculate growth rate
        def calculate_growth_rate(df, key, periods=1):
            if key not in df.index or df.shape[1] <= periods:
                return np.nan

            current = df.loc[key, df.columns[0]]
            previous = df.loc[key, df.columns[periods]]

            if previous == 0 or np.isnan(previous) or np.isnan(current):
                return np.nan

            return (current / previous) - 1

        # Valuation metrics
        market_cap = info.get('marketCap', np.nan)
        enterprise_value = info.get('enterpriseValue', market_cap)

        # Basic financial data
        metrics['Revenue'] = safe_get(latest_income, 'Total Revenue')
        metrics['Net_Income'] = safe_get(latest_income, 'Net Income')
        metrics['EBITDA'] = safe_get(latest_income, 'EBITDA')
        metrics['Gross_Profit'] = safe_get(latest_income, 'Gross Profit')
        metrics['Operating_Income'] = safe_get(latest_income,
                                               'Operating Income')

        metrics['Total_Assets'] = safe_get(latest_balance, 'Total Assets')
        metrics['Total_Liabilities'] = safe_get(latest_balance,
                                                'Total Liabilities Net Minority Interest')
        metrics['Total_Equity'] = safe_get(latest_balance,
                                           'Total Equity Gross Minority Interest')
        metrics['Total_Debt'] = safe_get(latest_balance, 'Total Debt')
        metrics['Cash_And_Equivalents'] = safe_get(latest_balance,
                                                   'Cash And Cash Equivalents')

        metrics['Operating_Cash_Flow'] = safe_get(latest_cash_flow,
                                                  'Operating Cash Flow')
        metrics['Free_Cash_Flow'] = safe_get(latest_cash_flow,
                                             'Free Cash Flow')
        metrics['Capital_Expenditure'] = safe_get(latest_cash_flow,
                                                  'Capital Expenditure')

        # P/E ratio
        metrics['P/E'] = market_cap / metrics['Net_Income'] if not np.isnan(
            metrics['Net_Income']) and metrics['Net_Income'] > 0 else np.nan

        # Forward P/E
        metrics['Forward_P/E'] = info.get('forwardPE', np.nan)

        # P/S ratio
        metrics['P/S'] = market_cap / metrics['Revenue'] if not np.isnan(
            metrics['Revenue']) and metrics['Revenue'] > 0 else np.nan

        # P/B ratio
        metrics['P/B'] = market_cap / metrics['Total_Equity'] if not np.isnan(
            metrics['Total_Equity']) and metrics[
                                                                     'Total_Equity'] > 0 else np.nan

        # P/FCF ratio
        metrics['P/FCF'] = market_cap / metrics[
            'Free_Cash_Flow'] if not np.isnan(metrics['Free_Cash_Flow']) and \
                                 metrics['Free_Cash_Flow'] > 0 else np.nan

        # EV/EBITDA
        metrics['EV/EBITDA'] = enterprise_value / metrics[
            'EBITDA'] if not np.isnan(metrics['EBITDA']) and metrics[
            'EBITDA'] > 0 else np.nan

        # EV/Revenue
        metrics['EV/Revenue'] = enterprise_value / metrics[
            'Revenue'] if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # PEG ratio
        eps_growth_5y = info.get('pegRatio', np.nan)
        metrics['PEG'] = metrics['P/E'] / eps_growth_5y if not np.isnan(
            eps_growth_5y) and eps_growth_5y > 0 else np.nan

        # Earnings Yield
        metrics['Earnings_Yield'] = (metrics[
                                         'Net_Income'] / market_cap) * 100 if not np.isnan(
            market_cap) and market_cap > 0 else np.nan

        # Dividend Yield
        metrics['Dividend_Yield'] = info.get('dividendYield', 0) * 100

        # FCF Yield
        metrics['FCF_Yield'] = (metrics[
                                    'Free_Cash_Flow'] / market_cap) * 100 if not np.isnan(
            market_cap) and market_cap > 0 else np.nan

        # Profitability metrics
        # Gross Margin
        metrics['Gross_Margin'] = (metrics['Gross_Profit'] / metrics[
            'Revenue']) * 100 if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # Operating Margin
        metrics['Operating_Margin'] = (metrics['Operating_Income'] / metrics[
            'Revenue']) * 100 if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # Net Margin
        metrics['Net_Margin'] = (metrics['Net_Income'] / metrics[
            'Revenue']) * 100 if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # FCF Margin
        metrics['FCF_Margin'] = (metrics['Free_Cash_Flow'] / metrics[
            'Revenue']) * 100 if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # ROE
        metrics['ROE'] = (metrics['Net_Income'] / metrics[
            'Total_Equity']) * 100 if not np.isnan(metrics['Total_Equity']) and \
                                      metrics['Total_Equity'] > 0 else np.nan

        # ROA
        metrics['ROA'] = (metrics['Net_Income'] / metrics[
            'Total_Assets']) * 100 if not np.isnan(metrics['Total_Assets']) and \
                                      metrics['Total_Assets'] > 0 else np.nan

        # ROIC (simplified)
        net_operating_profit = metrics['Operating_Income'] * (
                    1 - 0.21)  # Assuming 21% tax rate
        invested_capital = metrics['Total_Equity'] + metrics['Total_Debt'] - \
                           metrics['Cash_And_Equivalents']
        metrics['ROIC'] = (
                                      net_operating_profit / invested_capital) * 100 if not np.isnan(
            invested_capital) and invested_capital > 0 else np.nan

        # Growth metrics
        # Revenue Growth 1Y
        metrics['Revenue_Growth_1Y'] = calculate_growth_rate(
            income_stmt.loc[['Total Revenue']], 'Total Revenue', 1) * 100

        # Revenue Growth 3Y
        if income_stmt.shape[1] > 3:
            current_rev = safe_get(latest_income, 'Total Revenue')
            past_rev = safe_get(income_stmt.iloc[:, 3], 'Total Revenue')
            metrics['Revenue_Growth_3Y'] = ((current_rev / past_rev) ** (
                        1 / 3) - 1) * 100 if not np.isnan(
                current_rev) and not np.isnan(
                past_rev) and past_rev > 0 else np.nan
        else:
            metrics['Revenue_Growth_3Y'] = np.nan

        # EPS Growth 1Y
        metrics['EPS_Growth_1Y'] = calculate_growth_rate(
            income_stmt.loc[['Diluted EPS']], 'Diluted EPS', 1) * 100

        # FCF Growth 1Y
        metrics['FCF_Growth_1Y'] = calculate_growth_rate(
            cash_flow.loc[['Free Cash Flow']], 'Free Cash Flow', 1) * 100

        # Financial health metrics
        # Current Ratio
        current_assets = safe_get(latest_balance, 'Current Assets')
        current_liabilities = safe_get(latest_balance, 'Current Liabilities')
        metrics[
            'Current_Ratio'] = current_assets / current_liabilities if not np.isnan(
            current_liabilities) and current_liabilities > 0 else np.nan

        # Quick Ratio
        inventory = safe_get(latest_balance, 'Inventory')
        metrics['Quick_Ratio'] = (
                                             current_assets - inventory) / current_liabilities if not np.isnan(
            current_liabilities) and current_liabilities > 0 else np.nan

        # Debt to Equity
        metrics['Debt_to_Equity'] = metrics['Total_Debt'] / metrics[
            'Total_Equity'] if not np.isnan(metrics['Total_Equity']) and \
                               metrics['Total_Equity'] > 0 else np.nan

        # Debt to Assets
        metrics['Debt_to_Assets'] = metrics['Total_Debt'] / metrics[
            'Total_Assets'] if not np.isnan(metrics['Total_Assets']) and \
                               metrics['Total_Assets'] > 0 else np.nan

        # Interest Coverage Ratio
        interest_expense = safe_get(latest_income, 'Interest Expense', 0)
        metrics['Interest_Coverage'] = metrics['Operating_Income'] / abs(
            interest_expense) if abs(interest_expense) > 0 else np.nan

        # Debt to EBITDA
        metrics['Debt_to_EBITDA'] = metrics['Total_Debt'] / metrics[
            'EBITDA'] if not np.isnan(metrics['EBITDA']) and metrics[
            'EBITDA'] > 0 else np.nan

        # Altman Z-Score (simplified version)
        working_capital = current_assets - current_liabilities
        retained_earnings = safe_get(latest_balance, 'Retained Earnings')
        market_value_equity = market_cap
        sales = metrics['Revenue']

        if not np.isnan(metrics['Total_Assets']) and metrics[
            'Total_Assets'] > 0:
            z_score = (
                    1.2 * (working_capital / metrics['Total_Assets']) +
                    1.4 * (retained_earnings / metrics['Total_Assets']) +
                    3.3 * (metrics['Operating_Income'] / metrics[
                'Total_Assets']) +
                    0.6 * (market_value_equity / metrics[
                'Total_Liabilities']) +
                    0.999 * (sales / metrics['Total_Assets'])
            )
            metrics['Altman_Z_Score'] = z_score
        else:
            metrics['Altman_Z_Score'] = np.nan

        # Efficiency metrics
        # Asset Turnover
        metrics['Asset_Turnover'] = metrics['Revenue'] / metrics[
            'Total_Assets'] if not np.isnan(metrics['Total_Assets']) and \
                               metrics['Total_Assets'] > 0 else np.nan

        # Cash Flow metrics
        # FCF to Sales
        metrics['FCF_to_Sales'] = metrics['Free_Cash_Flow'] / metrics[
            'Revenue'] if not np.isnan(metrics['Revenue']) and metrics[
            'Revenue'] > 0 else np.nan

        # FCF to Net Income
        metrics['FCF_to_Net_Income'] = metrics['Free_Cash_Flow'] / metrics[
            'Net_Income'] if not np.isnan(metrics['Net_Income']) and abs(
            metrics['Net_Income']) > 0 else np.nan

        # CapEx to Sales
        metrics['CapEx_to_Sales'] = abs(metrics['Capital_Expenditure']) / \
                                    metrics['Revenue'] if not np.isnan(
            metrics['Revenue']) and metrics['Revenue'] > 0 else np.nan

        # Piotroski F-Score (simplified)
        f_score = 0

        # Profitability criteria
        if metrics['Net_Income'] > 0:
            f_score += 1
        if metrics['Operating_Cash_Flow'] > 0:
            f_score += 1
        if metrics['ROA'] > 0:
            f_score += 1
        if metrics['Operating_Cash_Flow'] > metrics['Net_Income']:
            f_score += 1

        # Leverage, Liquidity, and Source of Funds criteria
        if metrics['Debt_to_Assets'] < 0.4:  # Arbitrary threshold
            f_score += 1
        if metrics['Current_Ratio'] > 1:
            f_score += 1

        # Operating Efficiency criteria
        if metrics['Gross_Margin'] > 30:  # Arbitrary threshold
            f_score += 1
        if metrics['Asset_Turnover'] > 0.5:  # Arbitrary threshold
            f_score += 1

        metrics['Piotroski_F_Score'] = f_score

        return metrics

    def _calculate_fundamental_scores(self, metrics):
        """Calculate fundamental scores based on metrics.

        Args:
            metrics (dict): Dictionary of fundamental metrics.

        Returns:
            dict: Dictionary of fundamental scores.
        """
        scores = {}

        # Define scoring functions for different metric types
        def score_lower_is_better(value, thresholds):
            if np.isnan(value):
                return 50  # Neutral score for missing data
            for threshold, score in thresholds:
                if value <= threshold:
                    return score
            return 0  # Worst score if above all thresholds

        def score_higher_is_better(value, thresholds):
            if np.isnan(value):
                return 50  # Neutral score for missing data
            for threshold, score in thresholds:
                if value >= threshold:
                    return score
            return 0  # Worst score if below all thresholds

        # Value score - based on valuation metrics (lower is better for most)
        value_metrics = {
            'P/E': score_lower_is_better(metrics.get('P/E', np.nan),
                                         [(5, 100), (10, 80), (15, 60),
                                          (20, 40), (25, 20)]),
            'Forward_P/E': score_lower_is_better(
                metrics.get('Forward_P/E', np.nan),
                [(5, 100), (10, 80), (15, 60), (20, 40), (25, 20)]),
            'P/S': score_lower_is_better(metrics.get('P/S', np.nan),
                                         [(1, 100), (2, 80), (3, 60), (5, 40),
                                          (8, 20)]),
            'P/B': score_lower_is_better(metrics.get('P/B', np.nan),
                                         [(1, 100), (1.5, 80), (2, 60),
                                          (3, 40), (4, 20)]),
            'P/FCF': score_lower_is_better(metrics.get('P/FCF', np.nan),
                                           [(5, 100), (10, 80), (15, 60),
                                            (20, 40), (25, 20)]),
            'EV/EBITDA': score_lower_is_better(
                metrics.get('EV/EBITDA', np.nan),
                [(3, 100), (6, 80), (9, 60), (12, 40), (15, 20)]),
            'PEG': score_lower_is_better(metrics.get('PEG', np.nan),
                                         [(0.5, 100), (0.75, 80), (1, 60),
                                          (1.5, 40), (2, 20)]),
            'Earnings_Yield': score_higher_is_better(
                metrics.get('Earnings_Yield', np.nan),
                [(15, 100), (10, 80), (7, 60), (5, 40), (3, 20)]),
            'Dividend_Yield': score_higher_is_better(
                metrics.get('Dividend_Yield', np.nan),
                [(5, 100), (3, 80), (2, 60), (1, 40), (0.5, 20)])
        }

        scores['value_score'] = sum(value_metrics.values()) / len(
            value_metrics)

        # Growth score - based on growth metrics (higher is better)
        growth_metrics = {
            'Revenue_Growth_1Y': score_higher_is_better(
                metrics.get('Revenue_Growth_1Y', np.nan),
                [(25, 100), (15, 80), (10, 60), (5, 40), (0, 20)]),
            'Revenue_Growth_3Y': score_higher_is_better(
                metrics.get('Revenue_Growth_3Y', np.nan),
                [(20, 100), (12, 80), (8, 60), (4, 40), (0, 20)]),
            'EPS_Growth_1Y': score_higher_is_better(
                metrics.get('EPS_Growth_1Y', np.nan),
                [(30, 100), (20, 80), (10, 60), (5, 40), (0, 20)]),
            'FCF_Growth_1Y': score_higher_is_better(
                metrics.get('FCF_Growth_1Y', np.nan),
                [(30, 100), (20, 80), (10, 60), (5, 40), (0, 20)])
        }

        scores['growth_score'] = sum(growth_metrics.values()) / len(
            growth_metrics)

        # Quality score - based on profitability and financial health
        quality_metrics = {
            'Gross_Margin': score_higher_is_better(
                metrics.get('Gross_Margin', np.nan),
                [(50, 100), (40, 80), (30, 60), (20, 40), (10, 20)]),
            'Operating_Margin': score_higher_is_better(
                metrics.get('Operating_Margin', np.nan),
                [(25, 100), (15, 80), (10, 60), (5, 40), (0, 20)]),
            'Net_Margin': score_higher_is_better(
                metrics.get('Net_Margin', np.nan),
                [(20, 100), (15, 80), (10, 60), (5, 40), (0, 20)]),
            'ROE': score_higher_is_better(metrics.get('ROE', np.nan),
                                          [(25, 100), (20, 80), (15, 60),
                                           (10, 40), (5, 20)]),
            'ROA': score_higher_is_better(metrics.get('ROA', np.nan),
                                          [(15, 100), (10, 80), (7, 60),
                                           (4, 40), (2, 20)]),
            'ROIC': score_higher_is_better(metrics.get('ROIC', np.nan),
                                           [(20, 100), (15, 80), (10, 60),
                                            (7, 40), (4, 20)]),
            'Debt_to_Equity': score_lower_is_better(
                metrics.get('Debt_to_Equity', np.nan),
                [(0.2, 100), (0.5, 80), (1, 60), (1.5, 40), (2, 20)]),
            'Interest_Coverage': score_higher_is_better(
                metrics.get('Interest_Coverage', np.nan),
                [(10, 100), (7, 80), (5, 60), (3, 40), (1, 20)]),
            'Altman_Z_Score': score_higher_is_better(
                metrics.get('Altman_Z_Score', np.nan),
                [(3, 100), (2.5, 80), (2, 60), (1.5, 40), (1, 20)]),
            'Piotroski_F_Score': score_higher_is_better(
                metrics.get('Piotroski_F_Score', np.nan),
                [(8, 100), (6, 80), (5, 60), (4, 40), (3, 20)])
        }

        scores['quality_score'] = sum(quality_metrics.values()) / len(
            quality_metrics)

        # Momentum score - based on growth rates
        momentum_metrics = {
            'Revenue_Growth_1Y': growth_metrics['Revenue_Growth_1Y'],
            'EPS_Growth_1Y': growth_metrics['EPS_Growth_1Y'],
            'FCF_Growth_1Y': growth_metrics['FCF_Growth_1Y']
        }

        scores['momentum_score'] = sum(momentum_metrics.values()) / len(
            momentum_metrics)

        # Overall score - weighted average of individual scores
        scores['overall_score'] = (
                scores['value_score'] * 0.3 +
                scores['growth_score'] * 0.3 +
                scores['quality_score'] * 0.3 +
                scores['momentum_score'] * 0.1
        )

        return scores

    def _save_analysis_results(self, analysis_results):
        """Save fundamental analysis results to files.

        Args:
            analysis_results (dict): Dictionary of analysis results by ticker.
        """
        # Create directory if it doesn't exist
        os.makedirs('results/fundamental', exist_ok=True)

        # Save metrics for each ticker
        metrics_df = pd.DataFrame()
        scores_df = pd.DataFrame()

        for ticker, results in analysis_results.items():
            # Add metrics
            ticker_metrics = pd.Series(results['metrics'], name=ticker)
            metrics_df = pd.concat([metrics_df, ticker_metrics.to_frame().T])

            # Add scores
            ticker_scores = pd.Series(results['scores'], name=ticker)
            scores_df = pd.concat([scores_df, ticker_scores.to_frame().T])

        # Save to CSV
        metrics_df.to_csv('results/fundamental/metrics.csv')
        scores_df.to_csv('results/fundamental/scores.csv')

        logger.info("Fundamental analysis results saved to files")