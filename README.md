# CompassAllocation

Long‑Term S&P 500 Investment Model

This project is a complete framework for constructing a long‑only U.S. equity portfolio. It combines fundamental and technical analysis with machine‑learning forecasts and modern portfolio optimization to allocate across S&P 500 stocks.

Flexible data sources: Download price and fundamental data directly from Yahoo Finance or load your pre‑collected dataset from a wide‑format CSV/ZIP file.

Signal generation: Compute value, growth, quality, and momentum factors alongside technical indicators such as RSI, MACD, and trend strength. Four predictive models—an LSTM‑inspired classifier, ARIMA forecaster, HMM/GMM regime detector, and a GARCH/EWMA volatility estimator—provide directional probabilities and risk forecasts.

Portfolio construction: A mean–variance optimizer with shrinkage covariance, cash allocation and position limits builds a diversified portfolio. A lightweight risk overlay scales positions based on regime and volatility forecasts.

Evaluation & visualisation: End‑to‑end backtesting, performance metrics (annualised return, volatility, Sharpe, Sortino, max drawdown, VaR/CVaR) and rich plots (equity curve, drawdown, allocation and risk contributions) help you understand and tune your strategy.

The code is modular and configurable via a YAML file, so you can experiment with different universes, model parameters, scoring weights, and risk constraints without rewriting core logic. Whether you’re exploring systematic investing or building a personal portfolio, this repository provides a solid starting point.

To the reader, this is a project made to assess the effects and merit of using technical indicators for a buy-hold portfolio. Change the code accordingly for your trading strategy.  
