# Historical Strategy Backtester

Independent Streamlit app for retrospective comparisons of multiple trading strategies and symbols. The app requests historical Alpaca bars and corporate actions only; it does not submit orders.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and fill in the Alpaca API key and secret for account `accn_0`. Keep `.env` private; it is ignored by Git.
4. Start the app with `streamlit run macd_backtest_app.py`.

The default policy is MACD. Choose **Daily RSI recovery (buy 30 / sell 70)** for the RSI policy. It uses completed daily bars and fills at the next session open.

Run the saved QQQ study with `python rsi_strategy_research.py`. It compares RSI(14), 30/70 with buy-and-hold in fixed historical windows and saves `rsi_strategy_results.csv`. Data are split-adjusted; the study uses SIP through 2025-09-30 and IEX thereafter.

Run checks with `python -m unittest tests.test_macd_backtest_rsi tests.test_macd_backtest_supertrend`.
