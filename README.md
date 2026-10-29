# Historical Strategy Backtester

Independent Streamlit app for retrospective comparisons of multiple trading strategies and symbols. The app requests historical Alpaca bars and corporate actions only; it does not submit orders.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and fill in the Alpaca API key and secret for account `accn_0`. Keep `.env` private; it is ignored by Git.
4. Start the app with `streamlit run macd_backtest_app.py`.

The app includes intraday MACD, daily SMA, RSI, Bollinger Band mean reversion, a monthly 10-month SMA risk filter, monthly ETF relative-momentum rotation, SuperTrend, monthly 12-month absolute momentum, and an accumulation-breakout proxy. The relative-momentum policy rotates among index and sector ETFs based on trailing 12-month total return, with a 10-month moving-average trend filter; the selected symbol is its buy-and-hold benchmark. The Bollinger policy defaults to a 20-session average with 2 standard deviations, buying below the lower band and exiting at the middle band. The monthly filter changes between QQQ and cash only after a completed month closes across the 10-month average. The accumulation proxy looks for a tight 60-session base, positive 20-day Chaikin Money Flow, then a breakout on elevated volume. It does not identify the buyer. For broader volume coverage, select the SIP feed when available; IEX covers one exchange.

Run `python momentum_strategy_research.py` to compare monthly 12-month absolute momentum with buy-and-hold over fixed QQQ windows. It tests 0% and illustrative constant 4% annual cash yields, plus both configured transaction-cost scenarios, and saves `momentum_strategy_results.csv`. Data are split-adjusted; the study uses SIP through 2025-09-30 and IEX thereafter. The 4% rate is a sensitivity assumption, not historical cash yield. Run `python rsi_strategy_research.py` for the saved RSI comparison.

Run `python accumulation_strategy_research.py` for the same fixed windows and cost/yield scenarios using the accumulation-breakout defaults. Recent IEX single-exchange volume makes the latest reference-year result less conclusive.

Run checks with `python -m unittest tests.test_accumulation tests.test_monthly_momentum tests.test_macd_backtest_rsi tests.test_macd_backtest_supertrend`.
