# Strategy notes

## Candidate to keep: monthly ETF relative-momentum rotation

- At each completed month-end, rank SPY, QQQ, IWM, DIA, sector ETFs, and SMH by trailing 12-month total return.
- Hold the strongest ETF only when its 12-month return is positive and its month-end close is above its 10-month simple moving average. Otherwise hold cash.
- Rebalance at the next trading session open. Use the symbol selected in the app as the buy-and-hold benchmark.

### Screenshot result, 2021-10-01 to 2026-10-28

- Benchmark: QQQ; initial capital: $100,000.
- Strategy final value: $377,879.83 (+277.88%); QQQ buy-and-hold: $205,681.11 (+105.68%).
- Maximum strategy drawdown: -31.95%; time invested: 98.3%; 20 fill events.
- IEX feed, 0% cash yield, $0 commission, 5 bps adverse slippage per fill.

Treat this as a promising backtest candidate, not a confirmed edge. The recorded result is one historical window using IEX data and a fixed ETF universe; compare additional periods and feeds before relying on it.
