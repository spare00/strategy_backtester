"""Read-only QQQ 12-month momentum versus buy-and-hold across fixed windows."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from macd_backtest_core import NY, dividend_events, fetch_corporate_actions, fetch_daily_bars, run_buy_hold, run_daily_policy

CAPITAL = 100_000.0
END = date(2026, 10, 27)  # Exclusive; reference year ends 2026-10-26.
CUTOVER = date(2025, 10, 1)
WINDOWS = {
    "2017-2019": (date(2017, 1, 1), date(2020, 1, 1)),
    "2020-2021": (date(2020, 1, 1), date(2022, 1, 1)),
    "2022-2026-10-26": (date(2022, 1, 1), END),
    "Reference year 2025-10-27 to 2026-10-26": (date(2025, 10, 27), END),
}
COSTS = {"5 bps, $0 commission": (0.0, 5.0), "10 bps + $0.005/share/fill": (0.005, 10.0)}
CASH_YIELDS = (0.0, 4.0)  # Constant-rate sensitivity; 4% is illustrative, not historical.
POLICY = "12-month absolute momentum"


def evaluate(frame: pd.DataFrame, actions: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for period, (period_start, period_end) in WINDOWS.items():
        start = pd.Timestamp(datetime.combine(period_start, time.min, NY))
        end = pd.Timestamp(datetime.combine(period_end, time.min, NY))
        selected = frame[(frame.index >= start - pd.Timedelta(days=400)) & (frame.index < end)]
        traded = selected[selected.index >= start]
        if traded.empty:
            continue
        events = dividend_events(actions, period_start, period_end - timedelta(days=1))
        momentum_dividends = dividend_events(actions, period_start - timedelta(days=400),
                                              period_end - timedelta(days=1))
        for scenario, (fee, slip) in COSTS.items():
            for cash_yield in CASH_YIELDS:
                benchmark, bh_ledger = run_buy_hold(traded, CAPITAL, fee, slip, events, cash_yield)
                result, trades, ledger = run_daily_policy(
                    selected, CAPITAL, POLICY, fee, slip, start, events, cash_yield,
                    momentum_dividends)
                for name, data, equity, fills, exposure, distributions in (
                    (POLICY, result, result.equity, trades, result.invested.mean(),
                     sum(item["Distribution"] for item in ledger)),
                    ("Buy and hold", traded.assign(invested=True), benchmark,
                     [{"Side": "Buy"}, {"Side": "Sell"}], 1.0,
                     sum(item["Distribution"] for item in bh_ledger)),
                ):
                    drawdown = min(float((equity / equity.cummax() - 1).min()),
                                   float(equity.iloc[0] / CAPITAL - 1)) * 100
                    final = float(equity.iloc[-1])
                    rows.append({"Period": period, "Strategy": name, "Costs": scenario,
                                 "Cash yield APR (%)": cash_yield,
                                 "Actual start": str(equity.index[0].date()),
                                 "Actual end": str(equity.index[-1].date()),
                                 "Final value ($)": round(final, 2),
                                 "Return (%)": round((final / CAPITAL - 1) * 100, 2),
                                 "Max drawdown (%)": round(drawdown, 2),
                                 "Exposure (%)": round(float(exposure) * 100, 2),
                                 "Fill events": len(fills),
                                 "Dividend cash/receivable ($)": round(distributions, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    start = datetime.combine(date(2015, 1, 1), time.min, NY).astimezone(timezone.utc)
    cutover = datetime.combine(CUTOVER, time.min, NY).astimezone(timezone.utc)
    end = datetime.combine(END, time.min, NY).astimezone(timezone.utc)
    sip = fetch_daily_bars("QQQ", start, cutover - timedelta(microseconds=1), "sip")
    iex = fetch_daily_bars("QQQ", cutover, end, "iex")
    frame = pd.concat([sip, iex]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise SystemExit("Alpaca returned no QQQ daily bars.")
    actions = fetch_corporate_actions("QQQ", date(2015, 1, 1), END - timedelta(days=1))
    table = evaluate(frame, actions)
    output = Path(__file__).with_name("momentum_strategy_results.csv")
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nActual bars: {len(frame):,}; {frame.index.min().date()} to {frame.index.max().date()}; "
          f"SIP {sip.index.min().date()}–{sip.index.max().date()} ({len(sip):,}), "
          f"IEX {iex.index.min().date()}–{iex.index.max().date()} ({len(iex):,}); adjustment=split.")
    print(f"Cash-dividend records retrieved: {len(actions.get('cash_dividends', []))}; saved {output}")


if __name__ == "__main__":
    main()
