"""Read-only Alpaca data access and historical backtest calculations."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

NY = ZoneInfo("America/New_York")


def account_zero_credentials() -> tuple[str, str]:
    """Read only the first Alpaca key pair, labelled accn_0, from this project's .env."""
    active_account = False
    values: dict[str, str] = {}
    for line in Path(__file__).with_name(".env").read_text().splitlines():
        label = line.strip().lower()
        if label.startswith("# alpaca accn_"):
            active_account = label.partition("(")[0].strip() == "# alpaca accn_0"
        elif active_account and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() in {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}:
                values[key.strip()] = value.strip().strip("\"'")
    key, secret = values.get("ALPACA_API_KEY"), values.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca accn_0 credentials are missing from this project's .env.")
    return key, secret


def fetch_bars(symbol: str, interval: int, start: datetime, end: datetime, feed: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(*account_zero_credentials())
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(interval, TimeFrameUnit.Minute),
                               start=start, end=end, feed=feed, adjustment=Adjustment.SPLIT)
    bars = client.get_stock_bars(request).data.get(symbol, [])
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame([{"time": bar.timestamp.astimezone(NY), "open": float(bar.open),
                           "high": float(bar.high), "low": float(bar.low), "close": float(bar.close),
                           "volume": int(bar.volume)} for bar in bars]).set_index("time").sort_index()
    return frame[(frame.index.weekday < 5) & (frame.index.time >= time(9, 30)) & (frame.index.time < time(16, 0))]


def fetch_daily_bars(symbol: str, start: datetime, end: datetime, feed: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(*account_zero_credentials())
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=start, end=end, feed=feed, adjustment=Adjustment.SPLIT)
    bars = client.get_stock_bars(request).data.get(symbol, [])
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame([{"date": bar.timestamp.date(), "open": float(bar.open),
                           "high": float(bar.high), "low": float(bar.low), "close": float(bar.close),
                           "volume": int(bar.volume)} for bar in bars]).set_index("date").sort_index()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(NY)
    return frame


def fetch_corporate_actions(symbol: str, start: date, end: date) -> dict[str, list[dict]]:
    key, secret = account_zero_credentials()
    params = {"symbols": symbol, "types": "cash_dividend,forward_split,reverse_split,unit_split",
              "start": start.isoformat(), "end": end.isoformat(), "limit": 1000}
    actions: dict[str, list[dict]] = {}
    while True:
        request = Request("https://data.alpaca.markets/v1/corporate-actions?" + urlencode(params),
                          headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        with urlopen(request, timeout=30) as response:
            page = json.load(response)
        for kind, items in page.get("corporate_actions", {}).items():
            actions.setdefault(kind, []).extend(items)
        if not page.get("next_page_token"):
            return actions
        params["page_token"] = page["next_page_token"]


def dividend_events(actions: dict[str, list[dict]], start: date, end: date) -> list[dict]:
    splits = [item for key in ("forward_splits", "reverse_splits", "unit_splits")
              for item in actions.get(key, [])]
    events = []
    for item in actions.get("cash_dividends", []):
        ex_date = date.fromisoformat(item["ex_date"])
        if start <= ex_date <= end:
            later_factor = 1.0
            for split in splits:
                if date.fromisoformat(split["ex_date"]) > ex_date:
                    later_factor *= float(split["new_rate"]) / float(split["old_rate"])
            events.append({"ex_date": ex_date,
                           "pay_date": date.fromisoformat(item["payable_date"]) if item.get("payable_date") else None,
                           "rate": float(item["rate"]) / later_factor, "shares": None,
                           "amount": 0.0, "paid": False})
    return sorted(events, key=lambda event: event["ex_date"])


def ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def run_policy(frame: pd.DataFrame, initial_cash: float, policy: str, overnight: bool, fee_per_share: float,
               slippage_bps: float, range_start: pd.Timestamp | None = None,
               distributions: list[dict] | None = None) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Run one of the two fixed long-only rules, filling signals at the next bar open."""
    history = frame.copy()
    history["macd"] = ema(history.close, 12) - ema(history.close, 26)
    history["signal"] = ema(history.macd, 9)
    history["hist"] = history.macd - history.signal
    # A daily regime is known only after its session closes; expose it from the next session onward.
    daily_close = history.close.groupby(history.index.date).last()
    daily_ema = ema(daily_close, 200)
    history["regime_up"] = pd.Series(history.index.date, index=history.index).map((daily_close > daily_ema).shift(1)).fillna(False).astype(bool)
    history = history.dropna(subset=["macd", "signal"])
    start_i = int(history.index.searchsorted(range_start)) if range_start is not None else 0
    data = history.iloc[start_i:].copy()
    if data.empty:
        return data, [], []
    events = [dict(event) for event in (distributions or [])]
    cash, dividend_cash, shares, entry = initial_cash, 0.0, 0, None
    prior_close_shares = 0
    trades, equity = [], []
    for i in range(start_i, len(history)):
        row, stamp = history.iloc[i], history.index[i]
        for event in events:
            if event["shares"] is None and stamp.date() >= event["ex_date"]:
                # Ex-date entitlement is determined by the position carried through the prior close.
                event["shares"] = prior_close_shares
                event["amount"] = prior_close_shares * event["rate"]
            if event["shares"] is not None and not event["paid"] and event["pay_date"] and stamp.date() >= event["pay_date"]:
                dividend_cash += event["amount"]
                event["paid"] = True
        if i > 1:
            prev, prior = history.iloc[i - 1], history.iloc[i - 2]
            signal_time = history.index[i - 1]
            up = prior.macd <= prior.signal and prev.macd > prev.signal
            down = prior.macd >= prior.signal and prev.macd < prev.signal
            if policy == "MACD with daily EMA200 filter":
                buy, sell = up and bool(prev.regime_up), down
            else:
                buy = prev.macd < 0 and prev.macd > prior.macd
                sell = prev.macd > 0 and prev.macd < prior.macd
            if shares and sell:
                price = float(row.open) * (1 - slippage_bps / 10000)
                proceeds = shares * (price - fee_per_share)
                cash += proceeds
                trades.append({"Side": "Sell", "Signal time": signal_time, "Fill time": stamp,
                               "Price": price, "Shares": shares, "P&L": proceeds - entry["cost"]})
                shares, entry = 0, None
            if not shares and buy:
                price = float(row.open) * (1 + slippage_bps / 10000)
                qty = int(cash / (price + fee_per_share))
                if qty:
                    cost = qty * (price + fee_per_share)
                    cash -= cost
                    shares, entry = qty, {"cost": cost}
                    trades.append({"Side": "Buy", "Signal time": signal_time, "Fill time": stamp,
                                   "Price": price, "Shares": qty, "P&L": 0.0})
        receivable = sum(e["amount"] for e in events if e["shares"] is not None and not e["paid"])
        equity.append(cash + dividend_cash + shares * float(row.close) + receivable)
        last_bar = i == len(history) - 1 or history.index[i + 1].date() != stamp.date()
        if shares and last_bar and not overnight:
            price = float(row.close) * (1 - slippage_bps / 10000)
            cash += shares * (price - fee_per_share)
            trades.append({"Side": "Final liquidation", "Signal time": stamp, "Fill time": stamp,
                           "Price": price, "Shares": shares, "P&L": shares * (price - fee_per_share) - entry["cost"]})
            shares, entry = 0, None
            equity[-1] = cash + dividend_cash + receivable
        if last_bar:
            prior_close_shares = shares
    data["equity"] = equity
    data["invested"] = [bool(value) for value in _position_history(history, start_i, policy, overnight)]
    ledger = [{"Ex-date": e["ex_date"], "Payment date": e["pay_date"], "Rate / share": e["rate"],
               "Eligible shares": e["shares"] or 0, "Distribution": e["amount"],
               "Status": "Cash paid" if e["paid"] else "Receivable" if e["shares"] else "Not eligible"}
              for e in events]
    return data, trades, ledger


def _position_history(history: pd.DataFrame, start_i: int, policy: str, overnight: bool) -> list[int]:
    """Tiny pass used for exposure; mirror entries/exits without cash or accounting."""
    holding, result = False, []
    for i in range(start_i, len(history)):
        if i > 1:
            prev, prior = history.iloc[i - 1], history.iloc[i - 2]
            up = prior.macd <= prior.signal and prev.macd > prev.signal
            down = prior.macd >= prior.signal and prev.macd < prev.signal
            buy = up and bool(prev.regime_up) if policy == "MACD with daily EMA200 filter" else prev.macd < 0 and prev.macd > prior.macd
            sell = down if policy == "MACD with daily EMA200 filter" else prev.macd > 0 and prev.macd < prior.macd
            holding = False if holding and sell else holding
            holding = True if not holding and buy else holding
        if not overnight and (i == len(history) - 1 or history.index[i + 1].date() != history.index[i].date()):
            holding = False
        result.append(int(holding))
    return result


def run_buy_hold(data: pd.DataFrame, capital: float, fee_per_share: float, slippage_bps: float,
                 distributions: list[dict]) -> tuple[pd.Series, list[dict]]:
    events = [dict(event) for event in distributions]
    entry_date = data.index[0].date()
    entry_price = float(data.open.iloc[0]) * (1 + slippage_bps / 10000)
    shares = int(capital / (entry_price + fee_per_share))
    cash, equity = capital - shares * (entry_price + fee_per_share), []
    for stamp, row in data.iterrows():
        for event in events:
            if event["shares"] is None and stamp.date() >= event["ex_date"]:
                event["shares"] = shares if entry_date < event["ex_date"] else 0
                event["amount"] = event["shares"] * event["rate"]
            if event["shares"] is not None and not event["paid"] and event["pay_date"] and stamp.date() >= event["pay_date"]:
                cash += event["amount"]
                event["paid"] = True
        receivable = sum(e["amount"] for e in events if e["shares"] is not None and not e["paid"])
        equity.append(cash + receivable + shares * float(row.close))
    equity[-1] += -shares * float(data.close.iloc[-1]) * slippage_bps / 10000 - shares * fee_per_share
    ledger = [{"Ex-date": e["ex_date"], "Payment date": e["pay_date"], "Rate / share": e["rate"],
               "Eligible shares": e["shares"] or 0, "Distribution": e["amount"],
               "Status": "Cash paid" if e["paid"] else "Receivable" if e["shares"] else "Not eligible"}
              for e in events]
    return pd.Series(equity, index=data.index), ledger


def run_daily_policy(frame: pd.DataFrame, initial_cash: float, policy: str, fee_per_share: float,
                     slippage_bps: float, range_start: pd.Timestamp,
                     distributions: list[dict] | None = None) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Run a fixed long/cash daily rule; yesterday's close signal fills at today's open."""
    history = frame.copy()
    history["sma50"] = history.close.rolling(50, min_periods=50).mean()
    history["sma200"] = history.close.rolling(200, min_periods=200).mean()
    if policy == "Close above 200-day SMA":
        history["target"] = history.close > history.sma200
    elif policy == "50/200-day SMA cross":
        history["target"] = history.sma50 > history.sma200
    else:
        raise ValueError(f"Unknown daily policy: {policy}")
    history["target"] = history.target.shift(1).fillna(False).astype(bool)
    history = history.dropna(subset=["sma200"])
    start_i = int(history.index.searchsorted(range_start))
    data = history.iloc[start_i:].copy()
    if data.empty:
        return data, [], []
    events = [dict(event) for event in (distributions or [])]
    cash, dividend_cash, shares, prior_close_shares = initial_cash, 0.0, 0, 0
    trades, equity = [], []
    for i in range(start_i, len(history)):
        row, stamp = history.iloc[i], history.index[i]
        for event in events:
            if event["shares"] is None and stamp.date() >= event["ex_date"]:
                event["shares"] = prior_close_shares
                event["amount"] = prior_close_shares * event["rate"]
            if event["shares"] is not None and not event["paid"] and event["pay_date"] and stamp.date() >= event["pay_date"]:
                dividend_cash += event["amount"]
                event["paid"] = True
        signal_time = history.index[i - 1] if i else stamp
        target = bool(row.target)
        if target and not shares:
            price = float(row.open) * (1 + slippage_bps / 10000)
            qty = int(cash / (price + fee_per_share))
            if qty:
                cash -= qty * (price + fee_per_share)
                shares = qty
                trades.append({"Side": "Buy", "Signal time": signal_time, "Fill time": stamp,
                               "Price": price, "Shares": qty})
        elif shares and not target:
            price = float(row.open) * (1 - slippage_bps / 10000)
            cash += shares * (price - fee_per_share)
            trades.append({"Side": "Sell", "Signal time": signal_time, "Fill time": stamp,
                           "Price": price, "Shares": shares})
            shares = 0
        receivable = sum(e["amount"] for e in events if e["shares"] is not None and not e["paid"])
        equity.append(cash + dividend_cash + shares * float(row.close) + receivable)
        prior_close_shares = shares
    if shares:
        price = float(history.close.iloc[-1]) * (1 - slippage_bps / 10000)
        cash += shares * (price - fee_per_share)
        trades.append({"Side": "Final liquidation", "Signal time": history.index[-1],
                       "Fill time": history.index[-1], "Price": price, "Shares": shares})
        receivable = sum(e["amount"] for e in events if e["shares"] is not None and not e["paid"])
        equity[-1] = cash + dividend_cash + receivable
    data["equity"] = equity
    data["invested"] = data.target
    ledger = [{"Ex-date": e["ex_date"], "Payment date": e["pay_date"], "Rate / share": e["rate"],
               "Eligible shares": e["shares"] or 0, "Distribution": e["amount"],
               "Status": "Cash paid" if e["paid"] else "Receivable" if e["shares"] else "Not eligible"}
              for e in events]
    return data, trades, ledger


def rsi_indicators(frame: pd.DataFrame, length: int = 14, buy_level: float = 30,
                   sell_level: float = 70) -> pd.DataFrame:
    """Add Wilder RSI and confirmed threshold-cross signals; no signal before RSI is seeded."""
    if length < 1 or not 0 <= buy_level < sell_level <= 100:
        raise ValueError("RSI length must be positive and thresholds must satisfy 0 <= buy < sell <= 100.")
    result = frame.copy()
    changes = result.close.astype(float).diff().tolist()
    rsi = [math.nan] * len(changes)
    if len(changes) > length:
        gains = [max(change, 0.0) for change in changes[1:length + 1]]
        losses = [max(-change, 0.0) for change in changes[1:length + 1]]
        avg_gain, avg_loss = sum(gains) / length, sum(losses) / length
        for i in range(length, len(changes)):
            if i > length:
                avg_gain = (avg_gain * (length - 1) + max(changes[i], 0.0)) / length
                avg_loss = (avg_loss * (length - 1) + max(-changes[i], 0.0)) / length
            rsi[i] = 50.0 if avg_gain == avg_loss == 0 else 100.0 if avg_loss == 0 else (
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    result["rsi"] = rsi
    prior = result.rsi.shift(1)
    result["rsi_buy"] = (prior < buy_level) & (result.rsi >= buy_level)
    result["rsi_sell"] = (prior > sell_level) & (result.rsi <= sell_level)
    return result


def run_rsi_policy(frame: pd.DataFrame, initial_cash: float, fee_per_share: float,
                   slippage_bps: float, range_start: pd.Timestamp,
                   distributions: list[dict] | None = None, length: int = 14,
                   buy_level: float = 30, sell_level: float = 70) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Trade confirmed daily RSI recoveries at the next open; start in cash and hold overnight."""
    history = rsi_indicators(frame, length, buy_level, sell_level).dropna(subset=["rsi"])
    start_i = int(history.index.searchsorted(range_start))
    data = history.iloc[start_i:].copy()
    if data.empty:
        return data, [], []
    events = [dict(event) for event in (distributions or [])]
    cash, dividend_cash, shares, prior_close_shares = initial_cash, 0.0, 0, 0
    trades, equity, invested = [], [], []
    for i in range(start_i, len(history)):
        row, stamp = history.iloc[i], history.index[i]
        for event in events:
            if event["shares"] is None and stamp.date() >= event["ex_date"]:
                event["shares"] = prior_close_shares
                event["amount"] = prior_close_shares * event["rate"]
            if event["shares"] is not None and not event["paid"] and event["pay_date"] and stamp.date() >= event["pay_date"]:
                dividend_cash += event["amount"]
                event["paid"] = True
        previous = history.iloc[i - 1] if i else None
        signal_time = history.index[i - 1] if previous is not None else stamp
        buy = bool(previous.rsi_buy) if previous is not None else False
        sell = bool(previous.rsi_sell) if previous is not None else False
        if buy and not shares:
            price = float(row.open) * (1 + slippage_bps / 10000)
            quantity = int(cash / (price + fee_per_share))
            if quantity:
                cash -= quantity * (price + fee_per_share)
                shares = quantity
                trades.append({"Side": "Buy", "Signal time": signal_time, "Fill time": stamp,
                               "Price": price, "Shares": quantity})
        elif sell and shares:
            price = float(row.open) * (1 - slippage_bps / 10000)
            cash += shares * (price - fee_per_share)
            trades.append({"Side": "Sell", "Signal time": signal_time, "Fill time": stamp,
                           "Price": price, "Shares": shares})
            shares = 0
        receivable = sum(event["amount"] for event in events if event["shares"] is not None and not event["paid"])
        equity.append(cash + dividend_cash + shares * float(row.close) + receivable)
        invested.append(bool(shares))
        prior_close_shares = shares
    if shares:
        stamp, row = history.index[-1], history.iloc[-1]
        price = float(row.close) * (1 - slippage_bps / 10000)
        cash += shares * (price - fee_per_share)
        trades.append({"Side": "Final liquidation", "Signal time": stamp, "Fill time": stamp,
                       "Price": price, "Shares": shares})
        receivable = sum(event["amount"] for event in events if event["shares"] is not None and not event["paid"])
        equity[-1] = cash + dividend_cash + receivable
    data["equity"], data["invested"] = equity, invested
    ledger = [{"Ex-date": event["ex_date"], "Payment date": event["pay_date"], "Rate / share": event["rate"],
               "Eligible shares": event["shares"] or 0, "Distribution": event["amount"],
               "Status": "Cash paid" if event["paid"] else "Receivable" if event["shares"] else "Not eligible"}
              for event in events]
    return data, trades, ledger


def supertrend_indicators(frame: pd.DataFrame, atr_length: int = 7,
                          multiplier: float = 3.0) -> pd.DataFrame:
    """Add TradingView-style SuperTrend bands, line, direction, and confirmed flips.

    TR is max(high-low, abs(high-prev_close), abs(low-prev_close)); the first
    bar uses high-low. ATR is Wilder's RMA, seeded with the SMA of the first
    ``atr_length`` true ranges, then ``(previous * (length - 1) + TR) / length``.
    Basic bands are HL2 +/- multiplier*ATR. Final upper resets to its basic
    value if lower than its prior value or the prior close broke above the
    prior upper; final lower resets if higher than its prior value or the prior
    close broke below the prior lower. Direction starts bearish on the first
    valid ATR bar, then flips only when a completed close crosses its active
    band. This follows TradingView's published formula; feed/seed history may
    still change plotted values relative to another vendor.
    """
    if atr_length < 1 or multiplier <= 0:
        raise ValueError("ATR length and multiplier must be positive.")

    result = frame.copy()
    high, low, close = (result[column].astype(float).tolist() for column in ("high", "low", "close"))
    count = len(result)
    true_range = [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
                  if i else high[i] - low[i] for i in range(count)]
    atr = [math.nan] * count
    upper = [math.nan] * count
    lower = [math.nan] * count
    line = [math.nan] * count
    bullish = [False] * count
    buy = [False] * count
    sell = [False] * count
    if count >= atr_length:
        atr[atr_length - 1] = sum(true_range[:atr_length]) / atr_length
        for i in range(atr_length, count):
            atr[i] = (atr[i - 1] * (atr_length - 1) + true_range[i]) / atr_length
        for i in range(atr_length - 1, count):
            midpoint = (high[i] + low[i]) / 2
            basic_upper, basic_lower = midpoint + multiplier * atr[i], midpoint - multiplier * atr[i]
            if i == atr_length - 1:
                upper[i], lower[i] = basic_upper, basic_lower
            else:
                upper[i] = basic_upper if basic_upper < upper[i - 1] or close[i - 1] > upper[i - 1] else upper[i - 1]
                lower[i] = basic_lower if basic_lower > lower[i - 1] or close[i - 1] < lower[i - 1] else lower[i - 1]
                bullish[i] = close[i] > upper[i] if not bullish[i - 1] else close[i] >= lower[i]
                buy[i] = bullish[i] and not bullish[i - 1]
                sell[i] = not bullish[i] and bullish[i - 1]
            line[i] = lower[i] if bullish[i] else upper[i]

    result["atr"] = atr
    result["supertrend_upper"] = upper
    result["supertrend_lower"] = lower
    result["supertrend_line"] = line
    result["supertrend_bullish"] = bullish
    result["supertrend_buy"] = buy
    result["supertrend_sell"] = sell
    return result


def run_supertrend_policy(frame: pd.DataFrame, initial_cash: float, fee_per_share: float,
                          slippage_bps: float, range_start: pd.Timestamp,
                          distributions: list[dict] | None = None, atr_length: int = 7,
                          multiplier: float = 3.0) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Follow confirmed SuperTrend flips, filling the next bar open and holding overnight."""
    history = supertrend_indicators(frame, atr_length, multiplier).dropna(subset=["supertrend_line"])
    start_i = int(history.index.searchsorted(range_start))
    data = history.iloc[start_i:].copy()
    if data.empty:
        return data, [], []

    events = [dict(event) for event in (distributions or [])]
    cash, dividend_cash, shares, prior_close_shares = initial_cash, 0.0, 0, 0
    trades, equity, invested = [], [], []
    for i in range(start_i, len(history)):
        row, stamp = history.iloc[i], history.index[i]
        for event in events:
            if event["shares"] is None and stamp.date() >= event["ex_date"]:
                event["shares"] = prior_close_shares
                event["amount"] = prior_close_shares * event["rate"]
            if event["shares"] is not None and not event["paid"] and event["pay_date"] and stamp.date() >= event["pay_date"]:
                dividend_cash += event["amount"]
                event["paid"] = True

        previous = history.iloc[i - 1] if i else None
        signal_time = history.index[i - 1] if i else stamp
        buy = bool(previous.supertrend_buy) if previous is not None else False
        sell = bool(previous.supertrend_sell) if previous is not None else False
        if buy and not shares:
            price = float(row.open) * (1 + slippage_bps / 10000)
            quantity = int(cash / (price + fee_per_share))
            if quantity:
                cash -= quantity * (price + fee_per_share)
                shares = quantity
                trades.append({"Side": "Buy", "Signal time": signal_time, "Fill time": stamp,
                               "Price": price, "Shares": quantity})
        elif sell and shares:
            price = float(row.open) * (1 - slippage_bps / 10000)
            cash += shares * (price - fee_per_share)
            trades.append({"Side": "Sell", "Signal time": signal_time, "Fill time": stamp,
                           "Price": price, "Shares": shares})
            shares = 0

        receivable = sum(event["amount"] for event in events if event["shares"] is not None and not event["paid"])
        equity.append(cash + dividend_cash + shares * float(row.close) + receivable)
        invested.append(bool(shares))
        prior_close_shares = shares

    if shares:
        stamp, row = history.index[-1], history.iloc[-1]
        price = float(row.close) * (1 - slippage_bps / 10000)
        cash += shares * (price - fee_per_share)
        trades.append({"Side": "Final liquidation", "Signal time": stamp, "Fill time": stamp,
                       "Price": price, "Shares": shares})
        receivable = sum(event["amount"] for event in events if event["shares"] is not None and not event["paid"])
        equity[-1] = cash + dividend_cash + receivable

    data["equity"] = equity
    data["invested"] = invested
    ledger = [{"Ex-date": event["ex_date"], "Payment date": event["pay_date"], "Rate / share": event["rate"],
               "Eligible shares": event["shares"] or 0, "Distribution": event["amount"],
               "Status": "Cash paid" if event["paid"] else "Receivable" if event["shares"] else "Not eligible"}
              for event in events]
    return data, trades, ledger
