from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from macd_backtest_core import (NY, dividend_events, fetch_bars, fetch_corporate_actions,
                                fetch_daily_bars, run_buy_hold, run_daily_policy, run_policy,
                                run_rsi_policy, run_supertrend_policy)

INTRADAY_POLICIES = ("MACD turning up below zero", "MACD with daily EMA200 filter")
DAILY_POLICIES = ("Close above 200-day SMA", "50/200-day SMA cross", "RSI recovery (14, 30/70)")
SUPERTREND_POLICY = "SuperTrend (ATR arrows)"
ALL_POLICIES = INTRADAY_POLICIES + DAILY_POLICIES + (SUPERTREND_POLICY,)
POLICY_LABELS = {
    "Close above 200-day SMA": "Daily 200-day SMA trend (close above SMA200)",
    "50/200-day SMA cross": "Daily 50/200-day SMA cross",
    "RSI recovery (14, 30/70)": "Daily RSI recovery (buy 30 / sell 70)",
    "MACD turning up below zero": "MACD turning up below zero",
    "MACD with daily EMA200 filter": "MACD with daily EMA200 filter",
    SUPERTREND_POLICY: "SuperTrend (ATR arrows)",
}


def _sync_interval_to_policy() -> None:
    if st.session_state.policy in DAILY_POLICIES:
        st.session_state.bar_interval = "Daily"
    elif st.session_state.bar_interval == "Daily" and st.session_state.policy != SUPERTREND_POLICY:
        st.session_state.bar_interval = 10


def _sync_policy_to_interval() -> None:
    if st.session_state.bar_interval == "Daily" and st.session_state.policy not in DAILY_POLICIES + (SUPERTREND_POLICY,):
        st.session_state.policy = DAILY_POLICIES[0]
    elif st.session_state.bar_interval != "Daily" and st.session_state.policy in DAILY_POLICIES:
        st.session_state.policy = INTRADAY_POLICIES[0]


st.set_page_config(page_title="QQQ Strategy Backtest", layout="wide")
st.title("QQQ Historical Strategy Comparison")
st.caption("Historical Alpaca bars only. Simulated trades do not submit orders.")
st.info("Bars are split-adjusted. Dividends are held as cash, based on shares carried past the prior session close; they are credited on payment date. Amounts are gross before tax or withholding. Corporate-action records may be delayed.")

st.session_state.setdefault("bar_interval", 10)
st.session_state.setdefault("policy", INTRADAY_POLICIES[0])

with st.sidebar:
    st.header("Backtest setup")
    symbol = st.text_input("US symbol", "QQQ").strip().upper()
    capital = st.number_input("Initial capital ($)", min_value=100.0, value=100000.0, step=5000.0)
    interval = st.selectbox("Bar interval", [1, 5, 10, 15, 30, 60, "Daily"],
                            format_func=lambda n: "Daily" if n == "Daily" else f"{n} minute" if n == 1 else f"{n} minutes",
                            key="bar_interval", on_change=_sync_policy_to_interval)
    today = datetime.now(NY).date()
    start_date = st.date_input("Start date", today - timedelta(days=365), max_value=today, key="date_start")
    end_date = st.date_input("End date", today, max_value=today, key="date_end")
    policy = st.selectbox("Policy", ALL_POLICIES,
                          format_func=lambda value: POLICY_LABELS[value], key="policy",
                          on_change=_sync_interval_to_policy)
    daily = interval == "Daily"
    if policy == SUPERTREND_POLICY:
        atr_length = int(st.number_input("SuperTrend ATR length", min_value=1, max_value=100,
                                         value=7, step=1, key="supertrend_atr_length"))
        multiplier = st.number_input("SuperTrend multiplier", min_value=0.1, max_value=10.0,
                                     value=3.0, step=0.1, key="supertrend_multiplier")
        st.caption("Confirmed close-to-close flips only: buy on a bullish flip and sell on a bearish flip at the next bar open. ATR uses Wilder RMA, seeded by the mean of the first N true ranges; direction starts bearish until ATR is ready. No initial entry. Positions are held overnight; daily bars are not flattened at session end.")
    if policy == "RSI recovery (14, 30/70)":
        rsi_length = int(st.number_input("RSI length", min_value=1, max_value=100, value=14, step=1, key="rsi_length"))
        rsi_buy = st.number_input("Buy threshold", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="rsi_buy")
        rsi_sell = st.number_input("Sell threshold", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="rsi_sell")
        st.caption("Wilder RSI with an initial simple-average seed. Buy when RSI recovers across the buy threshold; sell when it falls across the sell threshold. Signals are confirmed at the daily close and fill at the next session open. Starts in cash, holds overnight, and does not pyramid.")
    fee = st.number_input("Commission per share per fill ($)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
    slippage = st.number_input("Adverse slippage per fill (bps)", min_value=0.0, value=5.0, step=1.0)
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)
    run = st.button("Run comparison", type="primary")

st.markdown("**Daily policies:** close above the 200-day SMA stays long; close below it moves to cash. The 50/200-day SMA rule stays long when SMA50 is above SMA200 and moves to cash below it. Daily signals use the prior completed session and fill at the next session open. RSI starts in cash, buys threshold recoveries, and sells threshold declines. Intraday MACD policies remain available when a minute interval is selected.")
if run:
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
    if policy == "RSI recovery (14, 30/70)" and rsi_buy >= rsi_sell:
        st.error("The buy threshold must be lower than the sell threshold.")
        st.stop()
    start = datetime.combine(start_date, time.min, NY).astimezone(timezone.utc)
    end = min(datetime.combine(end_date + timedelta(days=1), time.min, NY).astimezone(timezone.utc), datetime.now(timezone.utc) - timedelta(minutes=20))
    warm_start = datetime.combine(start_date - timedelta(days=400), time.min, NY).astimezone(timezone.utc)
    with st.spinner("Loading historical bars and cash distributions from Alpaca…"):
        try:
            frame = fetch_daily_bars(symbol, warm_start, end, feed) if daily else fetch_bars(symbol, interval, warm_start, end, feed)
            actions = fetch_corporate_actions(symbol, start_date - timedelta(days=280), end_date)
        except Exception as exc:
            st.error(f"Alpaca data request failed: {exc}")
            st.stop()
    range_start = pd.Timestamp(datetime.combine(start_date, time.min, NY))
    selected = frame[frame.index >= range_start - pd.Timedelta(days=400)]
    if frame.empty or frame[frame.index >= range_start].empty:
        st.warning("No regular-session bars were returned for this date range.")
        st.stop()
    distributions = dividend_events(actions, start_date, end_date)
    if policy == SUPERTREND_POLICY:
        result, trades, ledger = run_supertrend_policy(selected, capital, fee, slippage, range_start,
                                                        distributions, atr_length, multiplier)
    elif policy == "RSI recovery (14, 30/70)":
        result, trades, ledger = run_rsi_policy(selected, capital, fee, slippage, range_start,
                                                 distributions, rsi_length, rsi_buy, rsi_sell)
    elif daily:
        result, trades, ledger = run_daily_policy(selected, capital, policy, fee, slippage, range_start, distributions)
    else:
        overnight = policy == "MACD with daily EMA200 filter"
        result, trades, ledger = run_policy(selected, capital, policy, overnight, fee, slippage, range_start, distributions)
    if result.empty:
        st.warning("The available history is too short to warm the selected moving average.")
        st.stop()
    benchmark, benchmark_ledger = run_buy_hold(result, capital, fee, slippage, distributions)
    final_value = float(result.equity.iloc[-1])
    peak = result.equity.cummax()
    drawdown = min(float((result.equity / peak - 1).min()), float(result.equity.iloc[0] / capital - 1)) * 100
    bh_value = float(benchmark.iloc[-1])
    columns = st.columns(4)
    columns[0].metric("Final strategy value", f"${final_value:,.2f}", f"{final_value / capital - 1:.2%}")
    columns[1].metric("Buy and hold", f"${bh_value:,.2f}", f"{bh_value / capital - 1:.2%}")
    columns[2].metric("Max drawdown", f"{drawdown:.2f}%")
    columns[3].metric("Time invested", f"{result.invested.mean():.1%}")
    if fee == 0 or slippage == 0:
        st.warning("Any cost set to $0 is excluded. Confirm fees and slippage before relying on results.")
    timeframe = "daily bars" if daily else f"{interval}-minute regular-session bars"
    st.caption(f"{len(result):,} {timeframe}, {result.index[0]:%Y-%m-%d} to {result.index[-1]:%Y-%m-%d}; {len(trades)} fill events; {feed.upper()} feed. Prices are split-adjusted.")
    if policy == SUPERTREND_POLICY:
        st.caption(f"SuperTrend ATR({atr_length}) uses Wilder RMA seeded by the mean of its first {atr_length} true ranges, with multiplier {multiplier:g}; direction starts bearish on the first valid ATR bar. Signals use a completed bar and fill at the next {('session' if daily else 'regular-session')} open. Chart arrows mark signals; circles mark fills. No EOD flattening; final liquidation is at the last close.")
    if policy == "RSI recovery (14, 30/70)":
        st.caption(f"RSI({rsi_length}) uses Wilder smoothing seeded by the first {rsi_length} daily close changes. Flat prices yield 50; zero average loss yields 100. No signal is generated until RSI and its prior value are valid. Chart triangles mark confirmed signals; price chart circles mark actual fills. Overnight positions are liquidated at the final close for end-of-range valuation.")
    st.subheader("Cash distributions (not reinvested)")
    left, right = st.columns(2)
    left.caption("Selected policy")
    left.dataframe(pd.DataFrame(ledger), use_container_width=True, hide_index=True)
    right.caption("Buy and hold")
    right.dataframe(pd.DataFrame(benchmark_ledger), use_container_width=True, hide_index=True)
    price = result.reset_index().rename(columns={result.index.name or "index": "Time"})
    time_axis = "Time (New York)" if policy == SUPERTREND_POLICY else "Time"
    if policy == SUPERTREND_POLICY:
        price["Time"] = pd.to_datetime(price["Time"]).dt.tz_localize(None)
    line = alt.Chart(price).mark_line(color="#2563eb").encode(x=alt.X("Time:T", title=time_axis), y=alt.Y("close:Q", title=f"{symbol} price (USD)", scale=alt.Scale(zero=False)))
    if policy == SUPERTREND_POLICY:
        bull_line = (alt.Chart(price).transform_filter(alt.datum.supertrend_bullish == True)
                     .mark_line(color="#16a34a", strokeWidth=2)
                     .encode(x="Time:T", y=alt.Y("supertrend_line:Q", title=f"{symbol} price / SuperTrend (USD)")))
        bear_line = (alt.Chart(price).transform_filter(alt.datum.supertrend_bullish == False)
                     .mark_line(color="#dc2626", strokeWidth=2)
                     .encode(x="Time:T", y=alt.Y("supertrend_line:Q", title=f"{symbol} price / SuperTrend (USD)")))
        signals = price[price.supertrend_buy | price.supertrend_sell].copy()
        signals["Arrow"] = signals.supertrend_buy.map({True: "↑", False: "↓"})
        signals["Arrow price"] = signals.apply(lambda row: row.low if row.supertrend_buy else row.high, axis=1)
        signals["Signal color"] = signals.supertrend_buy.map({True: "#16a34a", False: "#dc2626"})
        arrows = (alt.Chart(signals).mark_text(fontSize=18, fontWeight="bold")
                  .encode(x="Time:T", y="Arrow price:Q", text="Arrow:N", color=alt.Color("Signal color:N", scale=None),
                          tooltip=[alt.Tooltip("Time:T"), alt.Tooltip("Arrow:N", title="Signal")]))
        line = line + bull_line + bear_line + arrows
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        buys = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
        sells = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
    else:
        if policy == SUPERTREND_POLICY:
            trades_df["Fill time"] = pd.to_datetime(trades_df["Fill time"]).dt.tz_localize(None)
        buy_shape, sell_shape = ("circle", "circle") if policy == SUPERTREND_POLICY else ("triangle-up", "triangle-down")
        buys = alt.Chart(trades_df[trades_df.Side == "Buy"]).mark_point(shape=buy_shape, filled=True, size=75, color="#16a34a").encode(x="Fill time:T", y="Price:Q")
        sells = alt.Chart(trades_df[trades_df.Side != "Buy"]).mark_point(shape=sell_shape, filled=True, size=75, color="#dc2626").encode(x="Fill time:T", y="Price:Q")
    st.subheader("Price and fills")
    st.altair_chart(line + buys + sells, use_container_width=True)
    if policy == "RSI recovery (14, 30/70)":
        rsi_frame = result.reset_index().rename(columns={result.index.name or "index": "Time"})
        signal_points = rsi_frame[rsi_frame.rsi_buy | rsi_frame.rsi_sell].copy()
        signal_points["Signal"] = signal_points.rsi_buy.map({True: "Buy signal", False: "Sell signal"})
        rsi_line = alt.Chart(rsi_frame).mark_line(color="#7c3aed").encode(
            x=alt.X("Time:T", title="Time"), y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100])))
        levels = alt.Chart(pd.DataFrame({"level": [rsi_buy, rsi_sell]})).mark_rule(
            color="#64748b", strokeDash=[5, 4]).encode(y="level:Q")
        signals = alt.Chart(signal_points).mark_point(filled=True, size=80).encode(
            x="Time:T", y="rsi:Q", shape=alt.Shape("Signal:N", scale=alt.Scale(domain=["Buy signal", "Sell signal"], range=["triangle-up", "triangle-down"]), legend=None),
            color=alt.Color("Signal:N", scale=alt.Scale(domain=["Buy signal", "Sell signal"], range=["#16a34a", "#dc2626"]), legend=None),
            tooltip=[alt.Tooltip("Time:T"), alt.Tooltip("Signal:N"), alt.Tooltip("rsi:Q", format=".2f")])
        fills = trades_df[trades_df.Side != "Final liquidation"].copy()
        if not fills.empty:
            fills["RSI"] = fills["Signal time"].map(result.rsi)
            fill_points = alt.Chart(fills).mark_point(shape="circle", filled=True, size=55, color="#111827").encode(
                x=alt.X("Fill time:T", title="Time"), y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Fill time:T"), alt.Tooltip("Side:N"), alt.Tooltip("Price:Q", format="$.2f")])
            rsi_chart = rsi_line + levels + signals + fill_points
        else:
            rsi_chart = rsi_line + levels + signals
        st.subheader("RSI, signals, and fills")
        st.altair_chart(rsi_chart, use_container_width=True)
    st.subheader("Equity curve")
    curves = pd.DataFrame({"Strategy": result.equity, "Buy and hold": benchmark})
    st.line_chart(curves)
    if not trades_df.empty:
        st.subheader("Fills")
        st.dataframe(trades_df.iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)
