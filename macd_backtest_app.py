from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from macd_backtest_core import (NY, dividend_events, fetch_bars, fetch_corporate_actions,
                                fetch_corporate_actions_many, fetch_daily_bars, fetch_daily_bars_many,
                                run_accumulation_policy, run_buy_hold, run_daily_policy, run_policy,
                                run_relative_momentum_policy, run_rsi_policy, run_supertrend_policy)

INTRADAY_POLICIES = ("MACD turning up below zero", "MACD with daily EMA200 filter")
MOMENTUM_POLICY = "12-month absolute momentum"
ACCUMULATION_POLICY = "Accumulation breakout"
BOLLINGER_POLICY = "Bollinger Band mean reversion"
MONTHLY_TREND_POLICY = "Monthly 10-month SMA risk filter"
RELATIVE_MOMENTUM_POLICY = "ETF relative momentum rotation"
ROTATION_UNIVERSE = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLY", "XLI", "XLP",
                     "XLU", "XLV", "XLE", "XLB", "SMH")
DAILY_POLICIES = ("Close above 200-day SMA", "50/200-day SMA cross", "RSI recovery (14, 30/70)",
                  BOLLINGER_POLICY, MONTHLY_TREND_POLICY, RELATIVE_MOMENTUM_POLICY,
                  MOMENTUM_POLICY, ACCUMULATION_POLICY)
SUPERTREND_POLICY = "SuperTrend (ATR arrows)"
ALL_POLICIES = INTRADAY_POLICIES + DAILY_POLICIES + (SUPERTREND_POLICY,)
POLICY_LABELS = {
    "Close above 200-day SMA": "Daily 200-day SMA trend (close above SMA200)",
    "50/200-day SMA cross": "Daily 50/200-day SMA cross",
    "RSI recovery (14, 30/70)": "Daily RSI recovery (buy 30 / sell 70)",
    BOLLINGER_POLICY: "Daily Bollinger Bands (mean reversion)",
    MONTHLY_TREND_POLICY: "Monthly 10-month SMA risk filter",
    RELATIVE_MOMENTUM_POLICY: "Monthly ETF relative momentum rotation",
    MOMENTUM_POLICY: "Monthly 12-month absolute momentum (cash when return ≤ 0)",
    ACCUMULATION_POLICY: "Accumulation breakout (base + volume confirmation)",
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


st.set_page_config(page_title="Historical Strategy Backtest", layout="wide")
st.title("Historical Strategy Comparison")
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
    if policy == ACCUMULATION_POLICY:
        base_days = int(st.number_input("Base window (sessions)", min_value=20, max_value=120, value=60, key="accum_base_days"))
        max_base_range = st.number_input("Maximum base range (%)", min_value=1.0, max_value=50.0, value=15.0, step=1.0, key="accum_max_range")
        breakout_volume = st.number_input("Breakout volume / prior 20-day average", min_value=1.0, max_value=10.0, value=1.5, step=0.1, key="accum_volume_multiple")
        cmf_threshold = st.number_input("Minimum 20-day CMF", min_value=-1.0, max_value=1.0, value=0.05, step=0.05, key="accum_cmf_threshold")
        st.caption("A price/volume proxy, not buyer identification. It looks for a tight base, positive money flow, and a high-volume breakout. Entry fills at the next session open.")
    if policy == BOLLINGER_POLICY:
        bollinger_length = int(st.number_input("Bollinger window (sessions)", min_value=5, max_value=100, value=20, step=1, key="bollinger_length"))
        bollinger_stddev = st.number_input("Bollinger standard deviations", min_value=0.5, max_value=4.0, value=2.0, step=0.1, key="bollinger_stddev")
        st.caption("Mean reversion: buy after a close below the lower band; sell after a close returns to the middle band. Signals fill at the next session open.")
    if policy == RELATIVE_MOMENTUM_POLICY:
        st.caption("At each completed month-end, rank SPY, QQQ, IWM, DIA, and sector ETFs by 12-month total return. Hold the strongest positive-return ETF only if it is above its 10-month average; otherwise hold cash. Rebalance at the next session open. The symbol field remains the buy-and-hold benchmark.")
    cash_yield = st.number_input("Annual yield while in cash (%)", min_value=0.0, max_value=20.0,
                                 value=0.0, step=0.25, key="cash_yield_pct")
    st.caption("Constant annual effective assumption, compounded per trading day on available cash and paid dividends. Applied to all policies and buy-and-hold; set it to your intended cash yield, or leave at 0%.")
    fee = st.number_input("Commission per share per fill ($)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
    slippage = st.number_input("Adverse slippage per fill (bps)", min_value=0.0, value=5.0, step=1.0)
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)
    if policy == ACCUMULATION_POLICY and feed == "iex":
        st.warning("IEX volume covers one exchange. Select SIP for broader market-wide volume when your account permits it.")
    run = st.button("Run comparison", type="primary")

st.markdown("**Daily policies:** close above the 200-day SMA stays long; close below it moves to cash. The 50/200-day SMA rule stays long when SMA50 is above SMA200 and moves to cash below it. RSI starts in cash and trades threshold recoveries. Bollinger mean reversion buys below the lower band and exits at the middle band. The monthly 10-month SMA filter stays invested during long-term uptrends and moves to cash after a month-end close below the average. ETF relative momentum rotates monthly into the strongest eligible index or sector ETF. Twelve-month absolute momentum uses trailing total return. Accumulation breakout looks for a tight base and rising buying pressure before a volume-confirmed breakout. Signals execute at the next trading session open. Intraday MACD policies remain available when a minute interval is selected.")
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
            if policy == RELATIVE_MOMENTUM_POLICY:
                rotation_symbols = list(dict.fromkeys((*ROTATION_UNIVERSE, symbol)))
                rotation_data = fetch_daily_bars_many(rotation_symbols, warm_start, end, feed)
                rotation_actions = fetch_corporate_actions_many(
                    rotation_symbols, start_date - timedelta(days=400), end_date)
                frame = rotation_data.get(symbol, pd.DataFrame())
                actions = rotation_actions.get(symbol, {})
            else:
                rotation_data = rotation_actions = None
                frame = fetch_daily_bars(symbol, warm_start, end, feed) if daily else fetch_bars(symbol, interval, warm_start, end, feed)
                actions = fetch_corporate_actions(symbol, start_date - timedelta(days=400), end_date)
        except Exception as exc:
            st.error(f"Alpaca data request failed: {exc}")
            st.stop()
    range_start = pd.Timestamp(datetime.combine(start_date, time.min, NY))
    selected = frame[frame.index >= range_start - pd.Timedelta(days=400)]
    if frame.empty or frame[frame.index >= range_start].empty:
        st.warning("No regular-session bars were returned for this date range.")
        st.stop()
    distributions = dividend_events(actions, start_date, end_date)
    momentum_dividends = dividend_events(actions, start_date - timedelta(days=400), end_date)
    if policy == SUPERTREND_POLICY:
        result, trades, ledger = run_supertrend_policy(selected, capital, fee, slippage, range_start,
                                                        distributions, atr_length, multiplier, cash_yield)
    elif policy == "RSI recovery (14, 30/70)":
        result, trades, ledger = run_rsi_policy(selected, capital, fee, slippage, range_start,
                                                 distributions, rsi_length, rsi_buy, rsi_sell, cash_yield)
    elif policy == ACCUMULATION_POLICY:
        result, trades, ledger = run_accumulation_policy(
            selected, capital, fee, slippage, range_start, distributions, cash_yield,
            base_days, max_base_range, breakout_volume, cmf_threshold)
    elif policy == RELATIVE_MOMENTUM_POLICY:
        rotation_frames = {ticker: bars[bars.index >= range_start - pd.Timedelta(days=400)]
                           for ticker, bars in rotation_data.items()}
        rotation_dividends = {ticker: dividend_events(rotation_actions.get(ticker, {}),
                                                       start_date - timedelta(days=400), end_date)
                              for ticker in rotation_frames}
        result, trades, ledger = run_relative_momentum_policy(
            rotation_frames, symbol, capital, fee, slippage, range_start, rotation_dividends, cash_yield)
    elif daily:
        result, trades, ledger = run_daily_policy(selected, capital, policy, fee, slippage, range_start,
                                                  distributions, cash_yield, momentum_dividends,
                                                  bollinger_length if policy == BOLLINGER_POLICY else 20,
                                                  bollinger_stddev if policy == BOLLINGER_POLICY else 2.0)
    else:
        overnight = policy == "MACD with daily EMA200 filter"
        result, trades, ledger = run_policy(selected, capital, policy, overnight, fee, slippage, range_start,
                                            distributions, cash_yield)
    if result.empty:
        st.warning("The available history is too short to warm the selected moving average.")
        st.stop()
    benchmark, benchmark_ledger = run_buy_hold(result, capital, fee, slippage, distributions, cash_yield)
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
    if policy == MOMENTUM_POLICY:
        st.caption(f"At month-end, the trailing 12-calendar-month total return includes reinvested cash distributions. A positive return holds the selected symbol; zero or negative return moves to cash. The decision takes effect at the next session open. Cash accrues at the selected constant {cash_yield:g}% annual effective yield, compounded over trading sessions; no taxes are modeled.")
    if policy == ACCUMULATION_POLICY:
        st.caption(f"Prior {base_days}-session range ≤ {max_base_range:g}%, CMF(20) ≥ {cmf_threshold:g}, then close above prior base high on volume ≥ {breakout_volume:g}× its prior 20-session average. Entry fills next open; exit is a close below SMA50 or prior base low. Cash yield: {cash_yield:g}% effective annual.")
    if policy == BOLLINGER_POLICY:
        st.caption(f"Bands use a {bollinger_length}-session simple average and {bollinger_stddev:g} population standard deviations. Buy after a close below the lower band; exit after a close reaches the middle band. Fills occur at the next session open. No stop-loss.")
    if policy == MONTHLY_TREND_POLICY:
        st.caption("Hold QQQ while the completed month-end close is above its 10-month simple moving average. Move to cash only after a completed month closes below it, and re-enter after a later month closes above it. The signal executes at the next trading session open; no intramonth exits.")
    if policy == RELATIVE_MOMENTUM_POLICY:
        st.caption("Rank the ETF universe by trailing 12-month total return at month-end. Hold the top ETF only when its 12-month return is positive and its month-end close is above its 10-month average; otherwise stay in cash. Trades occur at the next session open. The selected symbol is the benchmark.")
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
    if policy == BOLLINGER_POLICY:
        bands = alt.Chart(price).transform_fold(
            ["bb_middle", "bb_upper", "bb_lower"], as_=["Band", "Value"]
        ).mark_line(strokeDash=[4, 3], opacity=0.75).encode(
            x="Time:T", y="Value:Q", color=alt.Color("Band:N", title="Bollinger band"))
        line = line + bands
    if policy == MONTHLY_TREND_POLICY:
        trend_line = alt.Chart(price).mark_line(color="#f59e0b", strokeDash=[4, 3]).encode(
            x="Time:T", y=alt.Y("monthly_sma10:Q", title=f"{symbol} price / monthly SMA10 (USD)",
                                 scale=alt.Scale(zero=False)))
        line = line + trend_line
    trades_df = pd.DataFrame(trades)
    if policy == ACCUMULATION_POLICY:
        signal_sides = ("Buy", "Sell")
        signal_times = [trade["Signal time"] for trade in trades if trade["Side"] in signal_sides]
        setup_signals = price[price["Time"].isin(signal_times)].copy()
        setup_signals["Arrow"] = setup_signals.accumulation_buy.map({True: "↑", False: "↓"})
        setup_signals["Signal color"] = setup_signals.accumulation_buy.map({True: "#16a34a", False: "#dc2626"})
        upper = alt.Chart(price).mark_line(color="#94a3b8", strokeDash=[4, 3], opacity=0.7).encode(
            x="Time:T", y=alt.Y("base_high:Q", title=f"{symbol} price / prior base (USD)", scale=alt.Scale(zero=False)))
        lower = alt.Chart(price).mark_line(color="#94a3b8", strokeDash=[4, 3], opacity=0.7).encode(
            x="Time:T", y=alt.Y("base_low:Q", scale=alt.Scale(zero=False)))
        arrows = alt.Chart(setup_signals).mark_text(fontSize=18, fontWeight="bold").encode(
            x="Time:T", y=alt.Y("close:Q", scale=alt.Scale(zero=False)), text="Arrow:N",
            color=alt.Color("Signal color:N", scale=None),
            tooltip=[alt.Tooltip("Time:T"), alt.Tooltip("Arrow:N", title="Signal"),
                     alt.Tooltip("accum_cmf20:Q", format=".2f"),
                     alt.Tooltip("accum_volume_ratio:Q", format=".2f", title="Volume ratio")])
        line = line + upper + lower + arrows
    if trades_df.empty:
        buys = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
        sells = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
    elif policy == RELATIVE_MOMENTUM_POLICY:
        buys = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
        sells = alt.Chart(pd.DataFrame(columns=["Fill time", "Price"])).mark_point()
    else:
        if policy == SUPERTREND_POLICY:
            trades_df["Fill time"] = pd.to_datetime(trades_df["Fill time"]).dt.tz_localize(None)
        buy_shape, sell_shape = (("circle", "circle") if policy in (SUPERTREND_POLICY, ACCUMULATION_POLICY)
                                 else ("triangle-up", "triangle-down"))
        buys = alt.Chart(trades_df[trades_df.Side == "Buy"]).mark_point(shape=buy_shape, filled=True, size=75, color="#16a34a").encode(x="Fill time:T", y="Price:Q")
        sells = alt.Chart(trades_df[trades_df.Side != "Buy"]).mark_point(shape=sell_shape, filled=True, size=75, color="#dc2626").encode(x="Fill time:T", y="Price:Q")
    st.subheader(f"{symbol} benchmark price" if policy == RELATIVE_MOMENTUM_POLICY else "Price and fills")
    st.altair_chart(line + buys + sells, use_container_width=True)
    if policy == RELATIVE_MOMENTUM_POLICY:
        st.subheader("ETF held by the strategy")
        allocation = alt.Chart(price).mark_line(interpolate="step-after", strokeWidth=2).encode(
            x=alt.X("Time:T", title="Time"),
            y=alt.Y("rotation_symbol:N", title="Selected ETF", sort=list(ROTATION_UNIVERSE) + [symbol, "CASH"]))
        st.altair_chart(allocation, use_container_width=True)
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
    if policy == ACCUMULATION_POLICY:
        pressure = result.reset_index().rename(columns={result.index.name or "index": "Time"})
        pressure_signals = pressure[pressure["Time"].isin(signal_times)].copy()
        pressure_signals["Signal"] = pressure_signals.accumulation_buy.map({True: "Breakout buy", False: "Exit signal"})
        pressure_line = alt.Chart(pressure).mark_line(color="#0f766e").encode(
            x=alt.X("Time:T", title="Time"), y=alt.Y("accum_cmf20:Q", title="20-day CMF"))
        pressure_level = alt.Chart(pd.DataFrame({"level": [cmf_threshold]})).mark_rule(
            color="#64748b", strokeDash=[5, 4]).encode(y="level:Q")
        pressure_points = alt.Chart(pressure_signals).mark_point(filled=True, size=80).encode(
            x="Time:T", y="accum_cmf20:Q",
            shape=alt.Shape("Signal:N", scale=alt.Scale(domain=["Breakout buy", "Exit signal"],
                                                          range=["triangle-up", "triangle-down"]), legend=None),
            color=alt.Color("Signal:N", scale=alt.Scale(domain=["Breakout buy", "Exit signal"],
                                                           range=["#16a34a", "#dc2626"]), legend=None),
            tooltip=[alt.Tooltip("Time:T"), alt.Tooltip("Signal:N"),
                     alt.Tooltip("accum_cmf20:Q", format=".2f")])
        fill_sides = ["Buy", "Sell"]
        fills = (trades_df[trades_df.Side.isin(fill_sides)].copy() if not trades_df.empty
                 else pd.DataFrame(columns=["Side", "Signal time", "Fill time", "Price"]))
        if not fills.empty:
            fills["CMF"] = fills["Signal time"].map(result.accum_cmf20)
            fill_points = alt.Chart(fills).mark_point(shape="circle", filled=True, size=55, color="#111827").encode(
                x=alt.X("Fill time:T", title="Time"), y="CMF:Q",
                tooltip=[alt.Tooltip("Fill time:T"), alt.Tooltip("Side:N"),
                         alt.Tooltip("Price:Q", format="$.2f")])
            pressure_chart = pressure_line + pressure_level + pressure_points + fill_points
        else:
            pressure_chart = pressure_line + pressure_level + pressure_points
        st.subheader("Accumulation proxy (CMF), signals, and fills")
        st.altair_chart(pressure_chart, use_container_width=True)
    st.subheader("Equity curve")
    curves = pd.DataFrame({"Strategy": result.equity, "Buy and hold": benchmark})
    st.line_chart(curves)
    if not trades_df.empty:
        st.subheader("Fills")
        st.dataframe(trades_df.iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)
