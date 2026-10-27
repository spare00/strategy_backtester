import math
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from macd_backtest_core import run_supertrend_policy, supertrend_indicators


def synthetic_bars() -> pd.DataFrame:
    close = [10, 10, 10.5, 12, 13, 12, 11, 10, 9, 10, 12, 14, 15, 14, 13]
    return pd.DataFrame({
        "open": close,
        "high": [value + 0.5 for value in close],
        "low": [value - 0.5 for value in close],
        "close": close,
    }, index=pd.date_range("2024-01-01", periods=len(close), freq="D", tz="America/New_York"))


class SuperTrendBacktestTests(unittest.TestCase):
    def test_atr_seed_band_recurrence_and_direction_flips(self):
        result = supertrend_indicators(synthetic_bars(), atr_length=2, multiplier=1)
        self.assertTrue(math.isnan(result.atr.iloc[0]))
        self.assertEqual(result.atr.iloc[1], 1.0)  # Seed is mean(TR[0:2]).
        self.assertEqual(result.supertrend_upper.iloc[1], 11.0)
        self.assertEqual(result.supertrend_lower.iloc[1], 9.0)
        self.assertEqual(result.supertrend_upper.iloc[2], 11.0)  # Upper band carries forward.
        self.assertEqual(result.supertrend_lower.iloc[2], 9.5)  # Lower band ratchets upward.
        self.assertFalse(result.supertrend_bullish.iloc[1])  # Initial direction is bearish.
        self.assertFalse(result.supertrend_buy.iloc[1])
        self.assertTrue(result.supertrend_buy.iloc[3])
        self.assertTrue(result.supertrend_sell.iloc[6])
        self.assertTrue(result.supertrend_buy.iloc[10])

    def test_signals_fill_next_open_and_do_not_repeat_while_direction_holds(self):
        bars = synthetic_bars()
        result, trades, _ = run_supertrend_policy(
            bars, 1000, 0, 0, bars.index[0], atr_length=2, multiplier=1,
        )
        buys = [trade for trade in trades if trade["Side"] == "Buy"]
        sells = [trade for trade in trades if trade["Side"] == "Sell"]
        self.assertEqual(len(buys), 2)
        self.assertEqual(len(sells), 1)
        self.assertEqual((buys[0]["Signal time"], buys[0]["Fill time"], buys[0]["Price"]),
                         (bars.index[3], bars.index[4], bars.open.iloc[4]))
        self.assertEqual((sells[0]["Signal time"], sells[0]["Fill time"], sells[0]["Price"]),
                         (bars.index[6], bars.index[7], bars.open.iloc[7]))
        self.assertEqual((buys[1]["Signal time"], buys[1]["Fill time"], buys[1]["Price"]),
                         (bars.index[10], bars.index[11], bars.open.iloc[11]))
        self.assertTrue(result.invested.loc[bars.index[6]])  # No intraday/EOD flattening.
        self.assertFalse(result.invested.loc[bars.index[7]])

    def test_indicator_prefix_is_invariant_to_future_bars(self):
        bars = synthetic_bars()
        prefix = supertrend_indicators(bars.iloc[:11], 2, 1)
        full = supertrend_indicators(bars, 2, 1).iloc[:11]
        columns = ["atr", "supertrend_upper", "supertrend_lower", "supertrend_line",
                   "supertrend_bullish", "supertrend_buy", "supertrend_sell"]
        pd.testing.assert_frame_equal(prefix[columns], full[columns])
        changed_future = bars.copy()
        changed_future.iloc[11:, changed_future.columns.get_loc("high")] += 100
        changed_future.iloc[11:, changed_future.columns.get_loc("close")] += 100
        changed = supertrend_indicators(changed_future, 2, 1).iloc[:11]
        pd.testing.assert_frame_equal(prefix[columns], changed[columns])

    def test_app_exposes_supertrend_and_keeps_interval_and_dates(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        app = AppTest.from_file(str(app_path)).run()
        self.assertFalse(app.exception)
        self.assertIn("SuperTrend (ATR arrows)", app.selectbox(key="policy").options)
        self.assertIn("Daily 50/200-day SMA cross", app.selectbox(key="policy").options)
        self.assertEqual(app.selectbox(key="bar_interval").value, 10)
        start, end = app.date_input(key="date_start").value, app.date_input(key="date_end").value
        app.selectbox(key="policy").set_value("Close above 200-day SMA").run()
        self.assertEqual(app.selectbox(key="bar_interval").value, "Daily")
        app.selectbox(key="bar_interval").set_value(15).run()
        self.assertEqual(app.selectbox(key="policy").value, "MACD turning up below zero")
        app.selectbox(key="policy").set_value("SuperTrend (ATR arrows)").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="bar_interval").value, 15)
        self.assertEqual(app.number_input(key="supertrend_atr_length").value, 7)
        self.assertEqual(app.number_input(key="supertrend_multiplier").value, 3.0)
        app.selectbox(key="bar_interval").set_value("Daily").run()
        self.assertEqual(app.selectbox(key="policy").value, "SuperTrend (ATR arrows)")
        app.selectbox(key="bar_interval").set_value(5).run()
        self.assertEqual(app.selectbox(key="policy").value, "SuperTrend (ATR arrows)")
        self.assertEqual((app.date_input(key="date_start").value, app.date_input(key="date_end").value),
                         (start, end))


if __name__ == "__main__":
    unittest.main()
